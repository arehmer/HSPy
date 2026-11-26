# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 16:13:39 2024

@author: rehmer
"""
import os
import time
import cv2
import socket
import struct

from typing import List, Optional, Sequence, Tuple, Union

import queue
from queue import Queue
import threading
from threading import Condition

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import pickle as pkl


from collections import deque

from hspy.ipc.threads_base import (WThread_R1, RThread, RThread_R1, WThread,
                                   RWThread, RWThread_R1)
from hspy.LuT import LuT
from hspy.tparray import TPArray

from hspy.drivers._frame import AUX_FROM_DATACOLS
from hspy.drivers.i2c import I2C_HTPA32x32d
from hspy.drivers.spi import SPI_HTPA160x120dR1
from hspy.drivers.udp import UDP_HTPA
from hspy.drivers.usb import USB_HTPA

from hspy.ipc.plot_utils import TimeSeriesPlot


class USB_Client(WThread_R1):
    """
    Runs a USB_HTPA driver in its own thread: starts the device's frame
    stream and repeatedly puts acquired frames into write_buffer.

    As with the other clients, the driver is constructed (and therefore
    opened and initialised) by the caller and handed in -- a thread shares
    the parent process's memory, so this is safe. The thread is the only
    one that talks to the device while it runs: the stream is started in
    _setup() and stopped in _teardown(), both of which run inside this
    thread.

    Losing the device (unplugged, cable fault, or no frame for lost_after_s
    seconds) is reported ONCE; the thread then puts one final item
    {'success': False, 'device_lost': True, ...} into write_buffer, sets
    the device_lost event and ends. Other errors are printed once per
    distinct message, not on every attempt.

    Every item put into write_buffer is the driver's frame dictionary (see
    hspy.drivers._frame -- 'frame', 'frame_unit', 'DevID', 'serial', 't',
    the auxiliary sensor data, ...) plus the two keys this thread adds:
    'success' and 'image_id'. An item with success=False carries only
    those two, 'DevID' and 'serial'.
    """

    def __init__(self,
                 usb_driver,
                 write_buffer,
                 stream_cmd: bytes = b't',
                 name: str = 'usb_thread',
                 lost_after_s: float = 5.0,
                 **kwargs):
        """
        Parameters
        ----------
        usb_driver : USB_HTPA
            Opened and initialised driver instance. Everything about how a
            frame is acquired -- and the DevID the frames are labelled
            with -- is set on the driver, not here.
        write_buffer : queue.Queue
            Buffer that acquired frames are put into.
        stream_cmd : bytes
            Command that starts the stream. b't' (default) streams pixel
            values the device has already converted; b'K' streams them in
            dK, which the driver then reports as frame['pix_dK'] with
            frame_unit 'dK'.
        name : str
            Thread name.
        lost_after_s : float
            If no frame could be read for this long (only errors), the
            device is considered lost even if the driver did not report a
            disconnect.

        Returns
        -------
        None.

        """

        self.usb_driver = usb_driver
        self.stream_cmd = stream_cmd
        self.lost_after_s = lost_after_s

        # Set when the device is lost; check with usb_thread.device_lost.is_set()
        self.device_lost = threading.Event()

        super().__init__(name=name,
                         write_buffer=write_buffer,
                         **kwargs)

    # -- Thread body (the loop itself lives in WThread_R1.run) ---------------

    def _setup(self):

        # Set image_id counter
        self.image_id = 0

        # Error bookkeeping: print each distinct error once
        self._last_error = None
        self._n_errors = 0

        self.t0 = time.time()
        self._t_last_frame = time.monotonic()

        # Start continuous frame stream
        try:
            resp = self.usb_driver.start_stream(self.stream_cmd)
        except Exception as e:
            # Device gone / not usable: report once and end the thread (the
            # loop in run() does not start because _device_lost sets _exit)
            self._device_lost(f"could not start the stream: {e}")
            self._put(self._lost_item())
            return

        print(f"[{self.name}] Stream started: {resp!r}")

    def _target(self):

        try:
            # Acquire one frame from the stream
            data = self.usb_driver.acquire_frame()

        except Exception as e:

            # Device gone, or errors only for too long: report once, hand a
            # final 'device_lost' item to the consumer, end the loop
            if not self.usb_driver.is_connected:
                self._device_lost(str(e))
                return self._lost_item()
            if time.monotonic() - self._t_last_frame > self.lost_after_s:
                self._device_lost(f"no frame for {self.lost_after_s:g} s "
                                  f"(last error: {e})")
                return self._lost_item()

            # Other errors: print each distinct message once, don't spin
            self._n_errors += 1
            message = str(e)
            if message != self._last_error:
                print(f"[{self.name}] {message}")
                self._last_error = message
            time.sleep(0.05)

            result = self._failed_item()

        else:
            if self._n_errors:
                print(f"[{self.name}] Receiving frames again "
                      f"({self._n_errors} failed attempt(s) before).")
                self._n_errors = 0
                self._last_error = None
            self._t_last_frame = time.monotonic()

            # The driver's frame dictionary, plus the loop state it knows
            # nothing about. 'frame', 'DevID', 'serial' and 't' all come
            # from the driver -- see hspy.drivers._frame.
            result = dict(data)
            result['success'] = True

        # Store image_id
        result['image_id'] = self.image_id

        # Increment image_id
        self.image_id = self.image_id + 1

        return result

    def _device_lost(self, reason: str):
        """Report the loss once, flag it for the caller, end the loop."""
        print(f"[{self.name}] Connection to device {self.usb_driver.serial_number} "
              f"LOST: {reason}. Continuing without it.")
        self.device_lost.set()
        self._exit.set()

    def _failed_item(self) -> dict:
        """Item for one failed attempt: no frame, but still identifiable."""
        return {'success': False,
                'DevID': self.usb_driver.DevID,
                'serial': self.usb_driver.serial_number}

    def _lost_item(self) -> dict:
        """Final item for the consumer: tells it that no more frames come."""
        item = self._failed_item()
        item['device_lost'] = True
        item['image_id'] = self.image_id
        return item

    def _teardown(self):
        # Runs inside this thread once the loop has exited (stop() was
        # called, or the device was lost), so the device is never accessed
        # from two threads at once. Closing the driver is left to whoever
        # created it (same ownership as with I2C_Client).
        if self.usb_driver.is_streaming and self.usb_driver.is_connected:
            try:
                resp, _ = self.usb_driver.stop_stream()
                print(f"[{self.name}] Stream stopped: {resp!r}")
            except Exception as e:
                print(f"[{self.name}] Stopping stream failed: {e}")


class UDP_Client(WThread_R1):
    """
    Runs a UDP_HTPA driver in its own thread: starts the device's
    continuous bytestream and repeatedly puts assembled frames into
    write_buffer.

    Same shape as USB_Client: the driver is constructed by the caller and
    handed in, the stream is started in _setup() and stopped in
    _teardown(), both inside this thread. Discovering and binding the
    device is the driver's job now -- UDP_HTPA does it in its own open(),
    so this class is only the loop and the watchdog around it.

    Losing the device is reported ONCE; the thread then puts one final
    item {'success': False, 'device_lost': True, ...} into write_buffer,
    sets the device_lost event and ends. "Lost" means unplugged, network
    down, no first frame within first_frame_after_s of the stream
    starting, or no complete frame for lost_after_s after the last good
    one. Other errors are printed once per distinct message, not on every
    attempt.

    Every item put into write_buffer is the driver's frame dictionary (see
    hspy.drivers._frame) plus 'success' and 'image_id'.
    """

    def __init__(self,
                 udp_driver: UDP_HTPA,
                 write_buffer: Queue,
                 name: str = 'udp_thread',
                 stream_cmd: bytes = None,
                 lost_after_s: float = 5.0,
                 first_frame_after_s: float = 15.0,
                 **kwargs):
        """
        Parameters
        ----------
        udp_driver : UDP_HTPA
            Driver instance, already bound to its device (UDP_HTPA does
            that in its constructor). Which device that is, and how its
            frames look, is set on the driver.
        write_buffer : Queue
            Buffer that assembled frames are put into.
        name : str
            Thread name. Give each thread a distinct name when running
            several devices.
        stream_cmd : bytes, optional
            Command that starts the continuous bytestream. None (default)
            leaves the driver's own stream_cmd alone; pass one to override
            it.
        lost_after_s : float
            If no complete frame could be assembled for this long after
            the last good one (only errors / socket timeouts), the device
            is considered lost.
        first_frame_after_s : float
            Budget for the FIRST frame, measured from the moment the
            stream command was sent. A device can need a while to start
            sending, which is not the same failure as a device that was
            streaming and stopped, so it gets its own, larger budget. The
            socket timeout used while waiting for it is the driver's
            first_frame_timeout.

        Returns
        -------
        None.

        """

        self.udp_driver = udp_driver
        self.stream_cmd = stream_cmd
        self.lost_after_s = lost_after_s
        self.first_frame_after_s = first_frame_after_s

        # Set when the device is lost; check with udp_thread.device_lost.is_set()
        self.device_lost = threading.Event()

        super().__init__(name=name,
                         write_buffer=write_buffer,
                         **kwargs)

    @property
    def DevID(self) -> int:
        """Device number of the bound device, as the driver reports it."""
        return self.udp_driver.DevID

    def _setup(self):

        # Set image_id counter
        self.image_id = 0

        # Error bookkeeping: print each distinct error once
        self._last_error = None
        self._n_errors = 0

        self.t0 = time.time()

        # Watchdog references. Stamped once the stream is really running;
        # _t_last_frame stays None until a frame has actually arrived, so
        # that the wait for the first one is measured against its own,
        # larger budget.
        self._t_stream_start = None
        self._t_last_frame = None

        # A read in progress delays both the watchdog and stop(), so the
        # driver never waits longer for the first frame than that frame's
        # whole budget.
        self.udp_driver.first_frame_timeout = min(
            self.udp_driver.first_frame_timeout,
            self.first_frame_after_s)

        try:
            self.udp_driver.start_stream(self.stream_cmd)

            # From here on the device should be sending. This is what the
            # watchdog measures against until the first frame arrives.
            self._t_stream_start = time.monotonic()
        except Exception as e:
            # Device not reachable: report once and end the thread (the
            # loop in run() does not start because _device_lost sets _exit)
            self._device_lost(f"could not start the stream: {e}")
            self._put(self._lost_item())
            return

        print(f"[{self.name}] Stream started for device {self.DevID}.")

    def _target(self):

        try:
            # Try to assemble a frame from the udp packages. The driver
            # gives the first one the longer first_frame_timeout by
            # itself: the device needs a moment to act on the stream
            # command, which is not an error.
            data = self.udp_driver.acquire_frame()

        except Exception as e:

            # Nothing but errors for too long: the device is gone (unplugged,
            # network down, powered off). Report once, hand a final
            # 'device_lost' item to the consumer, end the loop.
            if self._t_last_frame is None:
                # No frame has arrived yet, so there is nothing to be
                # silent *since* -- the stream may simply not have started
                # yet. Measured from the stream command instead, against
                # its own budget.
                if (self._t_stream_start is not None
                        and time.monotonic() - self._t_stream_start
                        > self.first_frame_after_s):
                    self._device_lost(
                        f"no frame within {self.first_frame_after_s:g} s of "
                        f"starting the stream (last error: {e})")
                    return self._lost_item()

            elif time.monotonic() - self._t_last_frame > self.lost_after_s:
                self._device_lost(f"no frame for {self.lost_after_s:g} s "
                                  f"(last error: {e})")
                return self._lost_item()

            # Other errors (a single socket timeout, lost packets, ...):
            # print each distinct message once, and don't spin
            self._n_errors += 1
            message = str(e)
            if message != self._last_error:
                print(f"[{self.name}] {message}")
                self._last_error = message
            time.sleep(0.05)

            result = self._failed_item()

        else:
            if self._n_errors:
                print(f"[{self.name}] Receiving frames again "
                      f"({self._n_errors} failed attempt(s) before).")
                self._n_errors = 0
                self._last_error = None
            self._t_last_frame = time.monotonic()

            # The driver's frame dictionary, plus the loop state it knows
            # nothing about
            result = dict(data)
            result['success'] = True

        # Store image_id
        result['image_id'] = self.image_id

        # Increment image_id
        self.image_id = self.image_id + 1

        return result

    def _device_lost(self, reason: str):
        """Report the loss once, flag it for the caller, end the loop."""
        print(f"[{self.name}] Connection to device {self.DevID} LOST: "
              f"{reason}. Continuing without it.")
        self.device_lost.set()
        self._exit.set()

    def _failed_item(self) -> dict:
        """Item for one failed attempt: no frame, but still identifiable."""
        return {'success': False,
                'DevID': self.udp_driver.DevID,
                'serial': self.udp_driver.serial}

    def _lost_item(self) -> dict:
        """Final item for the consumer: tells it that no more frames come."""
        item = self._failed_item()
        item['device_lost'] = True
        item['image_id'] = self.image_id
        return item

    def _teardown(self):
        # Runs inside this thread once the loop has exited: stop the stream,
        # so the device is not left streaming into a socket nobody reads.
        # Releasing the device and closing the socket is driver.close(),
        # which belongs to whoever created the driver (same ownership as
        # with USB_Client).
        try:
            self.udp_driver.stop_stream()
            print(f"[{self.name}] Stream stopped.")
        except Exception as e:
            print(f"[{self.name}] Stopping stream failed: {e}")


class I2C_Client(WThread_R1):
    """
    Runs an I2C_HTPA32x32d driver in its own thread. The driver class only
    knows how to talk to the device (I2C bus reads, calibration, LuT
    mapping), it holds no threading state of its own; this class owns the
    loop and the write_buffer and calls the driver's acquire_frame() once
    per iteration.

    Every item put into write_buffer is the driver's frame dictionary (see
    hspy.drivers._frame) plus 'success' and 'image_id'.
    """

    def __init__(self,
                 i2c_driver: I2C_HTPA32x32d,
                 write_buffer: Queue,
                 name: str = 'i2c_thread',
                 **kwargs):
        """
        Parameters
        ----------
        i2c_driver : I2C_HTPA32x32d
            Driver instance providing acquire_frame(). Must already be
            open()'ed and init()'ed (this is done in I2C_HTPA32x32d.__init__).
            Everything about how a frame is acquired -- active_vdd,
            blind_vdd, applyCalib, keep_raw -- is set on the
            driver, not here.
        write_buffer : Queue
            Buffer that processed frames are put into for the downstream
            thread to consume.
        name : str
            Thread name.

        Returns
        -------
        None.

        """

        # Set I2C driver object as attribute
        self.i2c_driver = i2c_driver

        super().__init__(name=name,
                         write_buffer=write_buffer,
                         **kwargs)

    def _setup(self):

        # Set image_id counter
        self.image_id = 0

        # Error bookkeeping: print each distinct error once
        self._last_error = None
        self._n_errors = 0

        # Set time
        self.t0 = time.time()

    def _target(self):

        try:
            # Acquire and process one frame from the sensor
            data = self.i2c_driver.acquire_frame()

        except Exception as e:
            # Print each distinct message once rather than on every
            # iteration, and don't spin
            self._n_errors += 1
            message = str(e)
            if message != self._last_error:
                print(f"[{self.name}] {message}")
                self._last_error = message
            time.sleep(0.05)

            result = {'success': False,
                      'DevID': self.i2c_driver.DevID,
                      'serial': ''}

        else:
            if self._n_errors:
                print(f"[{self.name}] Acquiring frames again "
                      f"({self._n_errors} failed attempt(s) before).")
                self._n_errors = 0
                self._last_error = None

            # The driver's frame dictionary, plus the loop state it knows
            # nothing about
            result = dict(data)
            result['success'] = True

        # Store image_id
        result['image_id'] = self.image_id

        # Increment image_id
        self.image_id = self.image_id + 1

        return result

    def _teardown(self):
        # Runs inside this thread once the loop has exited. Putting the
        # sensor to sleep and closing the bus is driver.close(), which
        # belongs to whoever created the driver (same ownership as with
        # USB_Client).
        pass


class SPI_Client(WThread_R1):
    """
    Runs an SPI_HTPA160x120dR1 driver in its own thread. The I2C/SPI
    counterpart of USB_Client, and the same shape as I2C_Client: the
    driver does the device work, this class owns the loop and the
    write_buffer and calls acquire_frame() once per iteration.

    Every item put into write_buffer is the driver's frame dictionary (see
    hspy.drivers._frame) plus 'success' and 'image_id'.
    """

    def __init__(self,
                 spi_driver: SPI_HTPA160x120dR1,
                 write_buffer: Queue,
                 name: str = 'spi_thread',
                 rt_priority: Optional[int] = None,
                 **kwargs):
        """
        Parameters
        ----------
        spi_driver : SPI_HTPA160x120dR1
            Driver instance providing acquire_frame(). Must already be
            open()'ed and init()'ed (this is done in its __init__).
            Everything about how a frame is acquired -- eloff_every,
            applyCalib, keep_raw -- is set on the driver, not here.
        write_buffer : Queue
            Buffer that processed frames are put into for the downstream
            thread to consume.
        name : str
            Thread name.
        rt_priority : int, optional
            If given, this thread (and only this one) runs with the Linux
            real-time policy SCHED_FIFO at this priority (1..99; keep it
            below 50, the priority of the kernel's threaded interrupts).
            It then gets the CPU as soon as a conversion wait ends, which
            keeps the time between conversion STARTs constant: a varying
            idle time before a conversion shifts the following blocks
            (flicker of the top and bottom rows). Needs permission, e.g. a
            line '<user> - rtprio 50' in /etc/security/limits.d/; without
            it a warning is printed and the thread runs at normal priority.

        Returns
        -------
        None.

        """

        # Set SPI driver object as attribute
        self.spi_driver = spi_driver
        self.rt_priority = rt_priority

        super().__init__(name=name,
                         write_buffer=write_buffer,
                         **kwargs)

    def _setup(self):

        # Runs inside this thread: on Linux, pid 0 is the calling thread,
        # so only this thread becomes real-time
        if self.rt_priority is not None:
            try:
                os.sched_setscheduler(0, os.SCHED_FIFO,
                                      os.sched_param(self.rt_priority))
            except (AttributeError, OSError) as e:
                print(f"[{self.name}] Could not set real-time priority "
                      f"{self.rt_priority} ({e}); running at normal "
                      f"priority. Conversion timing and thus the top and "
                      f"bottom rows may jitter.")

        # Set image_id counter
        self.image_id = 0

        # Error bookkeeping: print each distinct error once
        self._last_error = None
        self._n_errors = 0

        # Set time
        self.t0 = time.time()

    def _acquire(self) -> dict:
        """One frame from the driver; SPI_AcquisitionThread reads raw ones."""
        return self.spi_driver.acquire_frame()

    def _target(self):

        try:
            # Acquire and process one frame from the sensor
            data = self._acquire()

        except Exception as e:
            # Print each distinct message once rather than on every
            # iteration, and don't spin
            self._n_errors += 1
            message = str(e)
            if message != self._last_error:
                print(f"[{self.name}] {message}")
                self._last_error = message
            time.sleep(0.05)

            result = {'success': False,
                      'DevID': self.spi_driver.DevID,
                      'serial': ''}

        else:
            if self._n_errors:
                print(f"[{self.name}] Acquiring frames again "
                      f"({self._n_errors} failed attempt(s) before).")
                self._n_errors = 0
                self._last_error = None

            # The driver's frame dictionary, plus the loop state it knows
            # nothing about
            result = dict(data)
            result['success'] = True

        # Store image_id
        result['image_id'] = self.image_id

        # Increment image_id
        self.image_id = self.image_id + 1

        return result

    def _teardown(self):
        # Runs inside this thread once the loop has exited. Putting the
        # sensor to sleep and closing the bus is driver.close(), which
        # belongs to whoever created the driver (same ownership as with
        # USB_Client).
        pass


class SPI_AcquisitionThread(SPI_Client):
    """
    Threading version of SPI_AcquisitionProcess: first stage of a two-stage
    SPI pipeline that only talks to the sensor.

    Writes the raw bytes of every frame (SPI_HTPA160x120dR1.acquire_raw(),
    about 42 kB) into write_buffer and starts the next frame right away; an
    SPI_ProcessingThread reading that buffer turns them into frame
    dictionaries. The SPI transfers and the sleeps during a conversion
    release the GIL, so processing frame k overlaps acquiring frame k+1::

        acq  = SPI_AcquisitionThread(spi_driver, raw_buffer)
        proc = SPI_ProcessingThread(spi_driver, raw_buffer, frame_buffer)

    Every item put into write_buffer is the dict of acquire_raw() plus
    'success' and 'image_id', or {'success': False, ...} like SPI_Client on
    a failed frame.
    """

    def __init__(self,
                 spi_driver: SPI_HTPA160x120dR1,
                 write_buffer: Queue,
                 name: str = 'spi_acquisition_thread',
                 **kwargs):

        super().__init__(spi_driver=spi_driver,
                         write_buffer=write_buffer,
                         name=name,
                         **kwargs)

    def _acquire(self) -> dict:
        return self.spi_driver.acquire_raw()


class SPI_ProcessingThread(RWThread_R1):
    """
    Threading version of SPI_ProcessingProcess: second stage of a two-stage
    SPI pipeline, see SPI_AcquisitionThread. Reads raw frames from
    read_buffer and writes the frame dictionaries of hspy.drivers._frame
    (plus 'success' and 'image_id') into write_buffer - the same items
    SPI_Client writes.
    """

    def __init__(self,
                 spi_driver: SPI_HTPA160x120dR1,
                 read_buffer: Queue,
                 write_buffer: Queue,
                 lut_csv_path=None,
                 name: str = 'spi_processing_thread',
                 **kwargs):
        """
        Parameters
        ----------
        spi_driver : SPI_HTPA160x120dR1
            The open driver the SPI_AcquisitionThread uses. This thread
            works on a hardware-free copy of it (calibration, LuT,
            options), made here, so the two threads share no driver state.
        read_buffer : Queue
            The write_buffer of the SPI_AcquisitionThread.
        write_buffer : Queue
            Buffer the processed frames are put into.
        lut_csv_path : str or Path, optional
            If given, a LuT is loaded from this CSV and replaces the one the
            driver found for the table number in its flash.
        name : str
            Thread name.
        """

        # Same copy SPI_AcquisitionProcess sends to SPI_ProcessingProcess
        # (__getstate__ leaves out the SPI handle and the GPIO callable)
        self.spi_processor = pkl.loads(pkl.dumps(spi_driver,
                                                 protocol=pkl.HIGHEST_PROTOCOL))

        if lut_csv_path is not None:
            lut = LuT()
            lut.from_csv(lut_csv_path)
            self.spi_processor.LuT = lut

        # Error bookkeeping: print each distinct error once
        self._last_error = None
        self._n_errors = 0

        super().__init__(name=name,
                         read_buffer=read_buffer,
                         write_buffer=write_buffer,
                         **kwargs)

    def _target(self):

        item = self._get()

        # A failed acquisition is passed on unchanged
        if not item.get('success', False):
            return item

        image_id = item.pop('image_id')
        del item['success']

        try:
            data = self.spi_processor.process_raw(item)

        except Exception as e:
            # Print each distinct message once rather than on every frame
            self._n_errors += 1
            message = str(e)
            if message != self._last_error:
                print(f"[{self.name}] {message}")
                self._last_error = message

            result = {'success': False,
                      'DevID': self.spi_processor.DevID,
                      'serial': ''}

        else:
            if self._n_errors:
                print(f"[{self.name}] Processing frames again "
                      f"({self._n_errors} failed frame(s) before).")
                self._n_errors = 0
                self._last_error = None

            result = dict(data)
            result['success'] = True

        result['image_id'] = image_id

        return result



# Header put in front of every UDP datagram sent by UDP_PickleServer /
# consumed by UDP_PickleClient: message id, index of this chunk, total
# number of chunks the message was split into.
_PICKLE_HEADER_FMT = '!IHH'
_PICKLE_HEADER_SIZE = struct.calcsize(_PICKLE_HEADER_FMT)

# Stay safely below the 65507 byte payload limit of a UDP datagram
_PICKLE_MAX_CHUNK = 60000


class UDP_PickleServer(RThread_R1):
    """
    Thread that reads arbitrary (picklable) data - e.g. a dict containing
    a numpy array and a pandas DataFrame - from an upstream buffer and
    sends it out via UDP to a single peer running a UDP_PickleClient.

    Since a pickled payload can easily exceed the size of a single UDP
    datagram, it is split into numbered chunks that are reassembled by
    the receiving UDP_PickleClient. As with any data sent over UDP,
    delivery is not guaranteed: a message of which one or more chunks
    are lost in transit is silently dropped by the receiver.
    """

    def __init__(self,
                 name:str,
                 read_buffer:Queue,
                 target_ip:str,
                 target_port:int,
                 **kwargs):

        self.target_addr = (target_ip, target_port)

        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self._msg_id = 0

        super().__init__(name = name,
                         read_buffer = read_buffer,
                         **kwargs)

    def _target(self):

        # Get data (e.g. a dict with a numpy array / pandas DataFrame)
        # from the upstream buffer
        data = self._get()

        # Serialize it
        payload = pkl.dumps(data)

        # Split the payload into chunks that fit into a single UDP
        # datagram
        chunks = [payload[i:i + _PICKLE_MAX_CHUNK]
                 for i in range(0, len(payload), _PICKLE_MAX_CHUNK)] or [b'']

        chunk_count = len(chunks)

        # Send every chunk in its own datagram, prefixed with a header
        # that allows the receiver to reassemble the message
        for chunk_index, chunk in enumerate(chunks):
            header = struct.pack(_PICKLE_HEADER_FMT,
                                 self._msg_id,
                                 chunk_index,
                                 chunk_count)
            self._socket.sendto(header + chunk, self.target_addr)

        # Increment (and wrap) the message id
        self._msg_id = (self._msg_id + 1) % 2**32

        return None

    def stop(self):
        super().stop()
        self._socket.close()


class UDP_PickleClient(WThread_R1):
    """
    Thread that receives data sent by a UDP_PickleServer instance,
    reassembles and unpickles it, and writes the resulting object
    (e.g. a dict containing a numpy array and a pandas DataFrame) into
    a buffer for downstream threads to consume.
    """

    def __init__(self,
                 name:str,
                 write_buffer:Queue,
                 listen_ip:str = '0.0.0.0',
                 listen_port:int = 0,
                 timeout:float = 1.0,
                 **kwargs):

        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind((listen_ip, listen_port))
        self._socket.settimeout(timeout)

        # Chunks of messages that have not been fully received yet,
        # keyed by message id
        self._pending = {}

        super().__init__(name = name,
                         write_buffer = write_buffer,
                         **kwargs)

    @property
    def listen_addr(self):
        return self._socket.getsockname()

    def _target(self):

        # Keep listening until either a full message has been
        # reassembled or the thread is stopped
        while not self._exit.is_set():

            try:
                packet, _addr = self._socket.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                # Socket was closed (e.g. during shutdown) while blocked in recvfrom
                break

            header = packet[:_PICKLE_HEADER_SIZE]
            chunk = packet[_PICKLE_HEADER_SIZE:]

            msg_id, chunk_index, chunk_count = struct.unpack(_PICKLE_HEADER_FMT,
                                                              header)

            chunks = self._pending.setdefault(msg_id, {})
            chunks[chunk_index] = chunk

            # Once every chunk of this message has arrived, reassemble
            # it in order and hand it back to be put in the write buffer
            if len(chunks) == chunk_count:
                payload = b''.join(chunks[i] for i in range(chunk_count))
                del self._pending[msg_id]
                return pkl.loads(payload)

        return None

    def stop(self):
        super().stop()
        self._socket.close()



 
 
class ImshowDisplay(threading.Thread):
    """
    The only thread that calls OpenCV HighGUI functions. Other threads call
    its namedWindow / imshow / setWindowTitle / destroyWindow methods
    (same signatures as cv2), which only queue the request and return
    immediately; this thread carries them out.
 
    Only the newest image per window is shown: if Imshow threads deliver
    faster than the display can draw, older images are skipped instead of
    piling up -- the Imshow threads are never blocked.
 
    New windows are tiled across the screen (window_size, columns), so
    several windows don't open on top of each other.
    """
 
    def __init__(self,
                 name: str = 'imshow_display',
                 window_size: Optional[Tuple[int, int]] = (640, 480),
                 columns: int = 3,
                 origin: Tuple[int, int] = (20, 20),
                 spacing: Tuple[int, int] = (20, 60),
                 period_s: float = 0.01):
        """
        Parameters
        ----------
        window_size : (width, height) every new window is resized to, or
                      None to keep OpenCV's default size (and position).
        columns     : windows per row when tiling.
        origin      : screen position of the first window.
        spacing     : (x, y) gap between windows (y includes the title bar).
        period_s    : max. time between two waitKey() calls, i.e. how often
                      windows are serviced (repaint, move, resize) even
                      when no new image arrives.
        """
        super().__init__(name=name, daemon=True)
 
        self.window_size = window_size
        self.columns = columns
        self.origin = origin
        self.spacing = spacing
        self.period_s = period_s
 
        self._lock = threading.Lock()
        self._ops = deque()            # window operations, in call order
        self._frames = {}              # window name -> newest image not yet shown
        self._wakeup = threading.Event()
        self._exit = threading.Event()
 
        # Only touched by this thread
        self._open = []                # windows created, in creation order
        self._n_created = 0
 
    # ── cv2-compatible API (called from any thread) ──────────────────────
 
    def namedWindow(self, winname: str, flags: int = cv2.WINDOW_NORMAL) -> None:
        self._queue_op(('create', winname, flags))
 
    def imshow(self, winname: str, mat: np.ndarray) -> None:
        with self._lock:
            self._frames[winname] = mat          # replaces an older, unshown image
        self._wakeup.set()
 
    def waitKey(self, delay: int = 0) -> int:
        # Events are processed by this thread's own waitKey() calls
        return -1
 
    def setWindowTitle(self, winname: str, title: str) -> None:
        self._queue_op(('title', winname, title))
 
    def destroyWindow(self, winname: str) -> None:
        with self._lock:
            self._frames.pop(winname, None)
            self._ops.append(('destroy', winname))
        self._wakeup.set()
 
    def stop(self) -> None:
        self._exit.set()
        self._wakeup.set()
 
    # ── Display loop (this thread only) ──────────────────────────────────
 
    def run(self):
        try:
            while not self._exit.is_set():
                self._wakeup.wait(self.period_s)
                self._wakeup.clear()
 
                with self._lock:
                    ops = list(self._ops)
                    self._ops.clear()
                    frames, self._frames = self._frames, {}
 
                for op in ops:
                    self._apply(op)
 
                for winname, mat in frames.items():
                    if winname in self._open:    # ignore images for closed windows
                        cv2.imshow(winname, mat)
 
                cv2.waitKey(1)
        finally:
            for winname in list(self._open):
                self._apply(('destroy', winname))
            cv2.waitKey(1)
 
    def _queue_op(self, op) -> None:
        with self._lock:
            self._ops.append(op)
        self._wakeup.set()
 
    def _apply(self, op) -> None:
        kind, winname = op[0], op[1]
        try:
            if kind == 'create' and winname not in self._open:
                cv2.namedWindow(winname, op[2])
                self._place(winname)
                self._open.append(winname)
 
            elif kind == 'title' and winname in self._open:
                cv2.setWindowTitle(winname, op[2])
 
            elif kind == 'destroy' and winname in self._open:
                cv2.destroyWindow(winname)
                self._open.remove(winname)
 
        except cv2.error as e:
            print(f"[{self.name}] {kind} '{winname}' failed: {e}")
 
    def _place(self, winname: str) -> None:
        if self.window_size is None:
            return
        w, h = self.window_size
        i = self._n_created
        self._n_created += 1
        x = self.origin[0] + (i % self.columns) * (w + self.spacing[0])
        y = self.origin[1] + (i // self.columns) * (h + self.spacing[1])
        cv2.resizeWindow(winname, w, h)
        cv2.moveWindow(winname, x, y)
 
 
class Imshow(RThread_R1):
    """
    Class for plotting frames and possibly bounding boxes in a thread.
 
    display : ImshowDisplay, optional
        If given, all window operations are carried out by that thread
        (required on Linux when several Imshow threads run at once). If
        None, this thread calls cv2 directly (as before).
    """
 
    def __init__(self,
                 name :str,
                 read_buffer:Queue,
                 ArrayType:int = None,
                 SensorType:int = None,
                 display: Optional[ImshowDisplay] = None,           # CHANGED
                 plot_ptat: bool = False,                           # ADDED
                 plot_window_s: Optional[float] = None,             # ADDED
                 **kwargs):
 
        if ArrayType is None and SensorType is not None:
            self.tparray = TPArray(SensorType = SensorType)
        elif ArrayType is not None and SensorType is None:
            self.tparray = TPArray(ArrayType = ArrayType)
        else:
            raise Exception('Either ArrayType or SensorType must be provided.')
 
        self.num_pix = len(self.tparray._pix)
 
        self.window_name = kwargs.pop('window_name','Sensor stream')
 
        # Window operations go to the display thread, or directly to cv2   # CHANGED
        self._gui = cv2 if display is None else display                    # CHANGED
 
        # Line plot of data['ptat_avg'] below the image                    # ADDED
        self.plot_ptat = plot_ptat                                         # ADDED
        self.ptat_plot = TimeSeriesPlot(title='PTAT avg', unit='digits',   # ADDED
                                        window_s=plot_window_s)            # ADDED
        self._t_first = None                                               # ADDED
 
        # Set time
        self.t0 = time.time()
 
        # Call parent class
        super().__init__(name = name,
                         read_buffer = read_buffer,
                         **kwargs)
 
    def _target(self):

        # Get result from upstream thread
        data = self._get()

        # Check success flag of upstream thread
        if data['success'] == True:

            # Every driver guarantees 'frame': the best pixel stage it
            # produced, with data['frame_unit'] saying what it is (see
            # hspy.drivers._frame)
            frame = data['frame']

            # Reshape if not the proper size
            if frame.ndim == 1:
                frame = frame[0:self.num_pix]
                frame = frame.reshape(self.tparray._npsize)
 
            # convert to opencv type
            frame = cv2.normalize(frame,frame,0,255,cv2.NORM_MINMAX)
            frame = frame.astype(np.uint8)
 
            # Save to dict
            data['frame_plot'] = frame
 
            # Write annotations to dict
            if 'annot_frame' in data.keys():
                annotations = data['annot_frame'].annotations
                annotations['confirmed'] = False
                annotations.loc[annotations['label']==1,'confirmed'] = True
                data['bboxes'] = annotations
 
        else:
            # If upstream thread failed, set success flag to False
            data['success'] = False
 
        return data

    def _on_idle(self):
        """Called while no frame arrives (see run()). Keeps the window
        responsive, so it still redraws and reacts to the window manager
        while the stream has not started yet or the device is gone."""
        try:
            self._gui.waitKey(1)
        except cv2.error:
            pass

    def run(self):

        self._gui.namedWindow(self.window_name, cv2.WINDOW_NORMAL)        # CHANGED

        # Check if thread has been stopped
        while not self._exit.is_set():

            try:
                # Execute target function
                data = self._target()

            except queue.Empty:
                # No frame within get_timeout. Keep the window responsive
                # and go back to the top of the loop, where the exit flag
                # is checked. Without this the thread waits in
                # read_buffer.get() for a frame that never comes once the
                # producer upstream has ended -- stop() is then ignored,
                # join() hangs and the window is never destroyed, which is
                # why callers used to push a dummy item into the buffer
                # just to get this loop moving again.
                self._on_idle()
                continue

            # Check success flag of upstream thread
            if data['success'] is True:
 
                # Get frame (processed)
                frame = data['frame_plot']
 
                # Scale it up by a factor of 10
                sf = 10
                frame = np.repeat(np.repeat(frame, sf, axis=0), sf, axis=1)
 
                # Convert frame to RGB to be able to plot colored
                # boxes
                frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB)
 
                # Get bboxes if available
                if 'bboxes' in data.keys():
                    bboxes = data['bboxes']
 
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale              = 0.4
                    fontColor              = (255,255,0)
                    thickness              = 1
                    lineType               = 2
 
                    # Draw bounding boxes
                    for b in bboxes.index:
 
                        box = bboxes.loc[[b]]
 
                        if box['confirmed'].item() != True:
                            continue
 
                        x,y = int(box['xtl'].item()),int(box['ytl'].item()),
                        w = int(box['xbr'].item()) - int(box['xtl'].item())
                        h = int(box['ybr'].item()) - int(box['ytl'].item())
 
                        # Scale coordinates
                        x = int(sf*x); y = int(sf*y); w = int(sf*w); h = int(sf*h)
 
                        frame = cv2.rectangle(frame,
                                              (x,y),
                                              (x+w,y+h),
                                              (255,255,0),1)
 
                        # Write score if it exists
                        if box['score'].item() != -99:
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            blxy = (x+1,y+10)
                            fontScale              = 0.4
                            fontColor              = (255,255,0)
                            thickness              = 1
                            lineType               = 2
 
                            score = f"{box['score'].item():.2}"
 
                            cv2.putText(frame,
                                        score,
                                        blxy,
                                        font,
                                        fontScale,
                                        fontColor,
                                        thickness,
                                        lineType)
 
                    # Write the number of persons in the upper left corner
                    num_persons = sum(bboxes['confirmed'] == True)
 
                    cv2.putText(frame,
                                f'Number of persons: {num_persons}',
                                (0,20),
                                font,
                                fontScale,
                                fontColor,
                                thickness,
                                lineType)
 
                # PTAT plot below the image                                # ADDED
                if self.plot_ptat:                                         # ADDED
                    frame = self._add_ptat_plot(frame, data)               # ADDED
 
                self._gui.imshow(self.window_name,frame)                   # CHANGED
                self._gui.waitKey(1)                                       # CHANGED
 
            else:
                print('else')
                pass
 
            # Signal that processing on this item in the read_buffer is done
            # self.read_buffer.task_done()
 
        # The opencv window needs to be closed inside the run function,
        # otherwise a window with the same name can never be opened until
        # the console is restarted
        if self._exit.is_set():
            self._gui.destroyWindow(self.window_name)                      # CHANGED
 
    def _add_ptat_plot(self, frame, data):                                 # ADDED
        """Append data['ptat_avg'] to the history and stack the plot
        below the image. The plot is a third of the image's height.

        Every frame carries 'ptat_avg' now, but it is nan for an array type
        that has no PTAT, so the value itself decides whether there is
        anything to plot."""
        ptat_avg = data.get('ptat_avg', np.nan)
        if np.isfinite(ptat_avg):
            t = data['t'].timestamp() if hasattr(data.get('t'), 'timestamp') else time.time()
            if self._t_first is None:
                self._t_first = t
            self.ptat_plot.add(t - self._t_first, ptat_avg)
        panel = self.ptat_plot.render(frame.shape[1], frame.shape[0] // 3)
        return np.vstack([frame, panel])
 
    def stop(self):
        self._exit.set()

_REC_SUFFIX = '.pkl'
_PART_SUFFIX = '.part'
_REC_FORMAT_VERSION = 1
 
 
class ImshowRecorder(Imshow):
    """
    Imshow thread with recording. Displays every frame exactly like Imshow;
    between start_recording() and stop_recording() every successfully
    received frame dict is also kept, and written to a pickle file in the
    background once recording stops.
 
    start_recording() / stop_recording() are the "commands": call them from
    any thread (e.g. the main thread reacting to user input).
    """
 
    def __init__(self,
                 name: str,
                 read_buffer,
                 ArrayType: int = None,
                 SensorType: int = None,
                 record_dir: Union[str, Path] = '.',
                 record_keys: Optional[Sequence[str]] = None,
                 exclude_keys: Sequence[str] = ('frame_plot',),
                 save_on_stop: bool = True,
                 **kwargs):
        """
        Parameters
        ----------
        name, read_buffer, ArrayType, SensorType, **kwargs :
            As for Imshow (e.g. window_name=..., display=...). Pass a shared
            ImshowDisplay as display= whenever several ImshowRecorder /
            Imshow threads run at once (required on Linux).
        record_dir : str or Path
            Directory recordings are written to when start_recording() is
            called without a file path. Created if it does not exist.
        record_keys : sequence of str, optional
            Keys of the frame dict to record. None (default) records all
            keys except exclude_keys.
        exclude_keys : sequence of str
            Keys never recorded. Default: 'frame_plot' (the normalised
            uint8 display image, which can be recomputed from the data).
        save_on_stop : bool
            If True (default) and a recording is still running when the
            thread is stopped, it is stopped and written to file.
        """
 
        super().__init__(name=name,
                         read_buffer=read_buffer,
                         ArrayType=ArrayType,
                         SensorType=SensorType,
                         **kwargs)
 
        self.record_dir = Path(record_dir)
        self.record_keys = None if record_keys is None else tuple(record_keys)
        self.exclude_keys = frozenset(exclude_keys)
        self.save_on_stop = save_on_stop
 
        # Recording state, shared between the display thread (_target) and
        # the thread calling start/stop_recording -> guarded by a lock
        self._rec_lock = threading.Lock()
        self._rec_active = False
        self._rec_frames: List[dict] = []
        self._rec_path: Optional[Path] = None
        self._rec_t_start: Optional[datetime] = None
 
        # Writer threads that have been started (one per recording)
        self._rec_writers: List[threading.Thread] = []
 
        # Window-title state, only touched by the display thread
        self._rec_title_shown = False
 
    # ── Commands ─────────────────────────────────────────────────────────
 
    @property
    def is_recording(self) -> bool:
        return self._rec_active
 
    @property
    def n_recorded(self) -> int:
        """Number of frames recorded so far in the current recording."""
        return len(self._rec_frames)
 
    def start_recording(self, file_path: Union[str, Path, None] = None) -> Path:
        """
        Start recording. Returns the path the recording will be written to.
 
        file_path : target file. Default:
                    <record_dir>/<thread name>_<YYYYmmdd_HHMMSS>.pkl
        """
        t_start = datetime.now()
 
        if file_path is None:
            stamp = t_start.strftime('%Y%m%d_%H%M%S')
            file_path = self.record_dir / f'{self.name}_{stamp}{_REC_SUFFIX}'
        file_path = Path(file_path)
 
        with self._rec_lock:
            if self._rec_active:
                raise RuntimeError(f"[{self.name}] Already recording to {self._rec_path}")
            self._rec_frames = []
            self._rec_path = file_path
            self._rec_t_start = t_start
            self._rec_active = True
 
        print(f"[{self.name}] Recording started -> {file_path}")
        return file_path
 
    def stop_recording(self) -> Optional[Path]:
        """
        Stop recording and write the recorded frames to file in a background
        thread. Returns immediately with the path the file is being written
        to, or None if no recording was running.
 
        Use wait_until_saved() to wait for the file to be complete.
        """
        t_stop = datetime.now()
 
        with self._rec_lock:
            if not self._rec_active:
                return None
            self._rec_active = False
            frames, self._rec_frames = self._rec_frames, []
            path = self._rec_path
            t_start = self._rec_t_start
 
        meta = {'format_version': _REC_FORMAT_VERSION,
                'source': self.name,
                'window_name': self.window_name,
                'tparray_npsize': getattr(self.tparray, '_npsize', None),
                't_start': t_start,
                't_stop': t_stop,
                'n_frames': len(frames)}
 
        print(f"[{self.name}] Recording stopped: {len(frames)} frames, writing {path} ...")
 
        # Non-daemon: the interpreter waits for the file to be complete
        # even if the main thread ends first
        writer = threading.Thread(target=self._write_recording,
                                  args=(path, meta, frames),
                                  name=f'{self.name}_writer',
                                  daemon=False)
        self._rec_writers.append(writer)
        writer.start()
        return path
 
    def wait_until_saved(self, timeout: Optional[float] = None) -> bool:
        """
        Wait until all recordings started so far are written. Returns True
        if all writers have finished.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        for writer in list(self._rec_writers):
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            writer.join(remaining)
        self._rec_writers = [w for w in self._rec_writers if w.is_alive()]
        return not self._rec_writers
 
    def stop(self):
        # Save a running recording when the thread is stopped
        if self.save_on_stop and self._rec_active:
            self.stop_recording()
        super().stop()
 
    # ── Display loop hook ────────────────────────────────────────────────
 
    def _target(self):
 
        # Get + preprocess the frame exactly like Imshow
        data = super()._target()
 
        # Keep it if recording. Only a shallow copy of the dict is stored:
        # the arrays are referenced, not copied (each frame from the
        # upstream thread comes with its own arrays).
        if self._rec_active and data.get('success') is True:
            entry = self._select_keys(data)
            with self._rec_lock:
                if self._rec_active:            # may have stopped meanwhile
                    self._rec_frames.append(entry)
 
        # Show the recording state in the window title. Goes through
        # self._gui like all window operations of Imshow: directly to cv2
        # (this thread owns the window) or to the ImshowDisplay thread.
        if self._rec_active != self._rec_title_shown:
            self._rec_title_shown = self._rec_active
            title = f'{self.window_name}  [REC]' if self._rec_active else self.window_name
            try:
                self._gui.setWindowTitle(self.window_name, title)
            except cv2.error:
                pass
 
        return data
 
    # ── Helpers ──────────────────────────────────────────────────────────
 
    def _select_keys(self, data: dict) -> dict:
        if self.record_keys is not None:
            return {k: data[k] for k in self.record_keys if k in data}
        return {k: v for k, v in data.items() if k not in self.exclude_keys}
 
    def _write_recording(self, path: Path, meta: dict, frames: List[dict]) -> None:
        """Runs in the writer thread."""
        part = path.with_name(path.name + _PART_SUFFIX)
        t0 = time.perf_counter()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(part, 'wb') as f:
                pkl.dump(meta, f, protocol=pkl.HIGHEST_PROTOCOL)
                for frame in frames:
                    pkl.dump(frame, f, protocol=pkl.HIGHEST_PROTOCOL)
                    time.sleep(0)         # yield the GIL to the display thread
            part.replace(path)            # complete -> final name
            print(f"[{self.name}] Saved {meta['n_frames']} frames to {path} "
                  f"({path.stat().st_size / 1e6:.1f} MB, "
                  f"{time.perf_counter() - t0:.1f} s)")
        except Exception as e:
            print(f"[{self.name}] Writing {path} failed: {e}")
            try:
                part.unlink()
            except OSError:
                pass
 
 
def load_recording(path: Union[str, Path]) -> Tuple[dict, List[dict]]:
    """
    Load a recording written by ImshowRecorder.
 
    Returns
    -------
    (meta, frames)
        meta   : dict with source, window_name, t_start, t_stop, n_frames, ...
        frames : list of the recorded frame dicts, in recording order
    """
    frames = []
    with open(path, 'rb') as f:
        meta = pkl.load(f)
        while True:
            try:
                frames.append(pkl.load(f))
            except EOFError:
                break
    return meta, frames


class DummyConsumer(RThread_R1):
    
    def __init__(self,
                 name:str,
                 read_buffer:Queue,
                 **kwargs):
        
        super().__init__(name=name,
                         read_buffer = read_buffer,
                         **kwargs)
    
    def _target(self):
        _ = self._get()
        print('Frame obtained!')
        return None

class DataRecord(RThread_R1):
    """
    Class for writing a stream of frames along with possible bounding
    boxes in a thread.
    """
    
    def __init__(self,
                 name:str,
                 read_buffer:Queue,
                 save_dir:Path,
                 **kwargs):
        
        
        # Check input arguments
        if not isinstance(save_dir,Path):
            raise TypeError(f'save_dir is type {type(save_dir)} instead of {Path}.')
              
        # Call __init__ of parent class
        super().__init__(name = name,
                         read_buffer = read_buffer,
                         **kwargs)
        
        # Assign user specified attributes
        self.save_dir = save_dir                                                # Directory to write results and recorded data to
        self.counter = 0                                                        # File counter, incremented after every write operation
        
        
        # Create a directory within save_dir to store recorded data
        self._init_rec_dir()
       
        
    def _init_rec_dir(self):
        '''
        Create a folder within save_dir, in which the recorded data is stored.
        The create directory is stored as class attribute rec_dir.
        Returns
        -------
        None.

        '''

        # rec_dir is named after the date and time of its creation
        formatted_datetime = datetime.now().strftime("%d_%m_%y_%H%M%S")
        self.rec_dir = self.save_dir / formatted_datetime
        
        # Create rec_dir and all its parents if necessary
        self.rec_dir.mkdir(parents=True,
                           exist_ok=True)
        
   
    def _target(self):
        """
        Gets result from upstream thread. Optionally converts the frame to a
        plotable format.

        Returns
        -------
        result : TYPE
            DESCRIPTION.

        """
        
        # Get result from upstream thread
        data = self._get()

        # Acquire timestamp from data, if existing
        if 't' in data.keys():
            timestamp = data['t']
        else:
            timestamp = datetime.now()
            data['t'] = timestamp
        
        # Make a filename
        filename = self.make_filename(self.counter,
                                      timestamp)
        
        # Pickle data in specified directory
        with open(self.rec_dir / filename, "wb") as f:
            pkl.dump(data, f)
            
        # Increment counter
        self.counter += 1
        
        return None

    def make_filename(self,
                      counter: int,
                      timestamp: datetime) -> str:
        
        if not isinstance(timestamp, datetime):
            raise TypeError('timestamp is type {type(timestamp)} instead of ' +\
                            'datetime.datetime.')
        if not isinstance(counter,int):
            raise TypeError('counter is type {type(counter)} instead of int.')
        
        # Convert datetime.datetime to string
        ts = timestamp.strftime("%Y%m%d_%H%M%S_%f")  # %f = microseconds
        
        return f"{counter:06d}_{ts}.pkl"
            
    # def run(self):
    #     """
    #     Function that it executed in the thread.

    #     Returns
    #     -------
    #     None.

    #     """
        
    #     # Open a cv namedWindow if specified
    #     if self.imshow == True:
    #         cv2.namedWindow(self.window_name,
    #                         cv2.WINDOW_NORMAL)
        
    #     # Check if thread has been stopped
    #     while self._exit == False:
            
    #         # Acquire the read condition
    #         with self.read_condition:

    #             # Wait until the upstream thread notifies this thread
    #             while self.read_buffer.empty():
    #                 self.read_condition.wait()    

    #             # Execute target function to get data from upstream thread
    #             data = self._target()
                               
    #             ############ Plotting part  ###################################
    #             if (data['success'] == True) and (self.imshow == True):
                    
    #                 # Get frame (processed)
    #                 frame = data['frame_plot']
                    
    #                 # Get bboxes if available
    #                 if 'bboxes' in data.keys():
                        
    #                     import pickle as pkl
                        
    #                     bboxes = data['bboxes']
                        
    #                     # Draw bounding boxes
    #                     for b in bboxes.index:
                            
    #                         box = bboxes.loc[[b]]
                            
    #                         x,y = int(box['xtl']),int(box['ytl']),
    #                         w = int(box['xbr'] - box['xtl'])
    #                         h = int(box['ybr'] - box['ytl'])
                            

    #                         frame = cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0) ,1)
                    
    #                 cv2.imshow(self.window_name,frame)
    #                 cv2.waitKey(1)
        
    #             ############ Recording part #######################################
    #             # Check if we're currently recording
    #             if not self.recording:
    
    #                 # Before recording, keep buffering the incoming data
    #                 self.pre_record_buffer.append(data)
    
    #                 # Check the start condition
    #                 if self._start_condition(data):
                        
    #                     # Once the starting condition is met, initialize
    #                     # a new folder and files in it to write to
    #                     self._initialize_recording_directory()
                        
    #                     # Write the pre-buffered data to the created files
    #                     self._write_data_to_files(self.pre_record_buffer)
                        
    #                     # Clear the pre-recorded buffer
    #                     self.pre_record_buffer.clear()  
                        
    #                     # Set recording flag
    #                     self.recording = True
                        
    #                     print("Recording started at frame " + str(data['image_id']) + \
    #                           ". Pre-recorded data included.")
    
                        
    #             else:
    
    #                 # During recording, write new data to file immediately
    #                 self._write_data_to_files([data])
                    
    #                 # self.recorded_data.append(data)
    
    #                 # Check if the stop condition is met
    #                 if self._stop_condition(data):
                        
    #                     # If so, unset recording flag
    #                     print("Recording stopped.")
    #                     self.recording = False
                        
    #                     # Inform the downstream thread that recorded data is available
    #                     # Acquire the write condition
    #                     with self.write_condition:
                            
    #                         # Write result to buffer
    #                         self.write_buffer.put({'rec_dir':self.rec_dir})
                            
    #                         # Notify the downstream thread, that item has been 
    #                         # placed in the write buffer
    #                         self.write_condition.notify()
                        
    #                     # Reset some class attributes
    #                     self.rec_dir = None
    #                     self.file_path = {}
                        
    #             # Notify the upstream thread, that the item has been retrieved
    #             # from the buffer and processed
    #             self.read_condition.notify()
        
    #     # If thread was stopped during recording, put the current data in the
    #     # write buffer and destroy the cv window if it exists
    #     if self._exit == True:
            
    #         if self.recording == True:
    #             self.write_buffer.put(self.recorded_data) 
    #             self.recorded_data = []
            
    #         if self.imshow == True:
    #             cv2.destroyWindow(self.window_name)
    

    
    # def _write_data_to_files(self,data:list):
    #     """
    #     Writes the values of specified keys (self.save_keys) to corresponding
    #     files. 

    #     Parameters
    #     ----------
    #     data : list or iterable
    #         A list containing dictionaries.

    #     Returns
    #     -------
    #     None.

    #     """
        
    #     # Loop through the iterable containing the data packages as dicts
    #     for data_dict in data:
            
    #         # Loop over keys to write to file
    #         for key in self.save_keys:
                
    #             # Check if key is in data dict
    #             if key in data_dict:
                    
    #                 # Check if corresponding file is empty
    #                 file_empty = os.path.getsize(self.file_path[key]) == 0
                            
    #                 # Parse values behind keys to a header and a numpy array
    #                 if key == 'bboxes':
                        
    #                     values = data_dict[key].values.flatten()
    #                     header = list(data_dict[key].columns)
                        
    #                 if key == 'frame':
                        
    #                     # Frame is in the format of a numpy array and needs to 
    #                     # be parsed to a pandas DataFrame
    #                     values = data_dict[key]
    #                     header = self.tparray.get_serial_data_order()
                    
    #                 # If file was empty, write a header first
    #                 if file_empty:
                        
    #                     with open(self.file_path[key],'w',newline='') as file:
    #                         writer = csv.writer(file)
    #                         writer.writerow(header)

                    
    #                 # In any case write values to file
    #                 if len(values)!=0:
    #                     with open(self.file_path[key],'a',newline='') as file:
    #                         writer = csv.writer(file)
    #                         writer.writerow(values)

class EMACounting_Thread(RThread_R1):
    """
    Thread that counts the number of confirmed tracks per frame, and returns
    an exponential moving average (EMA) of that number via
        
        s_t = a*x_t + (1-a) * s_t-1
        
        s_t:    Estimated average at time t
        a:      Filter coefficient
        x_t:    Detections at time t
        s_t-1:  Estimated average at time t-1
        
    Useful for counting the number of detected objects, e.g. persons, over time.
    """
    def __init__(self,
                 name : str,
                 read_buffer:Queue,
                 **kwargs):
        
        # self.T0 = T0                      # Desired time constant of EMA filter in seconds
        # self.a = None                     # Filter coefficient
        # self.filt_estimated  = False      # Boolean that indicates whether the filter coefficients have been estimated or not
        
        
        self.N_win = 10                # Number of initial frames from which dT is estimated, which is required to calculate the filter coefficient
        self.tau = 2
        
        self.Win = [np.nan]*self.N_win
        # self.k = 0                        # Counter that is incremented until N_window is reached
        # self.t0 = None                    # Time at receiving first detection result
        # self.dT_samples = []              # List of samples to estimate dT from
        
        # Call constructor of parent class
        super(EMACounting_Thread,self).__init__(name = name,
                                                read_buffer = read_buffer,
                                                **kwargs)
        
    @property
    def a(self):
        return self._a
    @a.setter
    def a(self, a:float):
        self._a = a
        
    @property
    def T0(self):
        return self._T0
    @T0.setter
    def T0(self, T0:int):
        self._T0 = T0

    @property
    def filt_estimated(self):
        return self._filt_estimated
    @filt_estimated.setter
    def filt_estimated(self, filt_estimated:bool):
        self._filt_estimated = filt_estimated
    
    @property
    def N_win(self):
        return self._N_win
    @N_win.setter
    def N_win(self, N_win:int):
        self._N_win = N_win
        
    @property
    def k(self):
        return self._k
    @k.setter
    def k(self, k:int):
        self._k = k
        
        
    def _target(self):
        
        
        # # ---------------------------------------------------------------------
        # # Estimate filter coefficients
        # # ---------------------------------------------------------------------
        
        # # At each call of this target function, sample the time since the last
        # # call to obtain an estimate for the sampling time interval dT
        # if not self.filt_estimated:
        #     if self.k == 0:
        #         self.t0 = time.time()
        #         self.k+=1
        #     elif self.k == 1:
        #         t_now = time.time()
        #         self.dT_samples.append(t_now-self.t0)
        #     elif self.k < self.N_init+1:
        #         t_now = time.time()
        #         self.dT_samples.append(t_now-self.dT_samples[-1])
        
        #     # If desired number of samples is acquired, estimate dT and a
        #     if len(self.dT_samples) == self.N_init:
        #         dT = np.mean(self.dT_samples)
        #         self.a = 1 - np.exp(-dT/self.T0)
                
        
        # ---------------------------------------------------------------------
        # Acquire data from read buffer
        # ---------------------------------------------------------------------
        result = self._get()

        # Try to obtain number of confirmed detections
        if result['success'] == True:
            if 'bboxes' in result.keys():
                bboxes = result['bboxes']
                
                n_conf = len(bboxes.loc[bboxes['confirmed']==True])
                
            else:
                n_conf = np.nan
                
        else:
            n_conf = np.nan
            
        # Add obtained confirmed detections to window
        self.Win.append(n_conf)
        self.Win = self.Win[1::]
        
        # Calculate the mean over the whole window
        n_avg = np.nanmean(np.array(self.Win))
        
        # Return the result as a dictionary Dictionary for storing results in
        result['n_avg'] = n_avg
        
        
        print(np.round(n_avg))

               
        return result
    
    

class FileWriter_Thread(RThread):
    """
    Thread for writing data into a file.
    """
    def __init__(self,
                 width:int,
                 height:int,
                 read_buffer:Queue,
                 read_condition:Condition,
                 **kwargs:dict):

        self.tparray = TPArray(width = width, height = height)                  # Array type

        self.save_dir = kwargs.pop('save_dir',Path.cwd())
        
        # Set time
        self.t0 = time.time()
        
        # Call parent class
        super().__init__(target = self._target_function,
                         read_buffer = read_buffer,
                         read_condition = read_condition,
                         **kwargs)
        
        # For debugging, write data to this attribute instead of to file
        self.debug_list = []
        
    def run(self):
        
        # Check if thread has been stopped
        while not self._exit.is_set():
            
            # Acquire the read condition
            with self.read_condition:
                
                # Wait until the upstream thread notifies this thread
                while self.read_buffer.empty():
                    self.read_condition.wait()   
                    
                
                # Get dictionary with data from upstream thread by calling target 
                # function
                data = self._target_function()

                
                # Data is a list of dictionaries, which need to be organized 
                # properly in order to be stored as a file
                organized_data = self._organize_data(data)
                
                try:
                    
                    # Create a folder with an expressive name
                    current_datetime = datetime.now()
                    formatted_datetime = current_datetime.strftime("%d_%m_%y_%H%M")
                    DevID = organized_data.pop('DevID')
                    folder = self.save_dir / (formatted_datetime + '_' + str(DevID))
                    
                    folder.mkdir(parents=True, exist_ok=True)
                    
                    # Save the remaining values in dict (DataFrames) to files
                    for key,df in organized_data.items():
                        file = folder / (key + '.df')
                        pkl.dump(df,open(file,'wb'))
                    
                    # self.read_buffer.task_done()
                    
                except self.read_buffer.empty:
                    continue
                
                # Notify the upstream thread, that the item has been retrieved
                # from the buffer
                self.read_condition.notify()
            
    def _organize_data(self,data:list):
        """
        organize dictionary such that content can be easily written to files

        Parameters
        ----------
        data : dict
            DESCRIPTION.

        Returns
        -------
        None.

        """
                
        # Initialize dictionaries/lists for storing re-organized data in
        frames = {}
        bboxes = []
        frames_proc = {}

        # Go through all dictionaries in the list and organize the data
        for frame_dict in data:

            # One row in TPArray's serial data order. The frame dict keeps
            # the pixels and the auxiliary data in separate keys (see
            # hspy.drivers._frame), so the row is assembled here rather
            # than sliced out of one flat vector -- which is what this used
            # to do, back when a client wrote the whole serial vector into
            # data['frame'].
            frames[frame_dict['image_id']] = self._serial_row(frame_dict)

            if 'bboxes' in frame_dict.keys():
                bboxes.append(frame_dict['bboxes'])


            if 'frame_proc' in frame_dict.keys():
                # Processed frame is a 2d array of pixel values
                temp = frame_dict['frame_proc'].reshape((-1,))

                # Append PTAT, elOff, etc, ...
                temp = np.hstack((temp,
                                  self._serial_aux(frame_dict) ))
                # frames_proc_temp
                frames_proc[frame_dict['image_id']] = temp

        # Make pandas DataFrames out of all of the dictionaries/lists
        df_frames = pd.DataFrame.from_dict(frames,
                                           orient='index',
                                           columns = self.tparray.get_serial_data_order())
    
        df_frames_proc = pd.DataFrame.from_dict(frames_proc,
                                                orient='index',
                                                columns = self.tparray.get_serial_data_order())
        
        df_bboxes = pd.concat(bboxes)
        
        
        # Put them in dictionary and return
        organized_data = {}
        
        organized_data['DevID'] = frame_dict['DevID']
        organized_data['frames'] = df_frames
        organized_data['frames_proc'] = df_frames_proc
        organized_data['bboxes'] = df_bboxes

        return organized_data

    def _serial_aux(self, frame_dict: dict) -> np.ndarray:
        """
        The non-pixel part of one frame, in TPArray's serial data order:
        e_off | vdd | T_amb | PTAT | ATC.

        A frame dict holds these as separate keys under the drivers' own
        names, and an array type that has no such values carries an empty
        array -- so this lines them up and pads whatever is short, rather
        than assuming every key has the length TPArray expects.
        """

        groups = self.tparray.DataCols.as_dict()
        parts = []

        for group, cols in groups.items():

            if group == 'pix' or not cols:
                continue

            values = np.asarray(frame_dict.get(AUX_FROM_DATACOLS[group],
                                               np.empty(0))).ravel()

            if values.size < len(cols):
                values = np.hstack([values,
                                    np.full(len(cols) - values.size, np.nan)])

            parts.append(values[:len(cols)])

        return np.hstack(parts) if parts else np.empty(0)

    def _serial_row(self, frame_dict: dict) -> np.ndarray:
        """One frame as a single row in TPArray's serial data order."""

        pixels = np.asarray(frame_dict['frame']).ravel()

        return np.hstack([pixels, self._serial_aux(frame_dict)])
    
    def _target_function(self):
        
        # print('Executed writer thread: ' + str(time.time()-self.t0) )
        
        # Get result from upstream thread
        upstream_dict = self.read_buffer.get()
    
        return upstream_dict