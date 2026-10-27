# -*- coding: utf-8 -*-
"""
Multiprocessing counterpart to threads.py / ProcessorThread.py.

Contains multiprocessing versions of:
  - USB_Client              (threads.py)          -> USB_ClientProcess
  - UDP_Client              (threads.py)          -> UDP_ClientProcess
  - I2C_Client              (threads.py)          -> I2C_ClientProcess
  - UDP_PickleServer        (threads.py)          -> UDP_PickleServerProcess
  - UDP_PickleClient        (threads.py)          -> UDP_PickleClientProcess
  - ProposalDetectorThread  (ProcessorThread.py)  -> ProposalDetectorProcess

DESIGN NOTE -- why these classes take *_cls / *_kwargs instead of a
ready-made instance
--------------------------------------------------------------------------
The threading versions (UDP_Client, I2C_Client) are constructed by passing
in an already-created, already-opened driver/reader object (an
HTPA_UDPReader with a bound socket, an I2C_HTPA32x32d with an open SMBus
handle). That works for threading because a thread shares the parent
process's memory -- the object, socket and all, is simply usable from the
new thread.

It does NOT work for multiprocessing: multiprocessing.Process pickles
whatever the child needs across the process boundary (or, on Linux with
the default 'fork' start method, silently duplicates open file
descriptors into the child -- which is arguably worse, since both parent
and child then think they own the same socket/bus handle). Either way,
handing a live socket or an open I2C bus handle to a child process is not
safe.

So every class below is constructed with a *class* + a *kwargs dict*
(everything needed to build the underlying driver/reader/detector) rather
than a live instance, and does the actual construction inside `_setup()`,
which -- per processes_base.py -- runs once, inside the child process,
right before the main loop starts. Only picklable configuration (classes,
dicts, strings, numbers) ever crosses the process boundary.

This applies to plain sockets too (UDP_PickleServerProcess /
UDP_PickleClientProcess below): even though a bare, not-yet-bound
socket.socket() often happens to survive being duplicated into a forked
child on Linux, that's start-method-dependent (it does NOT survive the
'spawn'/'forkserver' start methods, which are the default on Windows/macOS
and are frequently required e.g. together with CUDA). So sockets are also
created inside _setup(), never in __init__.
"""

import time
import socket
import struct
from datetime import datetime
import signal
import pickle as pkl
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union
import queue
from multiprocessing import Queue as MPQueue
from multiprocessing import Event as MPEvent
import threading

import cv2
import numpy as np

from hspy.LuT import LuT
from hspy.tparray import TPArray

from hspy.ipc.processes_base import WProcess_R1, RProcess_R1, RWProcess_R1
from hspy.ipc.plot_utils import TimeSeriesPlot


# Same wire-format constants as UDP_PickleServer / UDP_PickleClient in
# threads.py -- kept in sync manually since threads.py is a threading-only
# module and this one has no import dependency on it.
_PICKLE_HEADER_FMT = '!IHH'
_PICKLE_HEADER_SIZE = struct.calcsize(_PICKLE_HEADER_FMT)

# Stay safely below the 65507 byte payload limit of a UDP datagram
_PICKLE_MAX_CHUNK = 60000


class USB_ClientProcess(WProcess_R1):
    """
    Multiprocessing version of USB_Client. Runs a USB_HTPA driver in its own
    process, starts the device's frame stream and writes acquired frames
    into write_buffer.

    Losing the device (unplugged, cable fault, or no frame for lost_after_s
    seconds) is reported ONCE; the process then puts one final item
    {'success': False, 'device_lost': True, ...} into write_buffer, sets
    the device_lost event (readable from the parent) and ends. Other
    errors are printed once per distinct message, not on every attempt.

    Every item put into write_buffer carries 'success', 'image_id' and
    'DevID'; a successful one additionally carries 'frame' (the pixel
    values) plus everything USB_HTPA.acquire_frame() returned ('pix',
    'aux', 'ptat_avg', 't', 'serial_number', ...).
    """

    def __init__(self,
                 usb_driver_cls,
                 usb_driver_kwargs: dict,
                 write_buffer,
                 stream_cmd: bytes = b't',
                 name: str = 'usb_process',
                 lost_after_s: float = 5.0,
                 DevID: int = -1,
                 **kwargs):
        """
        Parameters
        ----------
        usb_driver_cls : type
            The USB_HTPA class (not an instance!) to construct inside the
            child process -- opening the USB device (libusb handle) in the
            parent and handing it to the child is not safe, so construction
            is deferred to _setup().
        usb_driver_kwargs : dict
            Keyword arguments to construct usb_driver_cls with, e.g.
            {'serial_number': '...'}.
        write_buffer : Queue / SharedMemoryQueue
            Buffer that acquired frames are put into.
        stream_cmd : bytes
            Command that starts the stream. b't' (default) streams pixel
            values the device has already converted; b'K' streams them in
            dK, which additionally end up in result['pix_dK'].
        name : str
            Process name. Give each process a distinct name when running
            several devices.
        lost_after_s : float
            If no frame could be read for this long (only errors), the
            device is considered lost even if the driver did not report a
            disconnect (e.g. some Windows USB errors after unplugging).
        DevID : int
            Optional device number, copied into every result as
            result['DevID'] so a consumer fed by several clients can tell
            them apart. -1 (default) means "not set"; USB devices are
            identified by result['serial_number'] anyway.

        Returns
        -------
        None.

        """

        self.usb_driver_cls = usb_driver_cls
        self.usb_driver_kwargs = usb_driver_kwargs
        self.stream_cmd = stream_cmd
        self.lost_after_s = lost_after_s
        self.DevID = DevID

        # Set by the child when the device is lost; check it in the parent
        # with usb_process.device_lost.is_set()
        self.device_lost = MPEvent()

        super().__init__(name=name,
                         write_buffer=write_buffer,
                         **kwargs)

    def _setup(self):

        # Ignore Ctrl+C in this process. On Windows Ctrl+C is delivered to
        # every process attached to the console, not only the parent. A
        # KeyboardInterrupt here would abort the loop in the middle of a USB
        # read, skip _teardown() and leave the device streaming. Instead the
        # parent catches Ctrl+C and calls stop(); the loop then exits after
        # the current frame and _teardown() stops the stream properly.
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        # Set image_id counter
        self.image_id = 0

        # Error bookkeeping: print each distinct error once
        self._last_error = None
        self._n_errors = 0

        # Set time
        self.t0 = time.time()
        self._t_last_frame = time.monotonic()

        # Opens the USB device here, inside the child process.
        # USB_HTPA.__init__ already calls open() + init() itself (init()
        # first stops a stream a previous run may have left running).
        self.usb_driver = None
        try:
            self.usb_driver = self.usb_driver_cls(**self.usb_driver_kwargs)

            # Start continuous frame stream
            resp = self.usb_driver.start_stream(self.stream_cmd)
            print(f"[{self.name}] Stream started: {resp!r}")
        except Exception as e:
            # Device gone / not usable: report once and end the process
            # (the loop in run() does not start because _exit is set).
            # _put() instead of write_buffer.put(): nothing may block here
            # either, or a pipeline whose consumer has not started yet
            # would hang in _setup() and never reach _teardown().
            self._device_lost(f"could not start the device: {e}")
            self._put(self._lost_item())
            return

    def _target(self):

        try:
            # Acquire one frame from the stream
            data = self.usb_driver.acquire_frame()

        except Exception as e:

            # Device gone: report once, hand a final 'device_lost' item to
            # the consumer and end the loop
            if not self.usb_driver.is_connected:
                self._device_lost(str(e))
                return self._lost_item()

            # Errors only, for too long: treat as lost as well
            if time.monotonic() - self._t_last_frame > self.lost_after_s:
                self._device_lost(f"no frame for {self.lost_after_s:g} s "
                                  f"(last error: {e})")
                return self._lost_item()

            # Other errors (timeout, sync error, ...): print each distinct
            # message once, and don't spin if the error comes back at once
            self._n_errors += 1
            message = str(e)
            if message != self._last_error:
                print(f"[{self.name}] {message}")
                self._last_error = message
            time.sleep(0.05)

            result = {'success': False}

        else:
            if self._n_errors:
                print(f"[{self.name}] Receiving frames again "
                      f"({self._n_errors} failed attempt(s) before).")
                self._n_errors = 0
                self._last_error = None
            self._t_last_frame = time.monotonic()

            # Set success flag and store frame data
            result = {'success': True}
            result.update(data)

            # Canonical key every client writes, whatever the transport is
            result['frame'] = data['pix']

        # Store image_id and device number
        result['image_id'] = self.image_id
        result['DevID'] = self.DevID

        # Increment image_id
        self.image_id = self.image_id + 1

        return result

    def _device_lost(self, reason: str):
        """Report the loss once, flag it for the parent, end the loop."""
        serial = self.usb_driver_kwargs.get('serial_number', '?')
        print(f"[{self.name}] Connection to device {serial} LOST: {reason}. "
              f"Continuing without it.")
        self.device_lost.set()
        self._exit.set()

    def _lost_item(self) -> dict:
        """Final item for the consumer: tells it that no more frames come."""
        return {'success': False,
                'device_lost': True,
                'serial_number': self.usb_driver_kwargs.get('serial_number'),
                'image_id': self.image_id,
                'DevID': self.DevID}

    def _teardown(self):
        # Runs inside the child process once the loop has exited: stop the
        # stream and release the device. Reaching this no longer depends on
        # the consumer still draining write_buffer -- WProcess_R1._put()
        # discards the oldest item instead of blocking -- so the shutdown
        # order of a pipeline is no longer load-bearing.
        self._release_device()

    def _release_device(self):
        usb_driver = getattr(self, 'usb_driver', None)
        if usb_driver is None:
            return
        self.usb_driver = None            # release only once
        if usb_driver.is_streaming and usb_driver.is_connected:
            try:
                resp, leftovers = usb_driver.stop_stream()
                print(f"[{self.name}] Stream stopped: {resp!r} "
                      f"({len(leftovers)} leftover frames discarded)")
            except Exception as e:
                print(f"[{self.name}] Stopping stream failed: {e}")
        try:
            usb_driver.close()
            print(f"[{self.name}] Device released.")
        except Exception as e:
            print(f"[{self.name}] Releasing device failed: {e}")

    def stop(self):
        # Only signals the child (this runs in the parent). The child's loop
        # exits after the current frame and _teardown() stops the stream.
        # Never terminate() this process, or the device keeps streaming.
        self._exit.set()


class UDP_PickleServerProcess(RProcess_R1):
    """
    Multiprocessing version of UDP_PickleServer. Reads arbitrary
    (picklable) data from an upstream buffer, in its own process, and
    sends it out via UDP to a single peer running a UDP_PickleClient /
    UDP_PickleClientProcess.

    Since a pickled payload can easily exceed the size of a single UDP
    datagram, it is split into numbered chunks that are reassembled by
    the receiver. As with any data sent over UDP, delivery is not
    guaranteed: a message of which one or more chunks are lost in transit
    is silently dropped by the receiver.
    """

    def __init__(self,
                 name: str,
                 read_buffer,
                 target_ip: str,
                 target_port: int,
                 **kwargs):
        """
        Parameters
        ----------
        read_buffer : Queue / SharedMemoryQueue
            Buffer that data to be transmitted is read from.
        target_ip, target_port : str, int
            Address of the peer running UDP_PickleClient(Process).

        Returns
        -------
        None.

        """

        self.target_ip = target_ip
        self.target_port = target_port

        super().__init__(name=name,
                         read_buffer=read_buffer,
                         **kwargs)

    def _setup(self):

        # The socket is created here, inside the child process -- not in
        # __init__ -- so that this works regardless of multiprocessing
        # start method ('fork', 'spawn' or 'forkserver').
        self.target_addr = (self.target_ip, self.target_port)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._msg_id = 0

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

    def _teardown(self):
        # Runs inside the child process once the loop has exited. Unlike
        # the threading version, stop() (which runs in the PARENT
        # process) cannot reach into this socket -- it only exists in the
        # child's memory -- so cleanup happens here instead.
        self._socket.close()

    def stop(self):
        self._exit.set()


class UDP_ClientProcess(WProcess_R1):
    """
    Multiprocessing version of UDP_Client. Runs an HTPA_UDPReader in its
    own process, binds one HTPA device, starts its continuous bytestream
    and writes assembled frames into write_buffer.

    Same shape as USB_ClientProcess: the reader is built from a class +
    kwargs inside the child process (see the design note at the top of
    this module), the device is bound and the stream started in _setup(),
    and both are undone in _teardown().

    Losing the device (unplugged, network down, or no complete frame for
    lost_after_s seconds) is reported ONCE; the process then puts one
    final item {'success': False, 'device_lost': True, ...} into
    write_buffer, sets the device_lost event (readable from the parent)
    and ends. Other errors are printed once per distinct message, not on
    every attempt.

    Every item put into write_buffer carries 'success', 'image_id' and
    'DevID'; a successful one additionally carries 'frame'.
    """

    def __init__(self,
                 udp_reader_cls,
                 udp_reader_kwargs: dict,
                 write_buffer,
                 IP: str = '',
                 DevID: int = -1,
                 Bcast_Addr: str = '',
                 stream_cmd: bytes = b'K',
                 name: str = 'udp_process',
                 lost_after_s: float = 5.0,
                 **kwargs):
        """
        Parameters
        ----------
        udp_reader_cls : type
            The HTPA_UDPReader class (not an instance!) to construct
            inside the child process.
        udp_reader_kwargs : dict
            Keyword arguments to construct udp_reader_cls with, e.g.
            {'ArrayType': 1}.
        write_buffer : Queue / SharedMemoryQueue
            Buffer that assembled frames are put into.
        IP, DevID, Bcast_Addr : see UDP_Client (threads.py) -- same
            binding semantics, just deferred to _setup().
        stream_cmd : bytes
            Command that starts the continuous bytestream. b'K' (default)
            streams pixel values in dK.
        name : str
            Process name. Give each process a distinct name when running
            several devices.
        lost_after_s : float
            If no complete frame could be assembled for this long (only
            errors / socket timeouts), the device is considered lost.

        Returns
        -------
        None.

        """

        # Checked here, in the parent, so a missing argument fails at
        # construction instead of inside the child a moment later
        if not ((DevID != -1 and len(Bcast_Addr) != 0) or len(IP) != 0):
            raise ValueError('Either IP or DevID and Bcast_Addr of the '
                             'device to be bound have to be specified!')

        self.udp_reader_cls = udp_reader_cls
        self.udp_reader_kwargs = udp_reader_kwargs

        self.IP = IP
        self.DevID = DevID
        self.Bcast_Addr = Bcast_Addr
        self.stream_cmd = stream_cmd
        self.lost_after_s = lost_after_s

        # Set by the child when the device is lost; check it in the parent
        # with udp_process.device_lost.is_set()
        self.device_lost = MPEvent()

        super().__init__(name=name,
                         write_buffer=write_buffer,
                         **kwargs)

    def _setup(self):

        # Ignore Ctrl+C in this process. On Windows Ctrl+C is delivered to
        # every process attached to the console, not only the parent. A
        # KeyboardInterrupt here would abort the loop in the middle of a
        # read, skip _teardown() and leave the device streaming into a
        # socket nobody reads any more. Instead the parent catches Ctrl+C
        # and calls stop(); the loop then exits after the current frame and
        # _teardown() stops the stream properly.
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        # Set image_id counter
        self.image_id = 0

        # Error bookkeeping: print each distinct error once
        self._last_error = None
        self._n_errors = 0

        self.t0 = time.time()
        self._t_last_frame = time.monotonic()

        # Set once the device is actually bound, so _teardown() only
        # releases what this process really acquired
        self._bound = False
        self.udp_reader = None

        try:
            # Construct the reader (and therefore open its socket) here,
            # inside the child process.
            self.udp_reader = self.udp_reader_cls(**self.udp_reader_kwargs)

            # Depending on whether the device's IP or the Device ID along
            # with the broadcast address are provided, the process of
            # finding and binding the corresponding htpa device differs
            if (self.DevID != -1) and (len(self.Bcast_Addr) != 0):
                self.udp_reader.broadcast(self.Bcast_Addr)
                self.udp_reader.bind_tparray(DevID=self.DevID)
            else:
                self.udp_reader.bind_tparray(IP=self.IP)

                # Save DevID of bound device to attribute
                devices = self.udp_reader.devices
                self.DevID = devices[devices['IP'] == self.IP].index.item()

            self._bound = True

            # Start continuous bytestream
            self.udp_reader.start_continuous_bytestream(self.DevID,
                                                        cmd=self.stream_cmd)
        except Exception as e:
            # Device gone / not reachable: report once and end the process
            # (the loop in run() does not start because _exit is set)
            self._device_lost(f"could not start the stream: {e}")
            self._put(self._lost_item())
            return

        print(f"[{self.name}] Stream started for device {self.DevID}.")

    def _target(self):

        try:
            # Try to assemble a frame from the udp packages
            frame = self.udp_reader.read_continuous_bytestream(self.DevID)

        except Exception as e:

            # Nothing but errors for too long: the device is gone (unplugged,
            # network down, powered off). Report once, hand a final
            # 'device_lost' item to the consumer, end the loop.
            if time.monotonic() - self._t_last_frame > self.lost_after_s:
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

            result = {'success': False}

        else:
            if self._n_errors:
                print(f"[{self.name}] Receiving frames again "
                      f"({self._n_errors} failed attempt(s) before).")
                self._n_errors = 0
                self._last_error = None
            self._t_last_frame = time.monotonic()

            # Set success flag and store frame
            result = {'success': True}
            result['frame'] = frame

        # Store image_id and device ID
        result['image_id'] = self.image_id
        result['DevID'] = self.DevID

        # Increment image_id
        self.image_id = self.image_id + 1

        return result

    def _device_lost(self, reason: str):
        """Report the loss once, flag it for the parent, end the loop."""
        print(f"[{self.name}] Connection to device {self.DevID} LOST: "
              f"{reason}. Continuing without it.")
        self.device_lost.set()
        self._exit.set()

    def _lost_item(self) -> dict:
        """Final item for the consumer: tells it that no more frames come."""
        return {'success': False,
                'device_lost': True,
                'image_id': self.image_id,
                'DevID': self.DevID}

    def _teardown(self):
        # Runs inside the child process once the loop has exited: stop the
        # stream and release the device, so it is not left streaming into a
        # socket nobody reads any more.
        self._release_device()

    def _release_device(self):
        udp_reader = getattr(self, 'udp_reader', None)
        if udp_reader is None:
            return
        self.udp_reader = None            # release only once

        if not getattr(self, '_bound', False):
            return
        self._bound = False

        try:
            udp_reader.stop_continuous_bytestream(DevID=self.DevID)
            print(f"[{self.name}] Stream stopped.")
        except Exception as e:
            print(f"[{self.name}] Stopping stream failed: {e}")

        try:
            # Sends the release command and closes the device's socket
            udp_reader.release_tparray(self.DevID)
        except Exception as e:
            print(f"[{self.name}] Releasing device failed: {e}")

    def stop(self):
        # Only signals the child (this runs in the parent). The child's loop
        # exits after the current frame and _teardown() stops the stream.
        # Never terminate() this process, or the device keeps streaming.
        self._exit.set()


class I2C_ClientProcess(WProcess_R1):
    """
    Multiprocessing version of I2C_Client. Runs an I2C_HTPA32x32d in its
    own process and writes acquired (+ converted / calibrated / LuT-mapped)
    frames into write_buffer.
    """

    def __init__(self,
                 i2c_driver_cls,
                 i2c_driver_kwargs: dict,
                 write_buffer,
                 lut_csv_path=None,
                 lut_offset: int = 0,
                 active_vdd: bool = True,
                 blind_vdd: bool = False,
                 applyCalib: bool = True,
                 calcdK: bool = True,
                 **kwargs):
        """
        Parameters
        ----------
        i2c_driver_cls : type
            The I2C_HTPA32x32d class (not an instance!) to construct
            inside the child process -- opening the SMBus handle in the
            parent and pickling/duplicating it into the child is not
            safe, so construction is deferred to _setup().
        i2c_driver_kwargs : dict
            Keyword arguments to construct i2c_driver_cls with, e.g.
            {'bus': 1}.
        write_buffer : Queue / SharedMemoryQueue
            Buffer that acquired frames are put into.
        lut_csv_path : str or Path, optional
            If given, a LuT is loaded from this CSV and assigned to the
            driver inside _setup() (mirrors setting i2c_driver.LuT in the
            threading version's calling script).
        lut_offset : int
            Passed through to LuT.from_csv().
        active_vdd, blind_vdd, applyCalib, calcdK :
            Forwarded to i2c_driver.acquire_frame() every iteration.

        Returns
        -------
        None.

        """

        self.i2c_driver_cls = i2c_driver_cls
        self.i2c_driver_kwargs = i2c_driver_kwargs

        self.lut_csv_path = lut_csv_path
        self.lut_offset = lut_offset

        self.active_vdd = active_vdd
        self.blind_vdd = blind_vdd
        self.applyCalib = applyCalib
        self.calcdK = calcdK

        super().__init__(name='i2c_process',
                         write_buffer=write_buffer,
                         **kwargs)

    def _setup(self):

        # Opens the I2C bus (SMBus handle) here, inside the child process.
        # I2C_HTPA32x32d.__init__ already calls open() + init() itself.
        self.i2c_driver = self.i2c_driver_cls(**self.i2c_driver_kwargs)

        if self.lut_csv_path is not None:
            lut = LuT()
            lut.from_csv(self.lut_csv_path, offset=self.lut_offset)
            self.i2c_driver.LuT = lut

        # Set image_id counter
        self.image_id = 0

        # Set time
        self.t0 = time.time()

    def _target(self):

        # Dictionary for storing results in
        result = {}

        try:
            # Acquire and process one frame from the sensor
            data = self.i2c_driver.acquire_frame(active_vdd=self.active_vdd,
                                                  blind_vdd=self.blind_vdd,
                                                  applyCalib=self.applyCalib,
                                                  calcdK=self.calcdK)

            # Set success flag and store frame data
            result['success'] = True
            result.update(data)
        except Exception as e:
            # Set success flag to False in case of failure
            result['success'] = False
            print(f"[{self.name}] {e}")

        # Store image_id
        result['image_id'] = self.image_id

        # Increment image_id
        self.image_id = self.image_id + 1

        return result

    def stop(self):
        self._exit.set()





class UDP_PickleServerProcess(RProcess_R1):
    """
    Multiprocessing version of UDP_PickleServer. Reads arbitrary
    (picklable) data from an upstream buffer, in its own process, and
    sends it out via UDP to a single peer running a UDP_PickleClient /
    UDP_PickleClientProcess.

    Since a pickled payload can easily exceed the size of a single UDP
    datagram, it is split into numbered chunks that are reassembled by
    the receiver. As with any data sent over UDP, delivery is not
    guaranteed: a message of which one or more chunks are lost in transit
    is silently dropped by the receiver.
    """

    def __init__(self,
                 name: str,
                 read_buffer,
                 target_ip: str,
                 target_port: int,
                 **kwargs):
        """
        Parameters
        ----------
        read_buffer : Queue / SharedMemoryQueue
            Buffer that data to be transmitted is read from.
        target_ip, target_port : str, int
            Address of the peer running UDP_PickleClient(Process).

        Returns
        -------
        None.

        """

        self.target_ip = target_ip
        self.target_port = target_port

        super().__init__(name=name,
                         read_buffer=read_buffer,
                         **kwargs)

    def _setup(self):

        # The socket is created here, inside the child process -- not in
        # __init__ -- so that this works regardless of multiprocessing
        # start method ('fork', 'spawn' or 'forkserver').
        self.target_addr = (self.target_ip, self.target_port)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._msg_id = 0

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

    def _teardown(self):
        # Runs inside the child process once the loop has exited. Unlike
        # the threading version, stop() (which runs in the PARENT
        # process) cannot reach into this socket -- it only exists in the
        # child's memory -- so cleanup happens here instead.
        self._socket.close()

    def stop(self):
        self._exit.set()


class UDP_PickleClientProcess(WProcess_R1):
    """
    Multiprocessing version of UDP_PickleClient. Runs in its own process,
    receives data sent by a UDP_PickleServer / UDP_PickleServerProcess,
    reassembles and unpickles it, and writes the resulting object into
    write_buffer for downstream processes to consume.
    """

    def __init__(self,
                 name: str,
                 write_buffer,
                 listen_ip: str = '0.0.0.0',
                 listen_port: int = 0,
                 timeout: float = 1.0,
                 **kwargs):
        """
        Parameters
        ----------
        write_buffer : Queue / SharedMemoryQueue
            Buffer that reassembled messages are put into.
        listen_ip, listen_port : str, int
            Local address to bind the receiving socket to.
        timeout : float
            recvfrom() timeout in seconds. Also bounds how quickly this
            process notices stop() was called (see note in _target()
            below) -- keep this reasonably small if fast shutdown matters.

        Returns
        -------
        None.

        """

        self.listen_ip = listen_ip
        self.listen_port = listen_port
        self.timeout = timeout

        # Small one-shot handshake queue: the child reports the address
        # it actually bound to, once, right after binding in _setup().
        # A plain multiprocessing.Queue (unlike a socket) is fine to
        # create here in the parent -- it's designed to cross the
        # process boundary safely.
        self._addr_queue = MPQueue(maxsize=1)
        self._resolved_addr = None

        super().__init__(name=name,
                         write_buffer=write_buffer,
                         **kwargs)

    def _setup(self):

        # Socket is created and bound here, inside the child process.
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind((self.listen_ip, self.listen_port))
        self._socket.settimeout(self.timeout)

        # Report the address actually bound to back to the parent (this
        # matters if listen_port=0 was passed and the OS assigned an
        # ephemeral port).
        self._addr_queue.put(self._socket.getsockname())

        # Chunks of messages that have not been fully received yet,
        # keyed by message id
        self._pending = {}

    def get_listen_addr(self, timeout: float = 5.0):
        """
        Blocks (from the PARENT process) until the child has bound its
        socket and reports the address back, or `timeout` seconds pass.

        Unlike UDP_PickleClient.listen_addr (a plain property that works
        immediately because the thread shares the parent's memory and the
        socket is created in __init__), the socket here only exists
        inside the child process and is only created once the process
        has actually started running (inside _setup()). So this must be
        called *after* .start(), and it can legitimately take a moment
        before the answer is available.

        Returns
        -------
        tuple (ip, port), or raises queue.Empty if the process hasn't
        bound within `timeout` seconds.
        """
        if self._resolved_addr is None:
            self._resolved_addr = self._addr_queue.get(timeout=timeout)
        return self._resolved_addr

    def _target(self):

        # Keep listening until either a full message has been
        # reassembled or the process is stopped.
        #
        # NOTE on shutdown: self._exit is a multiprocessing.Event, so it
        # IS visible here correctly even though stop() is called from the
        # parent process -- no special handling needed for that part.
        # What differs from the threading version is that stop() there
        # also closes the socket to unblock recvfrom() immediately; here,
        # stop() runs in the parent and has no way to reach into this
        # child's socket. Shutdown therefore relies entirely on the
        # recvfrom() timeout (set above from `timeout`) to periodically
        # re-check self._exit -- so worst-case shutdown latency for this
        # process is ~`timeout` seconds.
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

    def _teardown(self):
        self._socket.close()

    def stop(self):
        self._exit.set()


class ImshowProcess(RProcess_R1):
    """
    Multiprocessing version of Imshow. Reads frames (+ detections) from
    read_buffer and displays them in a cv2 window, running entirely in
    its own process.

    Note this is a case where multiprocessing is arguably a *better* fit
    than threading, not just a parallel option: GUI toolkits (including
    OpenCV's HighGUI backend) can be finicky about which thread they're
    driven from. Giving the display loop its own process sidesteps that
    class of issue entirely, at the cost of the usual IPC overhead.
    """

    def __init__(self,
                 name: str,
                 read_buffer,
                 ArrayType: int = None,
                 SensorType: int = None,
                 window_name: str = 'Sensor stream',
                 plot_ptat: bool = False,                                # ADDED
                 plot_window_s: float = None,                            # ADDED
                 **kwargs):
        """
        plot_ptat     : if True, a line plot of data['ptat_avg'] over time is
                        drawn below the image (in the same window).   # ADDED
        plot_window_s : show only the last plot_window_s seconds in that
                        plot; None (default) shows everything since the
                        start.                                        # ADDED
        """

        if ArrayType is None and SensorType is not None:
            self._tparray_kwargs = {'SensorType': SensorType}
        elif ArrayType is not None and SensorType is None:
            self._tparray_kwargs = {'ArrayType': ArrayType}
        else:
            raise Exception('Either ArrayType or SensorType must be provided.')

        self.window_name = window_name
        self.plot_ptat = plot_ptat                                       # ADDED
        self.plot_window_s = plot_window_s                               # ADDED

        super().__init__(name=name,
                         read_buffer=read_buffer,
                         **kwargs)

    def _setup(self):

        # TPArray construction and window creation both happen here,
        # inside the child process.
        self.tparray = TPArray(**self._tparray_kwargs)
        self.num_pix = len(self.tparray._pix)

        self.t0 = time.time()

        # PTAT history, kept in this process                             # ADDED
        self.ptat_plot = TimeSeriesPlot(title='PTAT avg', unit='digits', # ADDED
                                        window_s=self.plot_window_s)     # ADDED
        self._t_first = None                                             # ADDED

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def _target(self):

        # Get result from upstream process
        data = self._get()

        # Nothing to draw: a failed acquisition attempt, or the final
        # 'device_lost' item the client hands over before it ends. Keep
        # the window responsive instead of raising on a missing frame.
        if not data.get('success', True):
            if data.get('device_lost'):
                print(f"[{self.name}] Upstream reports the device as lost - "
                      f"no further frames.")
            cv2.waitKey(1)
            return

        # 'frame' is what every *_Client writes; 'pix_dK' / 'pix' are the
        # per-driver keys the I2C/SPI clients still use
        if 'frame' in data.keys():
            frame = data['frame']
        elif 'pix_dK' in data.keys():
            frame = data['pix_dK']
        else:
            frame = data['pix']

        # Reshape if not the proper size
        if frame.ndim == 1:
            frame = frame[0:self.num_pix]
            frame = frame.reshape(self.tparray._npsize)

        # Convert to opencv type
        frame = cv2.normalize(frame, frame, 0, 255, cv2.NORM_MINMAX)
        frame = frame.astype(np.uint8)

        # Scale it up by a factor of 10
        sf = 10
        frame = np.repeat(np.repeat(frame, sf, axis=0), sf, axis=1)

        # Convert frame to RGB to be able to plot colored boxes
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        # Draw bounding boxes / annotations, if present
        if 'annot_frame' in data.keys():

            annotations = data['annot_frame'].annotations
            annotations['confirmed'] = False
            annotations.loc[annotations['label'] == 1, 'confirmed'] = True
            bboxes = annotations

            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.4
            fontColor = (255, 255, 0)
            thickness = 1
            lineType = 2

            for b in bboxes.index:

                box = bboxes.loc[[b]]

                if box['confirmed'].item() != True:
                    continue

                x, y = int(box['xtl'].item()), int(box['ytl'].item())
                w = int(box['xbr'].item()) - int(box['xtl'].item())
                h = int(box['ybr'].item()) - int(box['ytl'].item())

                # Scale coordinates
                x = int(sf * x); y = int(sf * y); w = int(sf * w); h = int(sf * h)

                frame = cv2.rectangle(frame,
                                      (x, y),
                                      (x + w, y + h),
                                      (255, 255, 0), 1)

                # Write score if it exists
                if box['score'].item() != -99:
                    score = f"{box['score'].item():.2}"
                    cv2.putText(frame, score, (x + 1, y + 10), font,
                               fontScale, fontColor, thickness, lineType)

            # Write the number of persons in the upper left corner
            num_persons = sum(bboxes['confirmed'] == True)

            cv2.putText(frame,
                        f'Number of persons: {num_persons}',
                        (0, 20), font, fontScale, fontColor, thickness, lineType)

        # PTAT plot below the image                                      # ADDED
        if self.plot_ptat:                                               # ADDED
            frame = self._add_ptat_plot(frame, data)                     # ADDED

        cv2.imshow(self.window_name, frame)
        cv2.waitKey(1)

    def _add_ptat_plot(self, frame, data):                               # ADDED
        """Append data['ptat_avg'] to the history and stack the plot
        below the image. The plot is a third of the image's height."""
        if 'ptat_avg' in data:
            # Time axis: seconds since the first frame, from the frame's
            # own timestamp if it has one
            t = data['t'].timestamp() if hasattr(data.get('t'), 'timestamp') else time.time()
            if self._t_first is None:
                self._t_first = t
            self.ptat_plot.add(t - self._t_first, data['ptat_avg'])
        panel = self.ptat_plot.render(frame.shape[1], frame.shape[0] // 3)
        return np.vstack([frame, panel])

    def _on_idle(self):
        """Called by RProcess_R1.run() while no item arrives. Keeps the
        window responsive, so it still redraws and reacts to the window
        manager while the stream has not started yet or the device is
        gone."""
        try:
            cv2.waitKey(1)
        except cv2.error:
            pass

    def _teardown(self):
        # Window is destroyed here, inside the child process, once the
        # loop has exited -- this always runs (unlike the threading
        # version's `if self._exit == True:` check before destroying the
        # window, which relied on self._exit being reassigned to a plain
        # bool in stop() -- that reassignment breaks Event semantics and
        # isn't carried over here; self._exit is a proper
        # multiprocessing.Event throughout).
        cv2.destroyWindow(self.window_name)

    def stop(self):
        self._exit.set()

_REC_SUFFIX = '.pkl'
_PART_SUFFIX = '.part'
_REC_FORMAT_VERSION = 1
        
# class _RecordingTap:
#     """
#     Stands in for read_buffer inside the child process: forwards get() to
#     the real buffer and hands every item to a callback before returning it.
#     Lets ImshowRecorderProcess reuse ImshowProcess._target() unchanged.
#     All other attributes are forwarded to the real buffer.
#     """
 
#     def __init__(self, buffer, on_item):
#         self._buffer = buffer
#         self._on_item = on_item
 
#     def get(self, *args, **kwargs):
#         item = self._buffer.get(*args, **kwargs)
#         self._on_item(item)
#         return item
 
#     def __getattr__(self, attr):
#         return getattr(self._buffer, attr)
 
 
class _RecordingTap:
    """
    Stands in for read_buffer inside the child process: forwards get() to
    the real buffer and hands every item to a callback before returning it.
    Lets ImshowRecorderProcess reuse ImshowProcess._target() unchanged.
    All other attributes are forwarded to the real buffer.

    This used to poll the buffer in slices itself and call an on_idle
    callback in between, because the plain get() it wrapped would
    otherwise have waited for an item forever. Waiting is RProcess_R1's
    business now -- _get() reads with a timeout and run() calls
    _on_idle() whenever one of them comes up empty -- so there is one
    place left that decides how long a consumer waits, not two.
    """

    def __init__(self, buffer, on_item):
        self._buffer = buffer
        self._on_item = on_item

    def get(self, *args, **kwargs):
        item = self._buffer.get(*args, **kwargs)
        self._on_item(item)
        return item

    def __getattr__(self, attr):
        return getattr(self._buffer, attr)
 
 
class ImshowRecorderProcess(ImshowProcess):
    """
    ImshowProcess with recording. Displays every frame exactly like
    ImshowProcess; between start_recording() and stop_recording() every
    successfully received frame dict is also kept and, once recording
    stops, written to a pickle file in the background.
 
    start_recording() / stop_recording() / wait_until_saved() are called
    from the PARENT process (e.g. the main script reacting to user input).
    """
 
    def __init__(self,
                 name: str,
                 read_buffer,
                 ArrayType: int = None,
                 SensorType: int = None,
                 window_name: str = 'Sensor stream',
                 record_dir: Union[str, Path] = '.',
                 record_keys: Optional[Sequence[str]] = None,
                 exclude_keys: Sequence[str] = ('frame_plot',),
                 save_on_stop: bool = True,
                 window_geometry: Optional[Tuple[int, int, int, int]] = None,
                 **kwargs):
        """
        Parameters
        ----------
        name, read_buffer, ArrayType, SensorType, window_name, **kwargs :
            As for ImshowProcess (e.g. profile=True).
        window_geometry : (x, y, width, height), optional
            Position and size of the window on screen. Use different values
            per process so that several windows don't open on top of each
            other. None keeps OpenCV's default.
        record_dir : str or Path
            Directory recordings are written to when start_recording() is
            called without a file path. Created if it does not exist.
        record_keys : sequence of str, optional
            Keys of the frame dict to record. None (default) records all
            keys except exclude_keys.
        exclude_keys : sequence of str
            Keys never recorded.
        save_on_stop : bool
            If True (default) and a recording is still running when the
            process is stopped, it is written to file before the process
            exits.
        """
 
        super().__init__(name=name,
                         read_buffer=read_buffer,
                         ArrayType=ArrayType,
                         SensorType=SensorType,
                         window_name=window_name,
                         **kwargs)
 
        self.record_dir = Path(record_dir)
        self.record_keys = None if record_keys is None else tuple(record_keys)
        self.exclude_keys = tuple(exclude_keys)
        self.save_on_stop = save_on_stop
        self.window_geometry = window_geometry
 
        # Parent -> child: ('start', path, t_start, meta) / ('stop', t_stop, discard)
        #                  / ('clear_plot',)
        # Child -> parent: ('saved', path, n_frames) / ('failed', path, error)
        # Created here in the parent; multiprocessing queues are made to
        # cross the process boundary (unlike locks, sockets, devices).
        self._cmd_queue = MPQueue()
        self._status_queue = MPQueue()
 
        # Parent-side view of the recording state (only used in the parent)
        self._parent_recording = False
        self._parent_path: Optional[Path] = None
        self._parent_pending = 0          # saves requested but not yet confirmed
 
    # ── Commands (PARENT process) ────────────────────────────────────────
 
    @property
    def is_recording(self) -> bool:
        """True between start_recording() and stop_recording() (parent view)."""
        return self._parent_recording
 
    def start_recording(self,
                        file_path: Union[str, Path, None] = None,
                        meta: Optional[dict] = None) -> Path:
        """
        Start recording. Returns the path the recording will be written to.
 
        file_path : target file. Default:
                    <record_dir>/<process name>_<YYYYmmdd_HHMMSS>.pkl
        meta      : optional dict of extra information (e.g. measurement
                    conditions) stored in the file's metadata. Must be
                    picklable. The built-in keys (t_start, n_frames, ...)
                    take precedence over keys of the same name.
        """
        if self._parent_recording:
            raise RuntimeError(f"[{self.name}] Already recording to {self._parent_path}")
 
        t_start = datetime.now()
        if file_path is None:
            stamp = t_start.strftime('%Y%m%d_%H%M%S')
            file_path = self.record_dir / f'{self.name}_{stamp}{_REC_SUFFIX}'
        file_path = Path(file_path)
 
        self._cmd_queue.put(('start', str(file_path), t_start, dict(meta or {})))
        self._parent_recording = True
        self._parent_path = file_path
        return file_path
 
    def stop_recording(self, discard: bool = False) -> Optional[Path]:
        """
        Stop recording. The child writes the file in the background; this
        returns immediately with the file path, or None if no recording
        was running. Use wait_until_saved() to wait for the file.
 
        discard : if True, the recorded frames are thrown away and no file
                  is written (e.g. when a measurement was aborted). Returns
                  None in that case.
        """
        if not self._parent_recording:
            return None
        self._cmd_queue.put(('stop', datetime.now(), discard))
        self._parent_recording = False
        if discard:
            return None
        self._parent_pending += 1
        return self._parent_path
 
    def clear_plot(self) -> None:
        """
        Clear the PTAT plot below the image (plot_ptat=True); the time axis
        starts again at 0 with the next frame. Takes effect with the next
        frame the child receives.
        """
        self._cmd_queue.put(('clear_plot',))
 
    def wait_until_saved(self, timeout: Optional[float] = None) -> List[tuple]:
        """
        Wait until the child confirms every recording stopped so far.
 
        Returns the confirmations received, each ('saved', path, n_frames)
        or ('failed', path, error). Fewer entries than recordings means the
        timeout expired first.
        """
        results = []
        deadline = None if timeout is None else time.monotonic() + timeout
        while self._parent_pending > 0:
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                break
            try:
                results.append(self._status_queue.get(timeout=remaining))
            except queue.Empty:
                break
            self._parent_pending -= 1
        return results
 
    def stop(self):
        # A running recording is saved by the child on its way out
        if self._parent_recording and self.save_on_stop:
            self._parent_recording = False
            self._parent_pending += 1
        super().stop()
 
    # ── Child process ────────────────────────────────────────────────────
 
    def _setup(self):
 
        # Ignore Ctrl+C in this process. On Linux (and Windows) Ctrl+C is
        # delivered to every process attached to the terminal, which would
        # kill this process mid-recording. The parent catches Ctrl+C and
        # shuts down via stop(), which saves a running recording.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
 
        # Window, TPArray etc. exactly like ImshowProcess
        super()._setup()
 
        # Place the window
        if self.window_geometry is not None:
            x, y, w, h = self.window_geometry
            try:
                cv2.resizeWindow(self.window_name, w, h)
                cv2.moveWindow(self.window_name, x, y)
            except cv2.error:
                pass
 
        # Recording state -- lives only in the child, only touched by the
        # child's main thread (the display loop)
        self._rec_active = False
        self._rec_frames: List[dict] = []
        self._rec_path: Optional[Path] = None
        self._rec_t_start: Optional[datetime] = None
        self._rec_meta: dict = {}
        self._rec_writers: List[threading.Thread] = []
        self._rec_title_shown = None
        self._rec_device_lost = False      # device lost during the current recording
        self._device_lost = False          # upstream reported that its device is gone
 
        # Every item ImshowProcess._target() gets from the buffer passes
        # through _on_frame() first. While none arrive, RProcess_R1's loop
        # calls _on_idle() instead, once per self.get_timeout seconds.
        self.read_buffer = _RecordingTap(self.read_buffer, self._on_frame)
 
    def _on_frame(self, data):
        """Called by _RecordingTap for every item, before it is displayed."""
 
        self._poll_commands()
 
        if isinstance(data, dict) and data.get('device_lost'):
            if not self._device_lost:
                self._device_lost = True
                print(f"[{self.name}] Device lost - the window shows the last frame.")
            if self._rec_active:
                self._rec_device_lost = True
 
        if self._rec_active and isinstance(data, dict) and data.get('success') is True:
            # Only a shallow copy of the dict: arrays are referenced, not
            # copied (every item from the buffer is a fresh object)
            self._rec_frames.append(self._select_keys(data))
 
        self._update_title()
 
    def _on_idle(self):
        """Called by RProcess_R1.run() while no item arrives."""
        self._poll_commands()
        self._update_title()
        try:
            cv2.waitKey(1)                 # keep the window responsive
        except cv2.error:
            pass
 
    def _poll_commands(self):
        while True:
            try:
                cmd = self._cmd_queue.get_nowait()
            except queue.Empty:
                return
 
            if cmd[0] == 'start':
                if self._rec_active:                 # should not happen
                    self._save(datetime.now())
                _, path, t_start, meta = cmd
                self._rec_frames = []
                self._rec_path = Path(path)
                self._rec_t_start = t_start
                self._rec_meta = meta
                self._rec_device_lost = self._device_lost
                self._rec_active = True
                if self._device_lost:
                    print(f"[{self.name}] Recording started, but the device is lost: "
                          f"the file will contain no frames -> {path}")
                else:
                    print(f"[{self.name}] Recording started -> {path}")
 
            elif cmd[0] == 'stop':
                _, t_stop, discard = cmd
                if discard:
                    self._discard()
                else:
                    self._save(t_stop)
 
            elif cmd[0] == 'clear_plot':
                # Plot and time origin belong to ImshowProcess (plot_ptat)
                if hasattr(self, 'ptat_plot'):
                    self.ptat_plot.clear()
                    self._t_first = None
 
    def _discard(self):
        """Stop recording and throw the frames away."""
        if not self._rec_active:
            return
        print(f"[{self.name}] Recording discarded ({len(self._rec_frames)} frames): "
              f"{self._rec_path}")
        self._rec_active = False
        self._rec_frames = []
 
    def _save(self, t_stop: datetime):
        """Stop recording and write the frames in a writer thread."""
        if not self._rec_active:
            return
        self._rec_active = False
        frames, self._rec_frames = self._rec_frames, []
 
        meta = {**self._rec_meta,                         # user meta first ...
                'format_version': _REC_FORMAT_VERSION,    # ... built-in keys win
                'source': self.name,
                'window_name': self.window_name,
                'tparray_npsize': getattr(self.tparray, '_npsize', None),
                't_start': self._rec_t_start,
                't_stop': t_stop,
                'n_frames': len(frames),
                'device_lost': self._rec_device_lost}   # True: frames stop early
 
        print(f"[{self.name}] Recording stopped: {len(frames)} frames, "
              f"writing {self._rec_path} ...")
 
        writer = threading.Thread(target=self._write_recording,
                                  args=(self._rec_path, meta, frames),
                                  name=f'{self.name}_writer',
                                  daemon=False)
        self._rec_writers.append(writer)
        writer.start()
 
    def _write_recording(self, path: Path, meta: dict, frames: List[dict]) -> None:
        """Runs in a writer thread inside the child process."""
        part = path.with_name(path.name + _PART_SUFFIX)
        t0 = time.perf_counter()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(part, 'wb') as f:
                pkl.dump(meta, f, protocol=pkl.HIGHEST_PROTOCOL)
                for frame in frames:
                    pkl.dump(frame, f, protocol=pkl.HIGHEST_PROTOCOL)
                    time.sleep(0)         # yield the GIL to the display loop
            part.replace(path)            # complete -> final name
            print(f"[{self.name}] Saved {meta['n_frames']} frames to {path} "
                  f"({path.stat().st_size / 1e6:.1f} MB, "
                  f"{time.perf_counter() - t0:.1f} s)")
            self._status_queue.put(('saved', str(path), meta['n_frames']))
        except Exception as e:
            print(f"[{self.name}] Writing {path} failed: {e}")
            try:
                part.unlink()
            except OSError:
                pass
            self._status_queue.put(('failed', str(path), str(e)))
 
    def _update_title(self):
        # Show the recording / connection state in the window title
        state = (self._rec_active, self._device_lost)
        if state != self._rec_title_shown:
            self._rec_title_shown = state
            title = self.window_name
            if self._device_lost:
                title += '  [DISCONNECTED]'
            if self._rec_active:
                title += '  [REC]'
            try:
                cv2.setWindowTitle(self.window_name, title)
            except cv2.error:
                pass
 
    def _select_keys(self, data: dict) -> dict:
        if self.record_keys is not None:
            return {k: data[k] for k in self.record_keys if k in data}
        return {k: v for k, v in data.items() if k not in self.exclude_keys}
 
    def _teardown(self):
 
        # Commands that arrived after the last frame
        self._poll_commands()
 
        # Recording still running when the process was stopped
        if self._rec_active:
            if self.save_on_stop:
                self._save(datetime.now())
            else:
                self._discard()
 
        # Do not exit before every file is completely written
        for writer in self._rec_writers:
            writer.join()
 
        # Destroy the window (ImshowProcess)
        super()._teardown()
 
 
def load_recording(path: Union[str, Path]) -> Tuple[dict, List[dict]]:
    """
    Load a recording written by ImshowRecorderProcess (or ImshowRecorder --
    same format).
 
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


class PlotTimeSeries_Process(RProcess_R1):
    """
    Reads data from read_buffer (e.g. from Optris_ClientProcess) and shows
    each value as a time series in a cv2 window, running entirely in its own
    process. Every value gets its own panel; the panels are stacked
    vertically in one window.
 
    Expected items in read_buffer:
        {'success': True, 'timestamp': datetime, '<key>': value, ...}
    Items with success=False are skipped. A final 'device_lost' item (as
    sent by Optris_ClientProcess) keeps the plot open and marks it with
    'DEVICE LOST'.
    """
 
    # Panel titles and units for the Optris data keys. Hershey fonts used
    # by cv2.putText cannot draw the degree sign, hence 'degC'.
    TITLES = {'temp_process': 'Process temperature',
              'temp_head': 'Head temperature',
              'temp_box': 'Box temperature',
              'temp_act': 'Temperature (act.)',
              'eps': 'Emissivity',
              'trans': 'Transmission'}
 
    UNITS = {'temp_process': 'degC',
             'temp_head': 'degC',
             'temp_box': 'degC',
             'temp_act': 'degC',
             'eps': '-',
             'trans': '-'}
 
    # Keys in the incoming dicts that are bookkeeping, not data to plot
    _META_KEYS = {'success', 'timestamp', 'sample_id', 'image_id',
                  'device_lost', 'port', 'serial_number'}
 
    def __init__(self,
                 name: str,
                 read_buffer,
                 data_keys: list = None,
                 window_name: str = 'Optris time series',
                 plot_window_s: float = None,
                 width: int = 800,
                 panel_height: int = 200,
                 **kwargs):
        """
        Parameters
        ----------
        name : str
            Process name.
        read_buffer : Queue / SharedMemoryQueue
            Buffer to read the data from, i.e. the write_buffer of the
            Optris_ClientProcess.
        data_keys : list of str, optional
            Values to plot, e.g. ['temp_process', 'temp_head']. None
            (default) plots every numeric value found in the first valid
            item, i.e. whatever the producer was told to read.
        window_name : str
            Name of the cv2 window. Use distinct names when running
            several plot processes.
        plot_window_s : float, optional
            Show only the last plot_window_s seconds; None (default) shows
            everything since the start.
        width : int
            Window width in pixels.
        panel_height : int
            Height of each panel in pixels.
 
        Returns
        -------
        None.
 
        """
 
        if isinstance(data_keys, str):
            data_keys = [data_keys]
        self.data_keys = list(data_keys) if data_keys is not None else None
 
        self.window_name = window_name
        self.plot_window_s = plot_window_s
        self.width = width
        self.panel_height = panel_height
 
        super().__init__(name=name,
                         read_buffer=read_buffer,
                         **kwargs)
 
    def _setup(self):
 
        # Plot construction and window creation both happen here, inside
        # the child process.
        self.t0 = time.time()
 
        # One TimeSeriesPlot per key, created on the first valid item
        # (when data_keys=None, the keys are only known then)
        self.plots = None
        self._t_first = None
        self._device_lost = False
 
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        self._show(self._message_image('Waiting for data ...'))
 
    def _target(self):
 
        # Get result from upstream process. _get() waits with a timeout,
        # so stop() ends the loop and the window stays responsive while no
        # data arrives.
        try:
            data = self._get()
        except queue.Empty:
            self._on_idle()
            cv2.waitKey(1)
            return
 
        # Hook for subclasses (e.g. recording), sees every item first
        self._on_item(data)
 
        # Producer lost its device: no more data will come
        if data.get('device_lost'):
            self._device_lost = True
            self._show(self._render())
            return
 
        # Failed poll: nothing to plot
        if not data.get('success', True):
            cv2.waitKey(1)
            return
 
        # Create the plots on the first valid item
        if self.plots is None:
            self._create_plots(data)
 
        # Time axis: seconds since the first item, from the item's own
        # timestamp if it has one
        t = data.get('timestamp')
        t = t.timestamp() if hasattr(t, 'timestamp') else time.time()
        if self._t_first is None:
            self._t_first = t
 
        # Append the new values to the history
        for key, plot in self.plots.items():
            if key in data:
                plot.add(t - self._t_first, data[key])
 
        self._show(self._render())
 
    def _on_item(self, data):
        """Called for every item from read_buffer before it is plotted.
        Override in subclasses, e.g. PlotTimeSeries_ProcessRecord."""
        pass
 
    def _on_idle(self):
        """Called once per self.get_timeout seconds while no item
        arrives. Override in subclasses."""
        pass
 
    def _create_plots(self, data):
        """Create one TimeSeriesPlot per key to plot."""
        keys = self.data_keys
        if keys is None:
            keys = [k for k, v in data.items()
                    if k not in self._META_KEYS
                    and isinstance(v, (int, float, np.number))
                    and not isinstance(v, bool)]
        else:
            missing = [k for k in keys if k not in data]
            if missing:
                print(f"[{self.name}] Keys {missing} are not in the data "
                      f"(available: {list(data)}); their panels stay empty.")
 
        self.plots = {key: TimeSeriesPlot(title=self.TITLES.get(key, key),
                                          unit=self.UNITS.get(key, ''),
                                          window_s=self.plot_window_s)
                      for key in keys}
 
    def _render(self):
        """Stack the panels of all plots into one image."""
        if not self.plots:
            image = self._message_image('No data')
        else:
            panels = [plot.render(self.width, self.panel_height)
                      for plot in self.plots.values()]
            image = np.vstack(panels)
 
        if self._device_lost:
            # Top right corner, clear of the panel title
            text, font, scale, thick = 'DEVICE LOST', cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            (w, h), _ = cv2.getTextSize(text, font, scale, thick)
            cv2.putText(image, text, (image.shape[1] - w - 10, h + 10),
                        font, scale, (0, 0, 255), thick, cv2.LINE_AA)
        return image
 
    def _message_image(self, text):
        """Blank panel with a short message, shown before data arrives."""
        image = np.zeros((self.panel_height, self.width, 3), dtype=np.uint8)
        cv2.putText(image, text, (10, self.panel_height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1,
                    cv2.LINE_AA)
        return image
 
    def _show(self, image):
        cv2.imshow(self.window_name, image)
        cv2.waitKey(1)
 
    def _teardown(self):
        # Window is destroyed here, inside the child process, once the
        # loop has exited.
        cv2.destroyWindow(self.window_name)
 
    def stop(self):
        self._exit.set()
        

class PlotTimeSeries_ProcessRecord(PlotTimeSeries_Process):
    """
    PlotTimeSeries_Process with recording. Plots every item exactly like
    PlotTimeSeries_Process; between start_recording() and stop_recording()
    every successfully received item dict (timestamp + values) is also kept
    and, once recording stops, written to a pickle file in the background.
 
    start_recording() / stop_recording() / wait_until_saved() are called
    from the PARENT process (e.g. the main script reacting to user input).
    """
 
    def __init__(self,
                 name: str,
                 read_buffer,
                 data_keys: Optional[Sequence[str]] = None,
                 window_name: str = 'Optris time series',
                 record_dir: Union[str, Path] = '.',
                 record_keys: Optional[Sequence[str]] = None,
                 exclude_keys: Sequence[str] = ('success',),
                 save_on_stop: bool = True,
                 window_geometry: Optional[Tuple[int, int, int, int]] = None,
                 **kwargs):
        """
        Parameters
        ----------
        name, read_buffer, data_keys, window_name, **kwargs :
            As for PlotTimeSeries_Process (e.g. plot_window_s, width,
            panel_height, profile=True).
        window_geometry : (x, y, width, height), optional
            Position and size of the window on screen. Use different values
            per process so that several windows don't open on top of each
            other. None keeps OpenCV's default.
        record_dir : str or Path
            Directory recordings are written to when start_recording() is
            called without a file path. Created if it does not exist.
        record_keys : sequence of str, optional
            Keys of the item dict to record. None (default) records all
            keys except exclude_keys, i.e. 'timestamp', 'sample_id' and
            every value the producer reads -- also values that are not
            plotted.
        exclude_keys : sequence of str
            Keys never recorded. Default: ('success',), since only
            successful items are recorded anyway.
        save_on_stop : bool
            If True (default) and a recording is still running when the
            process is stopped, it is written to file before the process
            exits.
        """
 
        super().__init__(name=name,
                         read_buffer=read_buffer,
                         data_keys=data_keys,
                         window_name=window_name,
                         **kwargs)
 
        self.record_dir = Path(record_dir)
        self.record_keys = None if record_keys is None else tuple(record_keys)
        self.exclude_keys = tuple(exclude_keys)
        self.save_on_stop = save_on_stop
        self.window_geometry = window_geometry
 
        # Parent -> child: ('start', path, t_start, meta) / ('stop', t_stop, discard)
        #                  / ('clear_plot',)
        # Child -> parent: ('saved', path, n_items) / ('failed', path, error)
        # Created here in the parent; multiprocessing queues are made to
        # cross the process boundary (unlike locks, sockets, devices).
        self._cmd_queue = MPQueue()
        self._status_queue = MPQueue()
 
        # Parent-side view of the recording state (only used in the parent)
        self._parent_recording = False
        self._parent_path: Optional[Path] = None
        self._parent_pending = 0          # saves requested but not yet confirmed
 
    # ── Commands (PARENT process) ────────────────────────────────────────
 
    @property
    def is_recording(self) -> bool:
        """True between start_recording() and stop_recording() (parent view)."""
        return self._parent_recording
 
    def start_recording(self,
                        file_path: Union[str, Path, None] = None,
                        meta: Optional[dict] = None) -> Path:
        """
        Start recording. Returns the path the recording will be written to.
 
        file_path : target file. Default:
                    <record_dir>/<process name>_<YYYYmmdd_HHMMSS>.pkl
        meta      : optional dict of extra information (e.g. measurement
                    conditions) stored in the file's metadata. Must be
                    picklable. The built-in keys (t_start, n_items, ...)
                    take precedence over keys of the same name.
        """
        if self._parent_recording:
            raise RuntimeError(f"[{self.name}] Already recording to {self._parent_path}")
 
        t_start = datetime.now()
        if file_path is None:
            stamp = t_start.strftime('%Y%m%d_%H%M%S')
            file_path = self.record_dir / f'{self.name}_{stamp}{_REC_SUFFIX}'
        file_path = Path(file_path)
 
        self._cmd_queue.put(('start', str(file_path), t_start, dict(meta or {})))
        self._parent_recording = True
        self._parent_path = file_path
        return file_path
 
    def stop_recording(self, discard: bool = False) -> Optional[Path]:
        """
        Stop recording. The child writes the file in the background; this
        returns immediately with the file path, or None if no recording
        was running. Use wait_until_saved() to wait for the file.
 
        discard : if True, the recorded items are thrown away and no file
                  is written (e.g. when a measurement was aborted). Returns
                  None in that case.
        """
        if not self._parent_recording:
            return None
        self._cmd_queue.put(('stop', datetime.now(), discard))
        self._parent_recording = False
        if discard:
            return None
        self._parent_pending += 1
        return self._parent_path
 
    def clear_plot(self) -> None:
        """
        Clear all time series plots; the time axis starts again at 0 with
        the next item. Does not affect a running recording.
        """
        self._cmd_queue.put(('clear_plot',))
 
    def wait_until_saved(self, timeout: Optional[float] = None) -> List[tuple]:
        """
        Wait until the child confirms every recording stopped so far.
 
        Returns the confirmations received, each ('saved', path, n_items)
        or ('failed', path, error). Fewer entries than recordings means the
        timeout expired first.
        """
        results = []
        deadline = None if timeout is None else time.monotonic() + timeout
        while self._parent_pending > 0:
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                break
            try:
                results.append(self._status_queue.get(timeout=remaining))
            except queue.Empty:
                break
            self._parent_pending -= 1
        return results
 
    def stop(self):
        # A running recording is saved by the child on its way out
        if self._parent_recording and self.save_on_stop:
            self._parent_recording = False
            self._parent_pending += 1
        super().stop()
 
    # ── Reading a recording (any process) ────────────────────────────────
 
    @staticmethod
    def load_recording(path: Union[str, Path]) -> Tuple[dict, List[dict]]:
        """
        Read a recording written by this class. Returns (meta, items),
        where items is the list of recorded dicts, e.g.
        {'timestamp': datetime, 'temp_process': 36.6, ..., 'sample_id': 0}.
 
        To get a pandas DataFrame:
            meta, items = PlotTimeSeries_ProcessRecord.load_recording(path)
            df = pd.DataFrame(items).set_index('timestamp')
        """
        items = []
        with open(path, 'rb') as f:
            meta = pkl.load(f)
            while True:
                try:
                    items.append(pkl.load(f))
                except EOFError:
                    break
        return meta, items
 
    # ── Child process ────────────────────────────────────────────────────
 
    def _setup(self):
 
        # Ignore Ctrl+C in this process. On Linux (and Windows) Ctrl+C is
        # delivered to every process attached to the terminal, which would
        # kill this process mid-recording. The parent catches Ctrl+C and
        # shuts down via stop(), which saves a running recording.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
 
        # Window and plot state exactly like PlotTimeSeries_Process
        super()._setup()
 
        # Place the window
        if self.window_geometry is not None:
            x, y, w, h = self.window_geometry
            try:
                cv2.resizeWindow(self.window_name, w, h)
                cv2.moveWindow(self.window_name, x, y)
            except cv2.error:
                pass
 
        # Recording state -- lives only in the child, only touched by the
        # child's main thread (the plot loop)
        self._rec_active = False
        self._rec_items: List[dict] = []
        self._rec_path: Optional[Path] = None
        self._rec_t_start: Optional[datetime] = None
        self._rec_meta: dict = {}
        self._rec_writers: List[threading.Thread] = []
        self._rec_title_shown = None
        self._rec_device_lost = False      # device lost during the current recording
        self._rec_keys_seen: List[str] = []   # value keys of the recorded items
 
        self._update_title()
 
    def _on_item(self, data):
        """Called by PlotTimeSeries_Process._target() for every item,
        before it is plotted."""
 
        self._poll_commands()
 
        if isinstance(data, dict) and data.get('device_lost'):
            if not self._device_lost:
                self._device_lost = True
                print(f"[{self.name}] Device lost - the plot shows the data "
                      f"received so far.")
            if self._rec_active:
                self._rec_device_lost = True
 
        if self._rec_active and isinstance(data, dict) and data.get('success') is True:
            item = self._select_keys(data)
            if not self._rec_items:
                self._rec_keys_seen = list(item)
            self._rec_items.append(item)
 
        self._update_title()
 
    def _on_idle(self):
        """Called by PlotTimeSeries_Process._target() while no item arrives."""
        self._poll_commands()
        self._update_title()
 
    def _poll_commands(self):
        while True:
            try:
                cmd = self._cmd_queue.get_nowait()
            except queue.Empty:
                return
 
            if cmd[0] == 'start':
                if self._rec_active:                 # should not happen
                    self._save(datetime.now())
                _, path, t_start, meta = cmd
                self._rec_items = []
                self._rec_keys_seen = []
                self._rec_path = Path(path)
                self._rec_t_start = t_start
                self._rec_meta = meta
                self._rec_device_lost = self._device_lost
                self._rec_active = True
                if self._device_lost:
                    print(f"[{self.name}] Recording started, but the device is lost: "
                          f"the file will contain no data -> {path}")
                else:
                    print(f"[{self.name}] Recording started -> {path}")
 
            elif cmd[0] == 'stop':
                _, t_stop, discard = cmd
                if discard:
                    self._discard()
                else:
                    self._save(t_stop)
 
            elif cmd[0] == 'clear_plot':
                # Plots are created again from the next item, and the time
                # axis restarts at 0 (see PlotTimeSeries_Process._target)
                self.plots = None
                self._t_first = None
                self._show(self._message_image('Plot cleared - waiting for data ...'))
 
    def _discard(self):
        """Stop recording and throw the items away."""
        if not self._rec_active:
            return
        print(f"[{self.name}] Recording discarded ({len(self._rec_items)} items): "
              f"{self._rec_path}")
        self._rec_active = False
        self._rec_items = []
 
    def _save(self, t_stop: datetime):
        """Stop recording and write the items in a writer thread."""
        if not self._rec_active:
            return
        self._rec_active = False
        items, self._rec_items = self._rec_items, []
 
        meta = {**self._rec_meta,                         # user meta first ...
                'format_version': _REC_FORMAT_VERSION,    # ... built-in keys win
                'source': self.name,
                'window_name': self.window_name,
                'keys': self._rec_keys_seen,              # keys of each item
                'plotted_keys': list(self.plots) if self.plots else None,
                't_start': self._rec_t_start,
                't_stop': t_stop,
                'n_items': len(items),
                'device_lost': self._rec_device_lost}     # True: data stops early
 
        print(f"[{self.name}] Recording stopped: {len(items)} items, "
              f"writing {self._rec_path} ...")
 
        writer = threading.Thread(target=self._write_recording,
                                  args=(self._rec_path, meta, items),
                                  name=f'{self.name}_writer',
                                  daemon=False)
        self._rec_writers.append(writer)
        writer.start()
 
    def _write_recording(self, path: Path, meta: dict, items: List[dict]) -> None:
        """Runs in a writer thread inside the child process."""
        part = path.with_name(path.name + _PART_SUFFIX)
        t0 = time.perf_counter()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(part, 'wb') as f:
                pkl.dump(meta, f, protocol=pkl.HIGHEST_PROTOCOL)
                for item in items:
                    pkl.dump(item, f, protocol=pkl.HIGHEST_PROTOCOL)
            part.replace(path)            # complete -> final name
            print(f"[{self.name}] Saved {meta['n_items']} items to {path} "
                  f"({path.stat().st_size / 1e3:.1f} kB, "
                  f"{time.perf_counter() - t0:.2f} s)")
            self._status_queue.put(('saved', str(path), meta['n_items']))
        except Exception as e:
            print(f"[{self.name}] Writing {path} failed: {e}")
            try:
                part.unlink()
            except OSError:
                pass
            self._status_queue.put(('failed', str(path), str(e)))
 
    def _update_title(self):
        # Show the recording / connection state in the window title
        state = (self._rec_active, self._device_lost)
        if state != self._rec_title_shown:
            self._rec_title_shown = state
            title = self.window_name
            if self._device_lost:
                title += '  [DISCONNECTED]'
            if self._rec_active:
                title += '  [REC]'
            try:
                cv2.setWindowTitle(self.window_name, title)
            except cv2.error:
                pass
 
    def _select_keys(self, data: dict) -> dict:
        if self.record_keys is not None:
            return {k: data[k] for k in self.record_keys if k in data}
        return {k: v for k, v in data.items() if k not in self.exclude_keys}
 
    def _teardown(self):
 
        # Commands that arrived after the last item
        self._poll_commands()
 
        # Recording still running when the process was stopped
        if self._rec_active:
            if self.save_on_stop:
                self._save(datetime.now())
            else:
                self._discard()
 
        # Do not exit before every file is completely written
        for writer in self._rec_writers:
            writer.join()
 
        # Destroy the window (PlotTimeSeries_Process)
        super()._teardown()