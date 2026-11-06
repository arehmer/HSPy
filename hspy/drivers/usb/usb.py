# -*- coding: utf-8 -*-
"""
USB driver for Heimann Sensor HTPA thermopile array devices
(USB bulk interface, VID 0x32A7 / PID 0x0003).

Follows the same pattern as the I2C driver (hspy.drivers.i2c.I2C_HTPA32x32d):
one class that owns the connection and provides every method needed to talk
to the device. Threads (hspy.ipc.threads.USB_Client) and processes
(hspy.ipc.processes.USB_ClientProcess) only *use* this class; they contain no
device-specific logic themselves.

Replaces hspy.drivers.HSComUSB.HS_USBCom (+ StringHelper).

Typical usage::

    from hspy.drivers.usb import USB_HTPA

    # __init__ opens the device and calls init() (reads version + config,
    # determines ArrayType and the stream frame length)
    with USB_HTPA(serial_number='...') as dev:
        print(dev.version, dev.ArrayType)
        dev.start_stream()                 # b'K' -> pixel values in dK
        data = dev.acquire_frame()
        img = data['pix_dK'].reshape(dev.height, dev.width)
        dev.stop_stream()

Linux: grant access to the device with a udev rule, e.g.
    /etc/udev/rules.d/99-htpa.rules
    SUBSYSTEM=="usb", ATTRS{idVendor}=="32a7", ATTRS{idProduct}=="0003", MODE="0666"

Dependencies:
    pip install pyusb libusb-package
"""

import errno
import re
import struct
import sys
import time
from datetime import datetime
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import usb.core
import usb.util
import usb.backend.libusb1
import libusb_package

from hspy.tparray import TPArray


# ── USB identification ───────────────────────────────────────────────────────
_VID       = 0x32A7
_PID       = 0x0003
_LANGID_EN = 0x0409   # English, used to read the serial-number string
_INTERFACE = 0
_LIBUSB_ERROR_NO_DEVICE = -4   # libusb: "No such device (it may have been disconnected)"

# ── Device commands (ASCII) ──────────────────────────────────────────────────
_CMD_VERSION      = b'v'                    # firmware version string
_CMD_CONFIG       = b'G'                    # configuration string ("... Arraytype N ...")
_CMD_STREAM_DK    = b'K'                    # start stream, pixel values in dK
_CMD_STREAM_STOP  = b'x'                    # stop stream, device answers "STOP!"
_CMD_CALIB_READ   = b'n'                    # dump calibration data
_CMD_CALIB_WRITE  = b'Set EEPROM data\r\n'  # enter calibration-write mode

# ── Device responses ─────────────────────────────────────────────────────────
_RESP_STOP              = b'STOP'
_RESP_CALIB_WRITE_READY = 'Setting BCC'
_RESP_CALIB_CHUNK_ACK   = 'WrC'
_RESP_CALIB_WRITE_DONE  = 'Write was successful'

# ── Stream framing ───────────────────────────────────────────────────────────
_SYNC_WORDS = (0xB0D0, 0xF046)                  # last two uint16 words of every frame

# TPArray.DataCols group -> key in the dict returned by acquire_frame()
# (names as in the I2C driver where it has the same quantity)
_FRAME_PART_KEYS = {'e_off': 'eloff', 'vdd': 'vdd', 'T_amb': 'Tamb',
                    'PTAT': 'ptat', 'ATC': 'atc'}
_SYNC_BYTES = struct.pack('<2H', *_SYNC_WORDS)  # same, as they appear on the wire
_N_SYNC     = len(_SYNC_WORDS)

# ── Transfer sizes ───────────────────────────────────────────────────────────
_MAX_RESP_BYTES   = 1024   # max. size of one text response
_CALIB_CHUNK      = 1024   # fixed chunk size of calibration read / write
_MAX_BULK_PACKETS = 64     # max. packets requested per bulk read

# ── Timeouts / delays ────────────────────────────────────────────────────────
_T_DEFAULT_MS     = 300    # frame reads
_T_QUERY_MS       = 6000   # command -> response ('G' needs > 2 s)
_T_STOP_MS        = 2000   # waiting for "STOP!" after the stop command
_T_CALIB_REQ_MS   = 3000   # answer to the calibration-write request
_T_CALIB_READ_MS  = 3000   # per calibration chunk
_T_CALIB_ACK_MS   = 5000   # per calibration-write acknowledge
_T_LEFTOVER_MS    = 500    # trailing text after a calibration read
_T_FLUSH_MS       = 50     # flush_input(): stop once nothing arrived for this long
_T_IDLE_MS        = 500    # reset(): silence that proves the stream is stopped
                           # (must be longer than the frame interval of the device)
_T_FLUSH_MAX_S    = 5.0    # flush_input(): give up if data keeps coming for this long
_T_STOP_ATTEMPT_S = 2.0    # reset(): time per stop attempt before re-sending the stop command
_N_STOP_ATTEMPTS  = 3      # reset(): stop attempts before giving up
_T_RETRY_S        = 0.3    # delay between USB retries
_N_SET_CONFIG     = 5      # set_configuration() attempts
_N_READ_SN        = 3      # serial-number read attempts

RELEASE_DELAY_S   = 0.3    # Windows needs time after dispose_resources() before re-opening


class USB_Driver:
    pass


# ─────────────────────────────────────────────────────────────────────────────
class USB_HTPAError(Exception):
    """Raised on USB communication or data errors."""


class USB_HTPADisconnected(USB_HTPAError):
    """The device is gone (unplugged, powered off, cable fault). Every
    further access fails the same way; open a new driver once the device is
    back."""


class USB_HTPATimeout(USB_HTPAError):
    """The device did not send anything within the timeout."""


class USB_HTPASyncError(USB_HTPAError):
    """A frame did not end with the sync words. The frame is dropped and the
    stream is re-aligned, so the next read_frame() returns a valid frame."""


class USB_HTPACalibWriteError(USB_HTPAError):
    """Writing calibration data failed part-way."""

    def __init__(self, msg: str, bytes_written: int, total: int):
        super().__init__(msg)
        self.bytes_written = bytes_written
        self.total = total


# ─────────────────────────────────────────────────────────────────────────────
class USB_HTPA(USB_Driver):
    """
    Full USB driver for HTPA devices with the Heimann USB bulk interface.

    Thread safety: one instance must only be used from one thread at a time
    (exactly like the I2C driver). USB_Client / USB_ClientProcess guarantee
    that by doing all device access from their own loop.
    """

    RELEASE_DELAY_S = RELEASE_DELAY_S

    # ── Construction / context manager ───────────────────────────────────────

    def __init__(self,
                 serial_number: Optional[str] = None,
                 idVendor: int = _VID,
                 idProduct: int = _PID,
                 timeout_ms: int = _T_DEFAULT_MS,
                 auto_init: bool = True):
        """
        Parameters
        ----------
        serial_number : USB serial number of the device to open. None opens
                        the first device found.
        idVendor, idProduct : USB IDs.
        timeout_ms    : Default timeout for frame reads.
        auto_init     : If True (default) call init() right after opening,
                        i.e. query version + configuration and determine
                        ArrayType and frame length - analogous to the I2C
                        driver, which is ready to acquire frames as soon as
                        it is constructed. Set False if something else (e.g.
                        a GUI) takes care of the device configuration.
        """
        super(USB_HTPA, self).__init__()

        self.serial_number = serial_number
        self._vid = idVendor
        self._pid = idProduct
        self._timeout_ms = timeout_ms

        self._dev = None
        self._ep_in = None
        self._ep_out = None
        self._mps = 64                 # wMaxPacketSize of ep_in, set in open()

        # Receive buffer for the stream: bytes read from USB but not yet
        # returned as part of a frame (keeps the stream frame-aligned).
        self._rx = bytearray()
        self._disconnected = False     # set once libusb reports the device as gone
        self._streaming = False
        self._stream_cmd: Optional[bytes] = None

        # Populated by init() / set_array_type()
        self.version: Optional[str] = None
        self.config: Optional[str] = None
        self.ArrayType: Optional[int] = None
        self.tparray: Optional[TPArray] = None
        self.frame_len: Optional[int] = None   # uint16 words per frame, incl. sync words
        self._frame_parts: List[Tuple[str, int, int]] = []   # (key, start, stop) per part

        # Open the device
        self.open()

        # Initialize the device
        if auto_init:
            try:
                self.init()
            except Exception:
                self.close()
                raise

    def __enter__(self) -> "USB_HTPA":
        self.open()                    # no-op if already open
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __repr__(self) -> str:
        state = 'streaming' if self._streaming else ('open' if self.is_open else 'closed')
        return (f"USB_HTPA(SN={self.serial_number}, ArrayType={self.ArrayType}, "
                f"{state})")

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def is_open(self) -> bool:
        return self._dev is not None

    @property
    def is_connected(self) -> bool:
        """False once a read or write reported that the device is gone."""
        return self._dev is not None and not self._disconnected

    @property
    def is_streaming(self) -> bool:
        return self._streaming

    @property
    def stream_cmd(self) -> Optional[bytes]:
        return self._stream_cmd

    @property
    def width(self) -> Optional[int]:
        return None if self.tparray is None else self.tparray.width

    @property
    def height(self) -> Optional[int]:
        return None if self.tparray is None else self.tparray.height

    @property
    def n_pixels(self) -> Optional[int]:
        return None if self.tparray is None else self.tparray.width * self.tparray.height

    # ── Device discovery ─────────────────────────────────────────────────────

    @staticmethod
    def find_all_devices(idVendor: int = _VID,
                         idProduct: int = _PID,
                         retries: int = 3) -> List[str]:
        """Return the serial numbers of all connected devices."""
        backend = USB_HTPA._get_backend()
        serials: List[str] = []
        warnings: List[str] = []

        for attempt in range(retries):
            devices = list(usb.core.find(idVendor=idVendor, idProduct=idProduct,
                                         find_all=True, backend=backend))
            serials, warnings = [], []
            for d in devices:
                try:
                    sn = usb.util.get_string(d, d.iSerialNumber, langid=_LANGID_EN)
                    print(f"Found: VID=0x{idVendor:04X} PID=0x{idProduct:04X} SN={sn}")
                    if sn is not None:
                        serials.append(sn)
                except Exception as e:
                    warnings.append(f"Warning reading SN: {e}")
                finally:
                    try:
                        usb.util.dispose_resources(d)   # opened again in open()
                    except Exception:
                        pass
            if serials:
                return serials
            # Nothing found - device may not be released yet
            if attempt < retries - 1:
                print(f"No devices found (attempt {attempt + 1}/{retries}), retrying in 1s...")
                for w in warnings:
                    print(w)
                time.sleep(1.0)

        for w in warnings:
            print(w)
        return serials

    # ── Open / close ─────────────────────────────────────────────────────────

    def open(self) -> None:
        """Find the device, claim it and determine the bulk endpoints."""
        if self._dev is not None:
            return

        backend = self._get_backend()

        if self.serial_number is not None:
            dev = self._find_by_serial(backend)
        else:
            dev = usb.core.find(idVendor=self._vid, idProduct=self._pid, backend=backend)
            if dev is None:
                raise USB_HTPAError(
                    f"Device not found! (VID=0x{self._vid:04X}, PID=0x{self._pid:04X})")
            self.serial_number = self._read_serial(dev, attempts=1)

        self._dev = dev
        try:
            self._claim()
        except Exception:
            usb.util.dispose_resources(dev)
            self._dev = None
            raise

    def close(self) -> None:
        """Stop a running stream, then release the device."""
        if self._dev is None:
            return
        if self._streaming and not self._disconnected:   # a lost device can't be stopped
            try:
                self.stop_stream()
            except Exception as e:
                print(f"[{self.serial_number}] Stopping stream on close failed: {e}")
        try:
            usb.util.release_interface(self._dev, _INTERFACE)
            if sys.platform != 'win32':
                try:
                    self._dev.attach_kernel_driver(_INTERFACE)
                except Exception:
                    pass                      # no kernel driver present
        except Exception:
            pass
        finally:
            usb.util.dispose_resources(self._dev)
            self._dev = None
            self._ep_in = None
            self._ep_out = None
            self._rx.clear()
            self._streaming = False
            self._stream_cmd = None

    # ── High-level public API ────────────────────────────────────────────────

    def init(self, reset: bool = True) -> None:
        """
        Initialise the driver:
          1. Put the device into command mode (reset(), optional).
          2. Read the firmware version ('v').
          3. Read the configuration string ('G') and extract the ArrayType.
          4. Determine the array geometry and the stream frame length.
        """
        if reset:
            self.reset()

        self.version = self.query(_CMD_VERSION).strip()
        self.config = self.query(_CMD_CONFIG)

        array_type = self.parse_array_type(self.config)
        if array_type is None:
            raise USB_HTPAError(
                f"'Arraytype' not found in configuration string: {self.config!r}")

        self.set_array_type(array_type)

    def set_array_type(self, array_type: int) -> None:
        """Set ArrayType manually (e.g. when auto_init=False)."""
        self.ArrayType = int(array_type)
        self.tparray = TPArray(ArrayType=self.ArrayType)
        self.frame_len = self.calc_frame_len(self.tparray)
        self._frame_parts = self._calc_frame_parts(self.tparray, self.frame_len)

    @staticmethod
    def _calc_frame_parts(tparray: TPArray, frame_len: int) -> List[Tuple[str, int, int]]:
        """
        Where each part of a stream frame lies, from the serial data order in
        TPArray.DataCols:  pix | e_off | vdd | T_amb | PTAT | ATC.
        Returns [(key in acquire_frame()'s dict, start, stop), ...] for the
        non-pixel parts, or [] if TPArray's layout does not add up to the
        frame length (then only 'aux' is available).
        """
        groups = tparray.DataCols.as_dict()
        if sum(len(cols) for cols in groups.values()) + _N_SYNC != frame_len:
            print(f"Warning: TPArray data layout of ArrayType {tparray.ArrayType} does not "
                  f"match the frame length ({frame_len} words) - PTAT etc. not available.")
            return []
        parts, pos = [], 0
        for group, cols in groups.items():
            if group != 'pix' and cols:
                parts.append((_FRAME_PART_KEYS.get(group, group), pos, pos + len(cols)))
            pos += len(cols)
        return parts

    @staticmethod
    def calc_frame_len(tparray: TPArray) -> int:
        """Number of uint16 words per stream frame, including the 2 sync words."""
        n_pix = tparray.width * tparray.height
        dc = tparray.DevConst

        if tparray.DesignGen <= 3:
            n = (n_pix
                 + dc['NROFBLOCKS'] * dc['NROFPTAT']
                 + n_pix // dc['NROFBLOCKS']
                 + dc['NROFATC'] + 2)
        elif tparray.DesignGen == 4:
            n = n_pix + (dc['NROFBLOCKS'] + dc['NROFPTAT']) + 2
        else:
            raise USB_HTPAError(f"Unsupported DesignGen {tparray.DesignGen}")

        return n + _N_SYNC

    @staticmethod
    def parse_array_type(config: Optional[str]) -> Optional[int]:
        """Extract the ArrayType from the 'G' configuration string."""
        if not config:
            return None
        match = re.search(r'Arraytype\s+(\d+)', config)
        return int(match.group(1)) if match else None

    @classmethod
    def probe_array_type(cls, serial_number: Optional[str] = None, **kwargs) -> int:
        """
        Open the device, read its ArrayType and release it again. Useful in a
        parent process that needs the ArrayType (e.g. for ImshowProcess) while
        the device itself is opened inside a child process.
        """
        with cls(serial_number=serial_number, auto_init=True, **kwargs) as dev:
            array_type = dev.ArrayType
        time.sleep(RELEASE_DELAY_S)
        return array_type

    # ── Stream control ───────────────────────────────────────────────────────

    def reset(self,
              idle_ms: int = _T_IDLE_MS,
              attempts: int = _N_STOP_ATTEMPTS,
              attempt_s: float = _T_STOP_ATTEMPT_S) -> int:
        """
        Put the device into command mode, whatever state it is in.

        A program that was killed while streaming leaves the device
        streaming. Commands sent in that state are answered with frame data
        instead of text. reset() sends the stop command and discards
        everything (frames, "STOP!", any other reply) until the device has
        been silent for idle_ms. If data keeps coming, the stop command is
        sent again (up to `attempts` times).

        Before that, both endpoints are reset (clear_halt). If a process was
        killed in the middle of a transfer, host and device can disagree on
        the USB data toggle; the device then acknowledges but silently drops
        the next packet it receives - which would be the stop command.
        clear_halt resets the toggle on both sides.

        Returns the number of bytes discarded.
        """
        self._streaming = False
        self._stream_cmd = None
        self._rx.clear()

        self._clear_halt(self._ep_out, self._ep_in)

        discarded = 0
        for attempt in range(1, attempts + 1):
            self.write(_CMD_STREAM_STOP)
            n, quiet = self._drain(quiet_ms=idle_ms, max_s=attempt_s)
            discarded += n
            if quiet:
                if discarded:
                    print(f"[{self.serial_number}] reset: discarded {discarded} bytes "
                          f"of pending data (stop attempt {attempt})")
                return discarded
            print(f"[{self.serial_number}] reset: device still streaming after stop "
                  f"attempt {attempt}/{attempts}")
            self._clear_halt(self._ep_out)

        raise USB_HTPAError(
            f"Device keeps streaming after {attempts} stop attempts "
            f"({discarded} bytes discarded). Unplug and reconnect the device.")

    def start_stream(self,
                     cmd: Union[bytes, str] = _CMD_STREAM_DK,
                     frame_len: Optional[int] = None) -> Optional[str]:
        """
        Start the continuous frame stream.

        Parameters
        ----------
        cmd       : Stream command, b'K' (pixel values in dK, default) or
                    another stream command such as b't'.
        frame_len : Frame length in uint16 words incl. sync words. Only
                    needed if init() was not called (auto_init=False).

        Returns
        -------
        The device's text response, or None if it did not send one.
        """
        cmd = cmd.encode() if isinstance(cmd, str) else bytes(cmd)

        if frame_len is not None:
            self.frame_len = int(frame_len)
        if self.frame_len is None:
            raise USB_HTPAError("Frame length unknown: call init() or pass frame_len.")
        if self._streaming:
            raise USB_HTPAError("Stream already running.")

        self._rx.clear()
        try:
            resp = self.query(cmd)
        except USB_HTPATimeout:
            resp = None                # some firmware starts streaming without a reply

        self._streaming = True
        self._stream_cmd = cmd
        return resp

    def stop_stream(self,
                    cmd: Union[bytes, str] = _CMD_STREAM_STOP,
                    timeout_ms: int = _T_STOP_MS
                    ) -> Tuple[Optional[str], List[np.ndarray]]:
        """
        Stop the stream and wait for the device's "STOP!" response.

        Returns
        -------
        (response, leftover_frames)
            response        : stop response text, or None if none arrived
                              within timeout_ms.
            leftover_frames : complete frames that arrived between the stop
                              command and the response (sync words removed).
        """
        self.write(cmd)
        self._streaming = False
        self._stream_cmd = None

        # Collect everything until "STOP" shows up (frames still in flight
        # and the response may arrive in the same transfer)
        deadline = time.monotonic() + timeout_ms / 1e3
        stop_idx = -1
        while True:
            remaining_ms = int((deadline - time.monotonic()) * 1e3)
            if remaining_ms <= 0:
                break
            try:
                self._rx += self.read(self._mps * _MAX_BULK_PACKETS, remaining_ms)
            except USB_HTPATimeout:
                break
            stop_idx = self._rx.rfind(_RESP_STOP)
            if stop_idx >= 0:
                break

        if stop_idx >= 0:
            payload = bytes(self._rx[:stop_idx])
            response = self._rx[stop_idx:].decode('utf-8', errors='replace').strip()
        else:
            payload = bytes(self._rx)
            response = None
        self._rx.clear()

        return response, self._split_frames(payload)

    # ── Frame acquisition ────────────────────────────────────────────────────

    def read_frame(self, timeout_ms: Optional[int] = None) -> np.ndarray:
        """
        Read one raw stream frame.

        Returns
        -------
        np.ndarray (uint16), frame_len - 2 words (sync words removed).

        Raises
        ------
        USB_HTPATimeout   : no data within the timeout
        USB_HTPASyncError : frame did not end with the sync words. The broken
                            frame is dropped and the stream re-aligned, so the
                            next call returns a valid frame.
        """
        if self.frame_len is None:
            raise USB_HTPAError("Frame length unknown: call init() or set_array_type().")

        t = self._timeout_ms if timeout_ms is None else timeout_ms
        n_bytes = self.frame_len * 2

        while len(self._rx) < n_bytes:
            need = n_bytes - len(self._rx)
            # Request whole packets only (avoids overflow errors), extra
            # bytes stay in the receive buffer for the next frame
            size = min(-(-need // self._mps) * self._mps, self._mps * _MAX_BULK_PACKETS)
            try:
                self._rx += self.read(size, t)
            except USB_HTPATimeout:
                self._clear_halt(self._ep_in)
                raise

        if self._rx[n_bytes - len(_SYNC_BYTES):n_bytes] == _SYNC_BYTES:
            frame = np.frombuffer(bytes(self._rx[:n_bytes]), dtype='<u2')[:-_N_SYNC].copy()
            del self._rx[:n_bytes]
            return frame

        got = struct.unpack('<2H', bytes(self._rx[n_bytes - 4:n_bytes]))
        dropped = self._resync()
        raise USB_HTPASyncError(
            f"Sync error: expected 0x{_SYNC_WORDS[0]:04X} 0x{_SYNC_WORDS[1]:04X}, "
            f"got 0x{got[0]:04X} 0x{got[1]:04X} - dropped {dropped} bytes to re-align")

    def acquire_frame(self, timeout_ms: Optional[int] = None) -> dict:
        """
        Acquire one frame from the running stream and split it up. Intended
        to be called once per loop iteration by USB_Client / USB_ClientProcess,
        exactly like I2C_HTPA32x32d.acquire_frame().

        Returns
        -------
        dict
            'pix'           : np.ndarray uint16, width*height pixel values
                              (flattened, row-major - same convention as the
                              I2C driver)
            'pix_dK'        : same array, only present for the b'K' stream
                              (values are in dK)
            'aux'           : remaining non-pixel words of the frame (offsets,
                              PTAT, VDD, ...; layout depends on DesignGen)
            'eloff', 'vdd', 'Tamb', 'ptat', 'atc'
                            : the parts of 'aux', split according to
                              TPArray.DataCols (np.ndarray each; a part the
                              array type does not have is missing)
            'ptat_avg'      : mean of 'ptat' (float), same key as in the
                              I2C driver
            'frame_raw'     : the complete frame without sync words
            't'             : datetime of acquisition
            'serial_number' : device serial number
            'ArrayType'     : device ArrayType

            Does NOT contain 'success' or 'image_id' - the calling thread /
            process adds those.
        """
        if not self._streaming:
            raise USB_HTPAError("Stream not running: call start_stream() first.")
        if self.n_pixels is None:
            raise USB_HTPAError("ArrayType unknown: call init() or set_array_type().")

        frame = self.read_frame(timeout_ms)
        n = self.n_pixels

        data = {'pix': frame[:n],
                'aux': frame[n:],
                'frame_raw': frame,
                't': datetime.now(),
                'serial_number': self.serial_number,
                'ArrayType': self.ArrayType}

        if self._stream_cmd == _CMD_STREAM_DK:
            data['pix_dK'] = data['pix']

        # Non-pixel parts of the frame: 'eloff', 'vdd', 'Tamb', 'ptat', 'atc'
        # (views into frame_raw, no copies), plus the PTAT average
        for key, start, stop in self._frame_parts:
            data[key] = frame[start:stop]
        if 'ptat' in data:
            data['ptat_avg'] = float(data['ptat'].mean())

        return data

    # ── Calibration data ─────────────────────────────────────────────────────

    def read_calibration(self,
                         size: int,
                         cmd: Union[bytes, str] = _CMD_CALIB_READ,
                         chunk_size: int = _CALIB_CHUNK,
                         timeout_ms: int = _T_CALIB_READ_MS,
                         on_chunk: Optional[Callable[[bytes], None]] = None) -> bytes:
        """
        Read `size` bytes of calibration data.

        on_chunk : optional callback, called with every chunk as it arrives
                   (e.g. to forward chunks to a GUI).
        """
        self._check_not_streaming()
        self.write(cmd)

        data = bytearray()
        while len(data) < size:
            n = min(size - len(data), chunk_size)     # last chunk may be smaller
            try:
                chunk = self.read(n, timeout_ms)
            except USB_HTPATimeout as e:
                raise USB_HTPATimeout(
                    f"Timeout reading calibration data, received {len(data)}/{size} bytes") from e
            if not chunk:
                continue                              # ignore zero-length packets
            data += chunk
            if on_chunk is not None:
                on_chunk(chunk)

        # Check whether the device sends anything else
        try:
            leftover = self.read_text(_T_LEFTOVER_MS).strip()
            if leftover:
                print(f"[{self.serial_number}] Leftover after calib read: {leftover}")
        except USB_HTPATimeout:
            pass

        return bytes(data)

    def write_calibration(self,
                          data: bytes,
                          chunk_size: int = _CALIB_CHUNK,
                          ack_timeout_ms: int = _T_CALIB_ACK_MS,
                          on_progress: Optional[Callable[[int, int], None]] = None) -> None:
        """
        Write calibration data to the device.

        on_progress : optional callback(bytes_written, total), called after
                      every acknowledged chunk.

        Raises
        ------
        USB_HTPAError           : device did not accept the write request
        USB_HTPACalibWriteError : write failed part-way (.bytes_written, .total)
        """
        self._check_not_streaming()
        total = len(data)

        # Step 1: enter write mode
        resp = self.query(_CMD_CALIB_WRITE, timeout_ms=_T_CALIB_REQ_MS)
        if _RESP_CALIB_WRITE_READY not in resp:
            raise USB_HTPAError(f"Unexpected response to calibration-write request: {resp!r}")

        # Step 2: send chunks, wait for 'WrCxx' after each one
        sent = 0
        while sent < total:
            n = min(total - sent, chunk_size)
            self.write(data[sent:sent + n])

            try:
                ack = self.read_text(ack_timeout_ms).strip()
            except USB_HTPATimeout as e:
                raise USB_HTPACalibWriteError(
                    f"Timeout waiting for ack of chunk {sent // chunk_size}",
                    sent, total) from e

            if _RESP_CALIB_WRITE_DONE in ack:
                return
            if _RESP_CALIB_CHUNK_ACK in ack:
                sent += n
                if on_progress is not None:
                    on_progress(sent, total)
            else:
                raise USB_HTPACalibWriteError(f"Unexpected ack: {ack!r}", sent, total)

        raise USB_HTPACalibWriteError(
            f"All data sent but no '{_RESP_CALIB_WRITE_DONE}' received", sent, total)

    # ── Low-level I/O ────────────────────────────────────────────────────────

    def write(self, data: Union[bytes, str]) -> None:
        """Send raw bytes / a command without waiting for an answer."""
        self._check_open()
        data = data.encode() if isinstance(data, str) else data
        try:
            self._ep_out.write(data)
        except usb.core.USBTimeoutError as e:
            raise USB_HTPATimeout(f"USB write timed out: {e}") from e
        except usb.core.USBError as e:
            raise self._usb_error(e, 'write') from e

    def read(self, size: int = _MAX_RESP_BYTES, timeout_ms: Optional[int] = None) -> bytes:
        """Read up to `size` raw bytes from the IN endpoint."""
        self._check_open()
        t = self._timeout_ms if timeout_ms is None else timeout_ms
        try:
            return bytes(self._ep_in.read(size, timeout=t))
        except usb.core.USBTimeoutError as e:
            raise USB_HTPATimeout(f"No data within {t} ms") from e
        except usb.core.USBError as e:
            raise self._usb_error(e, 'read') from e

    def _usb_error(self, e: 'usb.core.USBError', what: str) -> USB_HTPAError:
        """Translate a pyusb error; a vanished device becomes USB_HTPADisconnected."""
        if (getattr(e, 'backend_error_code', None) == _LIBUSB_ERROR_NO_DEVICE
                or getattr(e, 'errno', None) == errno.ENODEV):
            self._disconnected = True
            return USB_HTPADisconnected(f"Device disconnected ({what}): {e}")
        return USB_HTPAError(f"USB {what} error: {e}")

    def read_text(self, timeout_ms: Optional[int] = None) -> str:
        """Read one text response without sending anything first."""
        return self.read(_MAX_RESP_BYTES, timeout_ms).decode('utf-8', errors='replace')

    def query(self, cmd: Union[bytes, str], timeout_ms: int = _T_QUERY_MS) -> str:
        """Send a command and return the device's text response."""
        self.write(cmd)
        try:
            return self.read_text(timeout_ms)
        except USB_HTPATimeout:
            self._clear_halt(self._ep_in, self._ep_out)
            raise

    def flush_input(self,
                    quiet_ms: int = _T_FLUSH_MS,
                    max_s: float = _T_FLUSH_MAX_S) -> int:
        """
        Discard pending input until nothing arrived for quiet_ms.
        Returns the number of bytes discarded.

        Raises USB_HTPAError if data keeps arriving for max_s (the device is
        still streaming).
        """
        n, quiet = self._drain(quiet_ms, max_s)
        if quiet:
            return n
        raise USB_HTPAError(
            f"Input did not go quiet within {max_s} s ({n} bytes discarded) - "
            f"device still streaming?")

    def _drain(self, quiet_ms: int, max_s: float) -> Tuple[int, bool]:
        """Discard input until nothing arrived for quiet_ms, at most max_s.
        Returns (bytes discarded, True if the input went quiet)."""
        n = len(self._rx)
        self._rx.clear()
        deadline = time.monotonic() + max_s
        while time.monotonic() < deadline:
            try:
                n += len(self.read(self._mps * _MAX_BULK_PACKETS, quiet_ms))
            except USB_HTPATimeout:
                return n, True
        return n, False

    # ── Private helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _get_backend():
        # Explicit backend, works on Windows and Linux
        backend = usb.backend.libusb1.get_backend(find_library=libusb_package.find_library)
        if backend is None:
            raise USB_HTPAError("libusb backend not found")
        return backend

    @staticmethod
    def _read_serial(dev, attempts: int = _N_READ_SN) -> Optional[str]:
        """Read the serial number WITHOUT set_configuration()."""
        for attempt in range(attempts):
            try:
                return usb.util.get_string(dev, dev.iSerialNumber, langid=_LANGID_EN)
            except Exception:
                if attempt < attempts - 1:
                    time.sleep(_T_RETRY_S)
        return None

    def _find_by_serial(self, backend):
        devices = list(usb.core.find(idVendor=self._vid, idProduct=self._pid,
                                     find_all=True, backend=backend))
        unreadable = 0
        for d in devices:
            sn = self._read_serial(d)
            if sn == self.serial_number:
                return d
            if sn is None:
                unreadable += 1
            usb.util.dispose_resources(d)        # not the requested device

        hint = f" ({unreadable} device(s) with unreadable SN)" if unreadable else ""
        raise USB_HTPAError(
            f"Device not found! (VID=0x{self._vid:04X}, PID=0x{self._pid:04X}, "
            f"SN={self.serial_number}){hint}")

    def _claim(self) -> None:
        dev = self._dev

        # Kernel driver only relevant on Linux
        if sys.platform != 'win32':
            if dev.is_kernel_driver_active(_INTERFACE):
                dev.detach_kernel_driver(_INTERFACE)

        for attempt in range(_N_SET_CONFIG):
            try:
                dev.set_configuration()
                break
            except usb.core.USBError:
                if attempt < _N_SET_CONFIG - 1:
                    print(f"set_configuration attempt {attempt + 1} failed, retrying...")
                    time.sleep(_T_RETRY_S)
                else:
                    raise

        intf = dev.get_active_configuration()[(_INTERFACE, 0)]
        self._ep_out = usb.util.find_descriptor(
            intf, custom_match=lambda e:
                usb.util.endpoint_direction(e.bEndpointAddress) == usb.util.ENDPOINT_OUT)
        self._ep_in = usb.util.find_descriptor(
            intf, custom_match=lambda e:
                usb.util.endpoint_direction(e.bEndpointAddress) == usb.util.ENDPOINT_IN)

        if self._ep_out is None or self._ep_in is None:
            raise USB_HTPAError("Bulk endpoints not found")

        self._mps = self._ep_in.wMaxPacketSize
        print(f"[{self.serial_number}] Connected: ep_out=0x{self._ep_out.bEndpointAddress:02X}, "
              f"ep_in=0x{self._ep_in.bEndpointAddress:02X}")

    def _resync(self) -> int:
        """Drop bytes up to and including the next sync pattern so the
        receive buffer starts at a frame boundary. Returns bytes dropped."""
        idx = self._rx.find(_SYNC_BYTES)
        if idx >= 0:
            dropped = idx + len(_SYNC_BYTES)
            del self._rx[:dropped]
        else:
            keep = len(_SYNC_BYTES) - 1          # pattern may straddle the next read
            dropped = len(self._rx) - keep
            del self._rx[:dropped]
        return dropped

    def _split_frames(self, buf: bytes) -> List[np.ndarray]:
        """Extract all complete, sync-terminated frames from buf."""
        frames: List[np.ndarray] = []
        if self.frame_len is None:
            return frames
        n_bytes = self.frame_len * 2
        pos = 0
        idx = buf.find(_SYNC_BYTES, pos)
        while idx >= 0:
            end = idx + len(_SYNC_BYTES)
            start = end - n_bytes
            if start >= pos:
                frames.append(np.frombuffer(buf[start:end], dtype='<u2')[:-_N_SYNC].copy())
                pos = end
            idx = buf.find(_SYNC_BYTES, end)
        return frames

    def _clear_halt(self, *endpoints) -> None:
        for ep in endpoints:
            try:
                self._dev.clear_halt(ep)
            except Exception:
                pass

    def _check_open(self) -> None:
        if self._dev is None:
            raise USB_HTPAError("Device not open")

    def _check_not_streaming(self) -> None:
        if self._streaming:
            raise USB_HTPAError("Not possible while the stream is running: call stop_stream() first.")