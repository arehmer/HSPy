# -*- coding: utf-8 -*-
"""
The frame dictionary contract of hspy.drivers, checked without a sensor.

Three things are verified here:

1. Every driver's ``acquire_frame()`` has the same name and the same
   (argument-free) signature.
2. A frame honours the contract in ``hspy.drivers._frame``: the keys that
   must always be there are there, 'frame' really is one of the pixel
   stages, and 'frame_unit' names that stage.
3. A frame survives being pickled and pushed through a
   ``SharedMemoryQueue`` -- the hard requirement, since that is how a
   frame reaches a consumer running in another process.

Point 3 is what a driver change is most likely to break silently, and it
needs no hardware: the frames come from ``new_frame()`` with arrays of the
right shape, and the UDP driver is exercised against a stand-in for its
reader rather than a device.

Run with:
    python -m pytest tests/test_frame_contract.py
or directly:
    python tests/test_frame_contract.py
"""

import inspect
import pickle as pkl
from datetime import datetime

import numpy as np

from hspy.drivers._frame import (new_frame, validate_frame, describe_frame,
                                 FrameError, AUX_FROM_DATACOLS)
from hspy.drivers.i2c import I2C_HTPA32x32d
from hspy.drivers.spi import SPI_HTPA160x120dR1
from hspy.drivers.udp import UDP_HTPA
from hspy.drivers.usb import USB_HTPA
from hspy.ipc.processes_base import SharedMemoryQueue
from hspy.tparray import TPArray


DRIVERS = (I2C_HTPA32x32d, SPI_HTPA160x120dR1, UDP_HTPA, USB_HTPA)


# ── 1. One method, one signature ─────────────────────────────────────────────

def test_every_driver_has_the_same_acquire_frame():
    """
    An acquisition loop calls driver.acquire_frame() and nothing else, so
    the name and the signature have to be the same for all four -- that is
    what lets one client class drive any of them.
    """

    for driver in DRIVERS:
        assert hasattr(driver, 'acquire_frame'), \
            f"{driver.__name__} has no acquire_frame()."

        signature = inspect.signature(driver.acquire_frame)
        assert list(signature.parameters) == ['self'], \
            (f"{driver.__name__}.acquire_frame{signature} takes arguments; "
             f"every per-acquisition option belongs in __init__.")


def test_acquisition_options_are_constructor_arguments():
    """The options that used to be acquire_frame() arguments live in
    __init__ now, and keep_raw exists on every driver."""

    expected = {I2C_HTPA32x32d: ('active_vdd', 'blind_vdd', 'applyCalib',
                                 'keep_raw', 'DevID'),
                SPI_HTPA160x120dR1: ('eloff_every', 'applyCalib',
                                     'keep_raw', 'DevID'),
                UDP_HTPA: ('stream_cmd', 'keep_raw'),
                USB_HTPA: ('timeout_ms', 'keep_raw', 'DevID')}

    for driver, options in expected.items():
        parameters = inspect.signature(driver.__init__).parameters
        missing = [o for o in options if o not in parameters]
        assert not missing, \
            f"{driver.__name__}.__init__ is missing {missing}."


def test_calcdK_is_decided_by_the_driver():
    """calcdK is no constructor argument: I2C and SPI set it in load_lut()
    from the table number, UDP and USB derive it from the stream command."""

    for driver in DRIVERS:
        assert 'calcdK' not in inspect.signature(driver.__init__).parameters, \
            f"{driver.__name__}.__init__ still takes calcdK."

    for driver in (I2C_HTPA32x32d, SPI_HTPA160x120dR1):
        assert list(inspect.signature(driver.load_lut).parameters) == \
            ['self', 'lut'], f"{driver.__name__}.load_lut() has another signature."

    for driver in (UDP_HTPA, USB_HTPA):
        assert isinstance(driver.calcdK, property), \
            f"{driver.__name__}.calcdK is not a property."


# ── 2. The contract itself ───────────────────────────────────────────────────

def _frame(interface='i2c', shape=(32, 32), **kwargs):
    """A frame with arrays of the right size for *shape*."""

    n = shape[0] * shape[1]

    defaults = dict(pix=np.arange(n, dtype=np.uint16),
                    eloff=np.zeros(n // 4, dtype=np.uint16),
                    ptat=np.full(8, 1000, dtype=np.uint16),
                    vdd=np.full(2, 30000, dtype=np.uint16),
                    Tamb=2980.0)
    defaults.update(kwargs)

    return new_frame(interface=interface,
                     ArrayType=10,
                     shape=shape,
                     **defaults)


def test_a_minimal_frame_validates():
    validate_frame(_frame())


def test_frame_aliases_the_best_stage():
    """
    'frame' is the best stage the driver produced and 'frame_unit' says
    which -- a consumer reads 'frame' and trusts 'frame_unit' blindly.
    """

    n = 32 * 32
    digits = np.arange(n, dtype=np.uint16)
    volts = np.arange(n, dtype=np.float64)
    temps = np.full(n, 2980.0)

    only_raw = _frame(pix=digits)
    assert only_raw['frame_unit'] == 'digits'
    assert only_raw['frame'] is only_raw['pix']

    calibrated = _frame(pix=digits, pix_comp=volts)
    assert calibrated['frame_unit'] == 'Ud'
    assert calibrated['frame'] is calibrated['pix_comp']

    mapped = _frame(pix=digits, pix_comp=volts, pix_dK=temps)
    assert mapped['frame_unit'] == 'dK'
    assert mapped['frame'] is mapped['pix_dK']

    for frame in (only_raw, calibrated, mapped):
        validate_frame(frame)


def test_auxiliary_data_is_always_present():
    """
    A key that is missing for one array type and present for another makes
    every consumer probe with 'in data'. An array type without ATC gets an
    empty array, and a missing scalar gets nan.
    """

    frame = _frame(atc=None, Tamb=None)

    assert frame['atc'].shape == (0,)
    assert np.isnan(frame['Tamb'])

    # ptat_avg is derived from 'ptat' when the driver did not compute it
    assert frame['ptat_avg'] == 1000.0

    validate_frame(frame)


def test_arrays_are_flattened():
    """Every array in a frame is 1-D, so that a consumer can reshape with
    frame['shape'] instead of guessing."""

    frame = _frame(pix=np.zeros((32, 32), dtype=np.uint16))

    assert frame['pix'].ndim == 1
    assert frame['frame'].size == 32 * 32

    validate_frame(frame)


def test_raw_is_opt_in():
    """The raw transport payload roughly doubles a pickled frame, so it is
    only there when the driver was asked for it."""

    assert 'raw' not in _frame()
    assert 'raw' in _frame(raw={'pix_raw': b'\x00\x01'})


def test_a_broken_frame_is_rejected():

    # Pixel count that does not match the shape
    try:
        _frame(pix=np.zeros(10, dtype=np.uint16))
    except FrameError:
        pass
    else:
        raise AssertionError('a wrong pixel count was accepted.')

    # No pixel stage at all
    try:
        new_frame(interface='i2c', ArrayType=10, shape=(32, 32))
    except FrameError:
        pass
    else:
        raise AssertionError('a frame without any pixel stage was accepted.')

    # An interface nobody implements
    try:
        _frame(interface='rs232')
    except FrameError:
        pass
    else:
        raise AssertionError('an unknown interface was accepted.')

    # A key that is not part of the contract
    frame = _frame()
    frame['pixels'] = frame['pix']
    try:
        validate_frame(frame)
    except FrameError:
        pass
    else:
        raise AssertionError('an unknown key was accepted.')


# ── 3. The multiprocessing requirement ───────────────────────────────────────

def test_a_frame_pickles():
    """
    Nothing in a frame holds an OS handle, so every value survives
    pickling. This is what makes a frame usable with multiprocessing at
    all.
    """

    for interface, shape in (('i2c', (32, 32)),
                             ('spi', (120, 160)),
                             ('udp', (32, 32)),
                             ('usb', (60, 40))):

        n = shape[0] * shape[1]
        frame = _frame(interface=interface,
                       shape=shape,
                       pix=np.arange(n, dtype=np.uint16),
                       pix_dK=np.full(n, 2980.0),
                       eloff=np.zeros(n // 4, dtype=np.uint16),
                       t=datetime.now())

        restored = pkl.loads(pkl.dumps(frame, protocol=pkl.HIGHEST_PROTOCOL))

        validate_frame(restored)
        assert restored['interface'] == interface
        assert restored['shape'] == shape
        np.testing.assert_array_equal(restored['frame'], frame['frame'])
        assert restored['t'] == frame['t']

        # The alias has to survive the round trip too, or 'frame_unit'
        # starts lying about which array 'frame' is
        assert restored['frame'] is restored['pix_dK']


def test_a_frame_crosses_a_shared_memory_queue():
    """
    The hard requirement: a frame goes through a SharedMemoryQueue, which
    is how it reaches a consumer in another process. Tested in one process
    (put then get) -- what is being checked is the pickling and the slot
    size, not the scheduling.
    """

    queue = SharedMemoryQueue(maxsize=2, slot_bytes=4 * 1024 * 1024)

    try:
        n = 120 * 160
        frame = _frame(interface='spi',
                       shape=(120, 160),
                       pix=np.arange(n, dtype=np.uint16),
                       pix_comp=np.arange(n, dtype=np.float64),
                       pix_dK=np.full(n, 2980.0),
                       eloff=np.zeros(1600, dtype=np.uint16),
                       ptat=np.full(24, 1000, dtype=np.uint16),
                       vdd=np.full(24, 30000, dtype=np.uint16),
                       atc=np.full(2, 500, dtype=np.uint16))

        # Plus what a client adds on top of the driver's frame
        frame['success'] = True
        frame['image_id'] = 7

        queue.put(frame, timeout=1.0)
        restored = queue.get(timeout=1.0)

        assert restored['success'] is True
        assert restored['image_id'] == 7
        assert restored['frame_unit'] == 'dK'
        np.testing.assert_array_equal(restored['frame'], frame['frame'])

        del restored['success'], restored['image_id']
        validate_frame(restored)
    finally:
        queue.close()


def test_the_raw_payload_still_fits_a_slot():
    """
    keep_raw=True is the size case worth checking: a 160x120 frame with its
    raw SPI bytes has to stay inside a slot, or the item is dropped at
    runtime with nothing but a printed warning.
    """

    n = 120 * 160

    raw = {'pix_raw': np.zeros((12, 2, 1607), dtype=np.uint8),
           'eloff_raw': np.zeros((2, 1607), dtype=np.uint8)}

    frame = _frame(interface='spi',
                   shape=(120, 160),
                   pix=np.arange(n, dtype=np.uint16),
                   pix_comp=np.arange(n, dtype=np.float64),
                   pix_dK=np.full(n, 2980.0),
                   eloff=np.zeros(1600, dtype=np.uint16),
                   raw=raw)

    size = len(pkl.dumps(frame, protocol=pkl.HIGHEST_PROTOCOL))

    assert size < 4 * 1024 * 1024, \
        (f"a frame with keep_raw=True pickles to {size / 1e6:.1f} MB, which "
         f"does not fit the 4 MB slots the examples use.")


# ── The UDP driver against a stand-in for its reader ─────────────────────────

class _FakeReader:
    """
    Enough of HTPA_UDPReader for UDP_HTPA.acquire_frame(): it hands out the
    group dictionary that HTPA_ByteStream_Converter produces, keyed the way
    TPArray.DataCols spells the groups.
    """

    def __init__(self, ArrayType=10, port=30444):
        self.tparray = TPArray(ArrayType=ArrayType)
        self.sockets = {}
        self.calls = 0

    def read_continuous_bytestream(self, dev_id, first_frame_timeout=None,
                                   return_raw=False):
        self.calls += 1

        groups = {}
        for group, cols in self.tparray.DataCols.as_dict().items():
            groups[group] = np.full(len(cols), 2980.0)

        if return_raw:
            return groups, [b'\x01\x02']
        return groups


def _udp_driver(**kwargs):
    """A UDP_HTPA that never touches the network."""

    driver = UDP_HTPA.__new__(UDP_HTPA)       # no open(), no socket

    driver._IP = '192.168.0.2'
    driver._DevID = 1172
    driver._Bcast_Addr = ''
    driver._ArrayType_requested = None
    driver._port = 30444
    driver.stream_cmd = kwargs.get('stream_cmd', b'K')
    driver.timeout_s = 1.0
    driver.first_frame_timeout = 5.0
    driver.keep_raw = kwargs.get('keep_raw', False)
    driver._MAC = '00.11.22.33.44.55'
    driver._t_first_frame = None
    driver._bound = True
    driver._streaming = True

    driver._reader = _FakeReader()
    driver.ArrayType = 10
    driver.tparray = driver._reader.tparray

    return driver


def test_udp_driver_builds_a_contract_frame():
    """
    The UDP path is the one that had to translate key names
    (TPArray.DataCols spells them e_off / T_amb / PTAT / ATC), so it is
    worth checking against a stand-in reader rather than only on paper.
    """

    driver = _udp_driver(stream_cmd=b'K')
    frame = driver.acquire_frame()

    validate_frame(frame)

    assert frame['interface'] == 'udp'
    assert frame['DevID'] == 1172
    assert frame['serial'] == '00.11.22.33.44.55'
    assert frame['shape'] == (32, 32)

    # b'K' streams dK, so that is the stage and the unit
    assert frame['frame_unit'] == 'dK'
    assert 'pix_dK' in frame
    assert frame['frame'] is frame['pix_dK']

    # The aux groups arrived under the drivers' own names, not DataCols'
    for group, key in AUX_FROM_DATACOLS.items():
        assert key in frame, f"{group} was not translated to {key}."
    assert frame['eloff'].size == 256      # 32x32: 256 electrical offsets
    assert frame['ptat'].size == 8

    # And it still pickles
    validate_frame(pkl.loads(pkl.dumps(frame)))


def test_udp_driver_reports_digits_for_other_stream_commands():
    """Only b'K' streams dK; anything else is digits, and the frame says so
    rather than silently calling raw values temperatures."""

    frame = _udp_driver(stream_cmd=b't').acquire_frame()

    assert frame['frame_unit'] == 'digits'
    assert 'pix' in frame
    assert 'pix_dK' not in frame

    validate_frame(frame)


def test_udp_driver_keeps_raw_on_request():

    frame = _udp_driver(keep_raw=True).acquire_frame()

    assert 'raw' in frame
    assert frame['raw'] == [b'\x01\x02']

    validate_frame(frame)


# ── Running this file directly ───────────────────────────────────────────────

if __name__ == '__main__':

    tests = [value for name, value in sorted(globals().items())
             if name.startswith('test_') and callable(value)]

    failures = []

    for test in tests:
        try:
            test()
        except Exception as e:
            failures.append((test.__name__, e))
            print(f"FAIL  {test.__name__}: {e}")
        else:
            print(f"ok    {test.__name__}")

    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed.")

    if failures:
        raise SystemExit(1)

    # One frame printed in full, as a readable description of the contract
    print('\nExample frame:')
    print(' ', describe_frame(_udp_driver().acquire_frame()))
