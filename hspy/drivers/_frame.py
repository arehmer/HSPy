# -*- coding: utf-8 -*-
"""
The frame dictionary every driver in hspy.drivers returns.

Every driver's ``acquire_frame()`` returns the same flat dictionary, whatever
the transport underneath is (I2C, SPI, UDP, USB). This module owns that
contract: the key names, their meaning, and the one function that assembles
them, so that four drivers cannot drift apart again.

Why flat, and why only these types
-----------------------------------
The dictionary is handed straight to a ``queue.Queue`` (threads) or a
``multiprocessing`` queue / ``SharedMemoryQueue`` (processes), and the latter
pickles every item. So:

* No nesting. A flat dict pickles cheaply and no consumer has to walk a tree
  to find a value.
* Only ``numpy`` arrays, Python scalars, ``str``, ``tuple`` and ``datetime``.
  Never a socket, an ``SMBus`` / ``spidev`` handle, a ``LuT``, or a pandas
  object -- the first two cannot cross a process boundary at all, and the
  last two are needlessly expensive to pickle once per frame.
* Size matters as much as type: a ``SharedMemoryQueue`` slot is a fixed number
  of bytes, so the raw transport payload (which roughly doubles the item) is
  only included when the driver was constructed with ``keep_raw=True``.

The keys
--------
Identity, always present::

    'interface'  : str            'i2c' | 'spi' | 'udp' | 'usb'
    'ArrayType'  : int
    'shape'      : (height, width)
    'DevID'      : int            -1 when the interface has no device number
    'serial'     : str            USB serial / UDP MAC-ID / '' for i2c and spi

Time, always present::

    't'          : datetime       wall clock, end of acquisition
    't_mono'     : float          time.monotonic() at the same instant

The image, always present::

    'frame'      : np.ndarray     1-D, height*width, row-major: THE image
    'frame_unit' : str            'digits' | 'Ud' | 'dK'

Processing stages. Which of these a driver produces is fixed by its
constructor options, so it does not change from frame to frame, but a key is
absent when the driver did not compute that stage::

    'pix'        : np.ndarray     uint16 ADC digits as read from the device
    'pix_comp'   : np.ndarray     float, compensated pixel voltages (Ud)
    'pix_dK'     : np.ndarray     float, object temperature in dK

Auxiliary sensor data, always present -- an empty array / nan when the array
type does not have it::

    'eloff', 'ptat', 'vdd', 'atc'  : np.ndarray
    'ptat_avg'                     : float, nan if n/a
    'Tamb'                         : float, dK, nan if n/a

Raw transport payload, only with ``keep_raw=True``::

    'raw'        : dict | bytes

``'frame'`` is the one key a consumer reads. It is the best stage the driver
produced -- ``pix_dK`` if a LuT ran, else ``pix_comp`` if calibration ran,
else ``pix`` -- and ``'frame_unit'`` says which. In-process it is an alias of
that stage's array, not a copy, so it costs nothing until the dict is pickled.
"""

import time
from datetime import datetime
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np


# ── The contract ─────────────────────────────────────────────────────────────

#: Accepted values of ``frame['interface']``.
INTERFACES = ('i2c', 'spi', 'udp', 'usb')

#: Accepted values of ``frame['frame_unit']``.
#:   'digits' - raw ADC digits as read from the device
#:   'Ud'     - compensated pixel voltages in digits
#:   'dK'     - object temperature in deci-Kelvin
FRAME_UNITS = ('digits', 'Ud', 'dK')

#: The pixel stages, best first. ``frame['frame']`` aliases the first one the
#: driver produced, and ``frame['frame_unit']`` is that stage's unit.
PIX_STAGES = (('pix_dK', 'dK'),
              ('pix_comp', 'Ud'),
              ('pix', 'digits'))

#: Auxiliary arrays that are always present (empty when the array type has no
#: such values).
AUX_ARRAYS = ('eloff', 'ptat', 'vdd', 'atc')

#: ``TPArray.DataCols`` group name -> key in this dictionary.
#:
#: ``DataCols`` spells the groups the way the ``.bds`` / ``.txt`` column
#: headers do, which is not the way the drivers' own code spells them. The
#: column order is used elsewhere and is not renamed, so the mapping lives
#: here and both the UDP and the USB driver translate through it.
AUX_FROM_DATACOLS = {'e_off': 'eloff',
                     'vdd':   'vdd',
                     'T_amb': 'Tamb',
                     'PTAT':  'ptat',
                     'ATC':   'atc'}

#: Every key a frame can carry, in the order the docstring lists them. Only
#: used for diagnostics (``describe_frame``) and by the tests.
FRAME_KEYS = ('interface', 'ArrayType', 'shape', 'DevID', 'serial',
              't', 't_mono',
              'frame', 'frame_unit',
              'pix', 'pix_comp', 'pix_dK',
              'eloff', 'ptat', 'vdd', 'atc', 'ptat_avg', 'Tamb',
              'raw')

#: Keys that must be present on every frame from every driver.
REQUIRED_KEYS = ('interface', 'ArrayType', 'shape', 'DevID', 'serial',
                 't', 't_mono', 'frame', 'frame_unit',
                 'eloff', 'ptat', 'vdd', 'atc', 'ptat_avg', 'Tamb')


class FrameError(ValueError):
    """Raised when a driver tried to build a frame that breaks the contract."""


# ── Assembling a frame ───────────────────────────────────────────────────────

def _as_1d(values, name: str, dtype=None) -> np.ndarray:
    """
    Coerce *values* to a flat (1-D) numpy array, row-major.

    Drivers hand in whatever their processing step produced: a 2-D image, a
    column vector out of a LuT lookup, a list. Every array in the frame dict
    is 1-D, so that consumers can reshape with ``frame['shape']`` and never
    have to guess.
    """
    if values is None:
        return None

    array = np.asarray(values)

    if dtype is not None and array.dtype != dtype:
        array = array.astype(dtype)

    if array.ndim != 1:
        # ravel() rather than reshape(-1): a C-contiguous array is a view,
        # anything else is copied, which is what we want either way.
        array = array.ravel()

    return array


def _scalar(value, name: str) -> float:
    """Coerce *value* to a plain float, mapping None onto nan."""
    if value is None:
        return float('nan')

    array = np.asarray(value)

    if array.size == 0:
        return float('nan')
    if array.size != 1:
        raise FrameError(f"frame['{name}'] must be a single value, "
                         f"got {array.size} of them.")

    return float(array.reshape(-1)[0])


def new_frame(interface: str,
              ArrayType: int,
              shape: Tuple[int, int],
              DevID: int = -1,
              serial: str = '',
              t: Optional[datetime] = None,
              t_mono: Optional[float] = None,
              pix=None,
              pix_comp=None,
              pix_dK=None,
              eloff=None,
              ptat=None,
              vdd=None,
              atc=None,
              ptat_avg=None,
              Tamb=None,
              raw=None) -> dict:
    """
    Assemble the frame dictionary documented at the top of this module.

    Called at the end of every driver's ``acquire_frame()``. Everything a
    driver did not compute is left out (the pixel stages) or filled with an
    empty array / nan (the auxiliary data), so a consumer never has to probe
    with ``in frame``.

    Parameters
    ----------
    interface : str
        One of ``INTERFACES``.
    ArrayType : int
        The device's array type.
    shape : (int, int)
        ``(height, width)`` of the pixel image.
    DevID : int
        Device number, or -1 when the interface has none.
    serial : str
        Serial number (USB), MAC-ID (UDP), or '' (I2C, SPI).
    t : datetime, optional
        Wall clock at the end of the acquisition. ``datetime.now()`` when not
        given.
    t_mono : float, optional
        ``time.monotonic()`` at the same instant. Taken now when not given.
        A caller that needs the two stamps to refer to the same moment should
        pass both.
    pix, pix_comp, pix_dK :
        The pixel stages, flattened row-major. Pass only what was computed;
        at least one is required.
    eloff, ptat, vdd, atc :
        Auxiliary sensor arrays. None becomes an empty array.
    ptat_avg, Tamb :
        Scalars. None becomes nan. ``Tamb`` is in dK.
    raw :
        The raw transport payload, or None. Only pass this when the driver
        was constructed with ``keep_raw=True``.

    Returns
    -------
    dict
        The frame. Contains no 'success' / 'image_id' key -- the calling
        thread / process adds those.
    """

    if interface not in INTERFACES:
        raise FrameError(f"interface must be one of {INTERFACES}, "
                         f"got {interface!r}.")

    height, width = (int(shape[0]), int(shape[1]))

    frame = {'interface': str(interface),
             'ArrayType': int(ArrayType),
             'shape': (height, width),
             'DevID': int(DevID),
             'serial': '' if serial is None else str(serial),
             't': datetime.now() if t is None else t,
             't_mono': time.monotonic() if t_mono is None else float(t_mono)}

    # ---- the pixel stages, and the canonical image aliasing the best one ---
    stages = {'pix': _as_1d(pix, 'pix'),
              'pix_comp': _as_1d(pix_comp, 'pix_comp'),
              'pix_dK': _as_1d(pix_dK, 'pix_dK')}

    for key, values in stages.items():
        if values is not None:
            frame[key] = values

    for key, unit in PIX_STAGES:
        if stages[key] is not None:
            frame['frame'] = stages[key]
            frame['frame_unit'] = unit
            break
    else:
        raise FrameError("a frame needs at least one of 'pix', 'pix_comp' or "
                         "'pix_dK'.")

    n_pix = frame['frame'].size
    if n_pix != height * width:
        raise FrameError(f"frame['frame'] has {n_pix} values, but "
                         f"shape {(height, width)} needs {height * width}.")

    # ---- auxiliary data: always present ------------------------------------
    for key, values in (('eloff', eloff), ('ptat', ptat),
                        ('vdd', vdd), ('atc', atc)):
        array = _as_1d(values, key)
        frame[key] = np.empty(0) if array is None else array

    # ptat_avg is derived from 'ptat' when the driver did not compute it
    # itself, so that it is never silently missing for an array that has a
    # PTAT at all.
    if ptat_avg is None and frame['ptat'].size:
        ptat_avg = frame['ptat'].mean()

    frame['ptat_avg'] = _scalar(ptat_avg, 'ptat_avg')
    frame['Tamb'] = _scalar(Tamb, 'Tamb')

    # ---- raw transport payload: opt-in -------------------------------------
    if raw is not None:
        frame['raw'] = raw

    return frame


# ── Checking a frame ─────────────────────────────────────────────────────────

def validate_frame(frame: Mapping) -> None:
    """
    Raise ``FrameError`` unless *frame* honours the contract above.

    Not called in the acquisition path -- ``new_frame()`` builds a valid frame
    by construction. This is for the tests and for a driver that assembles a
    frame some other way.
    """

    if not isinstance(frame, dict):
        raise FrameError(f"a frame must be a dict, got {type(frame)}.")

    missing = [key for key in REQUIRED_KEYS if key not in frame]
    if missing:
        raise FrameError(f"frame is missing {missing}.")

    unknown = [key for key in frame if key not in FRAME_KEYS]
    if unknown:
        raise FrameError(f"frame carries unknown key(s) {unknown}. Add them to "
                         f"hspy.drivers._frame.FRAME_KEYS if they belong in "
                         f"every driver's output.")

    if frame['interface'] not in INTERFACES:
        raise FrameError(f"frame['interface'] is {frame['interface']!r}.")

    if frame['frame_unit'] not in FRAME_UNITS:
        raise FrameError(f"frame['frame_unit'] is {frame['frame_unit']!r}.")

    shape = frame['shape']
    if not (isinstance(shape, tuple) and len(shape) == 2):
        raise FrameError(f"frame['shape'] must be a (height, width) tuple, "
                         f"got {shape!r}.")

    for key in ('frame',) + AUX_ARRAYS:
        array = frame[key]
        if not isinstance(array, np.ndarray):
            raise FrameError(f"frame['{key}'] must be a numpy array, "
                             f"got {type(array)}.")
        if array.ndim != 1:
            raise FrameError(f"frame['{key}'] must be 1-D, "
                             f"has shape {array.shape}.")

    if frame['frame'].size != shape[0] * shape[1]:
        raise FrameError(f"frame['frame'] has {frame['frame'].size} values, "
                         f"shape {shape} needs {shape[0] * shape[1]}.")

    # The canonical image must really be one of the stages, and the unit must
    # be that stage's unit -- a consumer trusts 'frame_unit' blindly.
    for key, unit in PIX_STAGES:
        if key in frame and frame['frame_unit'] == unit:
            if frame[key] is not frame['frame']:
                raise FrameError(f"frame['frame'] must be frame['{key}'] "
                                 f"for frame_unit {unit!r}.")
            break
    else:
        raise FrameError(f"frame_unit {frame['frame_unit']!r} has no matching "
                         f"pixel stage in the frame.")


def describe_frame(frame: Mapping) -> str:
    """One-line summary of a frame, for debugging and error messages."""

    parts = []
    for key in FRAME_KEYS:
        if key not in frame:
            continue
        value = frame[key]
        if isinstance(value, np.ndarray):
            parts.append(f"{key}={value.dtype}{list(value.shape)}")
        else:
            parts.append(f"{key}={value!r}")

    return ', '.join(parts)


__all__ = ['new_frame', 'validate_frame', 'describe_frame', 'FrameError',
           'INTERFACES', 'FRAME_UNITS', 'PIX_STAGES', 'AUX_ARRAYS',
           'AUX_FROM_DATACOLS', 'FRAME_KEYS', 'REQUIRED_KEYS']
