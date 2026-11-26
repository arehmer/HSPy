# Unified driver frame API

Status: **implemented.** Section 8 lists what landed where and what is not covered.

## 1. Where we were before this change

| | `I2C_HTPA32x32d` | `SPI_HTPA160x120dR1` | `HTPA_UDPReader` | `USB_HTPA` |
|---|---|---|---|---|
| module | `hspy/drivers/i2c/drivers.py` | `hspy/drivers/spi/drivers.py` | `hspy/readers.py` | `hspy/drivers/usb/usb.py` |
| frame method | `acquire_frame(active_vdd, blind_vdd, applyCalib, calcdK)` | `acquire_frame(read_eloff, applyCalib, calcdK)` | `read_continuous_bytestream(dev_id, first_frame_timeout)` | `acquire_frame(timeout_ms)` |
| one object = | one device | one device | **N devices** (DataFrame + socket dict) | one device |
| timestamp | `t` (`datetime`) | `t` (`datetime`) | none | `t` (`datetime`) |
| image keys | `pix` (digits), `pix_comp`, `pix_dK` | `pix`, `pix_comp`, `pix_dK` | `pix` (unit = stream cmd) | `pix`, `pix_dK` (only for `b'K'`) |
| aux keys | `eloff, ptat, vdd, ptat_avg, Tamb` | `eloff, ptat, vdd, atc, ptat_avg, vdd_avg, Tamb, atc_blind, ptat_blind, vdd_blind` | `e_off, vdd, T_amb, PTAT, ATC` (from `TPArray.DataCols`) | `eloff, vdd, Tamb, ptat, atc, ptat_avg, aux, frame_raw` |
| raw payload | `pix_raw` (per-block dict), `eloff_raw` | `pix_raw`, `eloff_raw` | none | `frame_raw` |
| device id | none | none | `dev_id` arg only | `serial_number`, `ArrayType` |

Consequences visible in the consumers:

* `Imshow._target()` / `ImshowProcess._target()` carried a fallback chain
  `data['frame']` else `data['pix_dK']` else `data['pix']`, because no key was guaranteed.
* `UDP_Client` / `UDP_ClientProcess` set `result['frame'] = udp_data['pix']` and then **threw
  the rest of the UDP frame away** — `e_off`, `PTAT`, `T_amb`, `ATC` never reached the queue.
* `I2C_Client` / `I2C_ClientProcess` did `result.update(data)`, so the full raw per-block byte
  dict went into the queue and the pickled item was far larger than it needed to be
  (`SharedMemoryQueue` slots are fixed size, 4 MB in the examples).
* `data['t']` existed for i2c/spi/usb but never for udp, so `DataRecord` and the PTAT plot
  silently fell back to `datetime.now()` / `time.time()`.
* There was **no SPI client at all**. `examples/live/multiprocessed/spi-imshow-mp.py` was a copy
  of the i2c script and actually drove `I2C_ClientProcess` + `I2C_HTPA32x32d`.
* `I2C_Client.stop()` called `self.udp_reader.stop_continuous_bytestream(...)` — a copy-paste
  bug; that attribute does not exist, so `stop()` raised.
* `FileWriter_Thread._organize_data()` assumed `frame_dict['frame']` was the whole serial data
  vector (it sliced `frame[len(tparray._pix):]`), which no client had produced since `frame`
  became pixels-only.

## 2. The frame dictionary

One flat dict — flat, not nested, so pickling stays cheap and consumers never walk a tree.
Every value is a `numpy` array, a Python scalar, a `str`, a `tuple` or a `datetime`: all
trivially picklable, nothing holding an OS handle.

```python
{
  # -- identity: always present -----------------------------------------
  'interface'  : str,            # 'i2c' | 'spi' | 'udp' | 'usb'
  'ArrayType'  : int,
  'shape'      : (height, width),
  'DevID'      : int,            # -1 when the interface has no device number
  'serial'     : str,            # USB serial / UDP MAC-ID / '' for i2c and spi

  # -- time: always present ---------------------------------------------
  't'          : datetime,       # wall clock, end of acquisition
  't_mono'     : float,          # time.monotonic() at the same instant

  # -- the image: always present ----------------------------------------
  'frame'      : np.ndarray,     # 1-D, height*width, row-major: THE image to use
  'frame_unit' : str,            # 'digits' | 'Ud' | 'dK'

  # -- processing stages: key absent when the driver did not compute it --
  'pix'        : np.ndarray,     # uint16 ADC digits as read from the device
  'pix_comp'   : np.ndarray,     # float, compensated pixel voltages (Ud)
  'pix_dK'     : np.ndarray,     # float, object temperature in dK

  # -- auxiliary sensor data: always present, empty array if n/a --------
  'eloff'      : np.ndarray,
  'ptat'       : np.ndarray,
  'vdd'        : np.ndarray,
  'atc'        : np.ndarray,
  'ptat_avg'   : float,          # nan if n/a
  'Tamb'       : float,          # dK, nan if n/a

  # -- raw transport payload: only when keep_raw=True -------------------
  'raw'        : dict | bytes,   # key absent when keep_raw=False (the default)
}
```

`'frame'` is the one key every consumer reads: the best stage the driver produced
(`pix_dK` if a LuT ran, else `pix_comp` if calibration ran, else `pix`), with `'frame_unit'`
saying what it is. In-process it is an alias of the stage array, not a copy, so it costs
nothing; the duplication only shows up in the pickled queue item.

Aux key spelling follows the lowercase convention the USB driver already established:
`_FRAME_PART_KEYS` in `hspy/drivers/usb/usb.py:78` maps `TPArray.DataCols`' group names
`e_off/T_amb/PTAT/ATC` onto `eloff/Tamb/ptat/atc`. `TPArray.DataCols` is **not** renamed — it
defines `.bds`/`.txt` column order and is used elsewhere; the UDP driver reuses that mapping.

Keys are also *stable*: a key that a given driver can ever produce is present on every frame
from that driver, so a consumer never has to probe with `in data`. The only conditional keys
are the stage keys (`pix_comp`, `pix_dK`) and `raw`, both switched by constructor options that
do not change during a run.

### Picklability (hard requirement)

Everything above pickles with the default protocol today — the numpy arrays, the `datetime`,
the plain scalars, the `(h, w)` tuple. No sockets, no `SMBus`/`spidev` handles, no `LuT`
objects, no pandas objects ever enter the dict. `SharedMemoryQueue` pickles each item into a
fixed-size slot, so **size** is the real constraint, which is why `keep_raw=False` is the
default: the per-block `pix_raw`/`eloff_raw` dicts roughly double the payload.

### What the client threads / processes add

Status keys stay with the client — the driver knows nothing about loop state:

```python
'success'     : bool
'image_id'    : int
'device_lost' : True        # only on the final item
```

`DevID`, `serial` and `t` move **into** the driver dict; the client no longer sets `DevID`, it
reads it back from the frame, since a driver always knows which device it talks to.

## 3. Common method and signature

```python
frame = driver.acquire_frame()      # no arguments, every driver
```

All per-acquisition options move into `__init__` and become writable attributes, so the
hot-loop call is argument-free and literally identical for every interface:

| option | drivers | note |
|---|---|---|
| `applyCalib` | i2c, spi | |
| `calcdK` | i2c, spi | not a constructor argument: `init()` calls `load_lut()`, which loads the LuT matching the table number in the EEPROM/flash from `hspy/luts` and sets `calcdK` to whether one was found (and `applyCalib` is on). Assigning `driver.LuT` sets it too. Read-only property on udp/usb: `stream_cmd == b'K'`. |
| `active_vdd`, `blind_vdd` | i2c | |
| `eloff_every` | spi | **new**: replaces per-call `read_eloff`. `1` = every frame, `8` = every 8th (what §15.3 recommends). The driver keeps the counter and reuses the last offsets in between, so the client no longer has to toggle anything. |
| `timeout_ms` | usb | |
| `stream_cmd` | usb, udp | already a client argument today; moves to the driver |
| `keep_raw` | all | default `False` |

They stay plain attributes, so `driver.calcdK = False` between frames still works.

## 4. UDP: new single-device `UDP_HTPA`

New package:

```
hspy/drivers/udp/__init__.py      # UDP_Driver, UDP_HTPA, HTPA_UDPReader,
                                  # HTPA_ByteStream_Converter, HTPA_UDPTimeout
hspy/drivers/udp/drivers.py       # the code
```

`HTPA_UDPReader`, `HTPA_ByteStream_Converter` and `HTPA_UDPTimeout` move out of
`hspy/readers.py` unchanged in behaviour. `hspy/readers.py` keeps the file readers
(`HTPAdGUI_FileReader`) and re-exports the moved names with a `DeprecationWarning`, so
existing scripts keep importing.

On top of that, a new **single-device** driver, symmetric with `USB_HTPA`:

```python
class UDP_HTPA(UDP_Driver):
    def __init__(self, IP='', DevID=-1, Bcast_Addr='', ArrayType=None,
                 port=30444, stream_cmd=b'K', timeout_s=1.0,
                 first_frame_timeout=5.0, keep_raw=False): ...

    # lifecycle, same names as USB_HTPA
    def open(self): ...            # discover (if DevID+Bcast_Addr), bind, one socket
    def init(self): ...            # ArrayType from the bind call string, build TPArray
    def start_stream(self): ...
    def stop_stream(self): ...
    def close(self): ...           # stop stream + release device + close socket
    def acquire_frame(self): ...   # one frame, the dict of §2

    # introspection, same names as USB_HTPA
    is_open / is_connected / is_streaming / width / height / n_pixels / ArrayType / serial
```

Internally it reuses `HTPA_UDPReader`'s working pieces (`_bind_tparray`, `_read_port`, the
package-assembly loop, `HTPA_ByteStream_Converter`) rather than reimplementing them — the
packet-level code is correct and well commented, it is only the N-device bookkeeping that goes
away. Two side benefits fall out of this:

* **`ArrayType` is auto-detected.** The bind call string already carries `Arraytype`, `MAC-ID`,
  `IP` and `DevID` (`_callstring_to_information`), so `UDP_HTPA` can set up its own `TPArray`.
  Today the examples have to set `udp_reader.ArrayType` by hand after binding.
* **`t`, `DevID` and `serial` (the MAC-ID) get into the frame dict**, which the current
  bytestream converter cannot provide because it only sees packets.

`UDP_Client` / `UDP_ClientProcess` then lose all their binding, discovery and release logic
(roughly 80 lines each) and keep only the watchdog, which is the part worth keeping.

## 5. Consumer changes

`hspy/ipc/threads.py`
* `USB_Client`, `UDP_Client`, `I2C_Client`: call `driver.acquire_frame()`, `result.update(data)`,
  add `success` / `image_id` only. No more `result['frame'] = data['pix']` and no more
  per-iteration option forwarding.
* new `SPI_Client` (modelled on `USB_Client`: `_setup` / `_target` / `_teardown`, device-lost
  item, print-each-error-once).
* `Imshow._target()`: fallback chain collapses to `frame = data['frame']`.
* **bug** `I2C_Client.stop()`: drop the `self.udp_reader` lines (just `self._exit.set()`, as
  the base class loop + `_teardown` handle the rest).
* **bug** `FileWriter_Thread._organize_data()`: rebuild the serial-order row from `data['pix']`
  plus the aux keys instead of slicing `data['frame']`.

An item with `success=False` carries `success`, `image_id`, `DevID` and `serial` and nothing
else — so a consumer can still tell which device fell silent, which the old bare
`{'success': False}` could not.

`hspy/ipc/processes.py`
* the same for `USB_ClientProcess`, `UDP_ClientProcess`, `I2C_ClientProcess`; new
  `SPI_ClientProcess` (built from `spi_driver_cls` + `spi_driver_kwargs`, per the module's
  design note).
* `ImshowProcess._target()`: fallback chain collapses to `data['frame']`.
* `I2C_ClientProcess` / `SPI_ClientProcess` now ignore Ctrl+C in the child and close the driver
  in `_teardown()` — the process built it, so the process releases it. The USB and UDP versions
  already did both.

`examples/live/*`
* `spi-imshow-mp.py` rewritten to actually use SPI (it used to drive i2c), plus a threaded
  `spi-imshow.py`.
* `udp-imshow.py` / `udp-imshow-mp.py` switched to `UDP_HTPA`; the manual `ArrayType`
  assignment goes away.
* `i2c-imshow*.py` / `usb-imshow*.py`: option kwargs move from the client constructor into the
  driver kwargs where they are passed at all.

## 6. Decisions taken

| # | decision |
|---|---|
| 1 | Canonical image key is **`'frame'`** (+ `'frame_unit'`), with `pix` / `pix_comp` / `pix_dK` kept as the individual stages. |
| 2 | UDP gets a **new single-device `UDP_HTPA`**; `HTPA_UDPReader` moves along for discovery and multi-device use. |
| 3 | Scope is **all four drivers** (usb included) **plus the missing `SPI_Client` / `SPI_ClientProcess`**. |
| 4 | **One client class per interface** stays — no generic `Frame_Client`. |
| 5 | **Clean break** on key names; examples and docstrings updated in the same change. |
| 6 | Per-acquisition options live **in the constructor** as writable attributes; `acquire_frame()` takes no arguments. SPI's `read_eloff` becomes a driver-owned `eloff_every` counter. |
| 7 | Raw bytes are **off by default**, opt-in via `keep_raw=True`, exposed as a single `'raw'` key. |
| 8 | **All three consumer bugs** above get fixed as part of this change. |

## 7. Migration notes

For code outside this repository:

| before | now |
|---|---|
| `from hspy.readers import HTPA_UDPReader` | `from hspy.drivers.udp import HTPA_UDPReader` (the old import warns but works) |
| `udp_reader.read_continuous_bytestream(dev_id)` | `udp_driver.acquire_frame()` on a `UDP_HTPA`; the reader still has the old method |
| `driver.acquire_frame(applyCalib=False)` | `driver.applyCalib = False`, or pass it to the constructor |
| `SPI...acquire_frame(read_eloff=…)` per frame | `SPI_HTPA160x120dR1(eloff_every=8)`; the driver counts |
| `UDP_Client(udp_reader=…, IP=…, DevID=…, Bcast_Addr=…)` | `UDP_Client(udp_driver=UDP_HTPA(IP=…))` |
| `UDP_ClientProcess(udp_reader_cls=…, udp_reader_kwargs=…, IP=…)` | `UDP_ClientProcess(udp_driver_cls=UDP_HTPA, udp_driver_kwargs={'IP': …})` |
| `USB_Client(…, DevID=0)` | `USB_HTPA(…, DevID=0)` — the label lives on the driver |
| `data['pix_dK']` / `data['pix']` to get the image | `data['frame']`, with `data['frame_unit']` |
| `udp_data['e_off'] / ['T_amb'] / ['PTAT'] / ['ATC']` | `data['eloff'] / ['Tamb'] / ['ptat'] / ['atc']` |
| `data['serial_number']` (USB frames) | `data['serial']` — `serial_number` is still the driver's constructor argument |
| `data['aux']`, `data['frame_raw']` (USB frames) | `data['raw']` with `keep_raw=True` |
| `udp_reader.ArrayType = …` before streaming | nothing: `UDP_HTPA` reads it off the bind answer |

## 8. What landed

New:
* [hspy/drivers/_frame.py](../hspy/drivers/_frame.py) — the contract: `new_frame()`,
  `validate_frame()`, `describe_frame()`, and the `DataCols` → aux-key mapping that the UDP and
  USB drivers share.
* [hspy/drivers/udp/](../hspy/drivers/udp) — `UDP_Driver`, `UDP_HTPA`, `UDP_HTPAError`, plus
  `HTPA_UDPReader`, `HTPA_ByteStream_Converter` and `HTPA_UDPTimeout` moved out of
  `hspy/readers.py` (which re-exports them with a `DeprecationWarning`).
* `SPI_Client` in [threads.py](../hspy/ipc/threads.py) and `SPI_ClientProcess` in
  [processes.py](../hspy/ipc/processes.py) — there was no SPI client at all before.
* [examples/live/threaded/spi-imshow.py](../examples/live/threaded/spi-imshow.py), and
  `spi-imshow-mp.py` rewritten to actually drive SPI.
* [tests/test_frame_contract.py](../tests/test_frame_contract.py) — 14 checks: one signature
  per driver, the contract itself, the pickle and `SharedMemoryQueue` round trip, and the UDP
  driver against a stand-in for its reader. Needs no sensor.

Bugs fixed on the way:
* `I2C_Client.stop()` called `self.udp_reader.stop_continuous_bytestream(...)` on an attribute
  that does not exist, so stopping the thread raised.
* `FileWriter_Thread._organize_data()` sliced `data['frame']` as if it were the whole serial
  data vector. It now assembles the row from `data['pix']` and the auxiliary keys, in
  `TPArray.DataCols` order, padding a group the frame is short on.
* The UDP clients dropped every non-pixel part of each frame.
* `i2c-imshow.py` / `i2c-imshow-mp.py` read `args.target_ip` and `args.target_port`, which the
  parser never defined, so both scripts died at startup. `i2c-imshow.py` also imported `hsod`,
  which this repository does not depend on.

Deliberately left alone:
* `TPArray.DataCols` keeps its `e_off` / `T_amb` / `PTAT` / `ATC` group names — they define the
  `.bds` / `.txt` column headers. The translation lives in `_frame.AUX_FROM_DATACOLS`.
* `processes.py` defines `UDP_PickleServerProcess` twice (identical bodies, the second wins).
  Pre-existing, harmless, and nothing to do with the frame API.
* The hardware paths themselves. `convert_i2c_data`, `convert_spi_data`, `apply_calib` and
  `apply_LuT` still work on the driver's internal dict exactly as before; only
  `acquire_frame()` translates that into the frame dict at the end. Nothing here has been run
  against a real sensor.
