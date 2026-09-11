"""
HTPA160x120dR1L3.95/0.8 Thermopile Array Sensor - Python SPI Interface
Based on Heimann Sensor datasheet Rev12 (09.04.2026), sections 11-16.

Hardware (Figure 3, bottom view):
  Pin 1 (EE_Enable) - Slave select, switches between sensor and internal flash
  Pin 2 (VSS)       - Ground (0 V)
  Pin 3 (VDD)       - 3.3 V - 3.6 V supply
  Pin 4 (SCLK)      - Serial clock
  Pin 5 (MOSI)      - Serial data into sensor
  Pin 6 (MISO)      - Serial data out of sensor

Chip select (Section 12):
  The sensor and the built-in flash (SST26VF016B, 2048 kByte) share the same
  SPI lines and are distinguished solely by the EE_Enable level:

      EE_Enable = HIGH  ->  sensor selected
      EE_Enable = LOW   ->  internal flash selected

  With a plain spidev chip select this maps to SPI_CS_HIGH while talking to
  the sensor and to the normal (active low) polarity while talking to the
  flash.  Because EE_Enable has to be toggled, no other device may share
  these SPI lines.

SPI frame format (Figures 8 and 9):
  Mode 0 (CPOL = 0, CPHA = 0), MSB first, 8-bit command byte.
  Write:  [CMD][DATA]
  Read:   [CMD] followed by as many read bytes as required; the first read
          bit appears on MISO with the falling edge of SCLK after the last
          command bit, so no dummy byte is needed.
  Max. SPI clock 13 MHz (Table 5).
"""

import struct
import time
from datetime import datetime
from typing import Callable, Dict, List, Optional
import numpy as np

from hspy.LuT import LuT

# ---------------------------------------------------------------------------
# Attempt to import spidev; fall back to a stub so the module can be imported
# on non-Linux hosts for testing / offline use.
# ---------------------------------------------------------------------------
try:
    from spidev import SpiDev
    _SPIDEV_AVAILABLE = True
except ImportError:
    _SPIDEV_AVAILABLE = False

    class SpiDev:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "spidev is not installed. Run: pip install spidev"
            )


# -- Sensor register / command bytes (Tables 6-15 of datasheet) --------------
_CMD_CONFIG   = 0x01  # Configuration Register  (write only)
_CMD_STATUS   = 0x02  # Status Register         (read only)
_CMD_TRIM1    = 0x03  # Trim Reg 1: REF_CAL + MBIT_TRIM
_CMD_TRIM2    = 0x04  # Trim Reg 2: BIAS_TRIM_TOP
_CMD_TRIM3    = 0x05  # Trim Reg 3: BIAS_TRIM_BOT
_CMD_TRIM4    = 0x06  # Trim Reg 4: CLK_TRIM
_CMD_READ_TOP = 0x0A  # Read Data 1: top half of array    (1606 bytes)
_CMD_READ_BOT = 0x0B  # Read Data 2: bottom half of array (1606 bytes)

# -- Configuration register bit positions (Table 6) -------------------------
_BIT_WAKEUP   = (1 << 0)  # 1 = on, 0 = sleep
_BIT_BLIND    = (1 << 1)  # 1 = sample electrical offsets
_BIT_RFU_CFG  = (1 << 2)  # reserved for future use
_BIT_START    = (1 << 3)  # trigger conversion
_SHIFT_BLOCK  = 4         # BLOCK occupies bits 7:4

# -- MBIT + REF_CAL bit masks (Table 8) -------------------------------------
_BITMASK_MBIT   = (1 << 4) - 1          # bits 3:0
_BITMASK_REFCAL = ((1 << 2) - 1) << 4   # bits 5:4

# -- CLK_TRIM bit mask (Table 11) -------------------------------------------
_BITMASK_CLKTRIM = (1 << 6) - 1         # bits 5:0

# -- Status register bit positions (Table 7) --------------------------------
_BIT_EOC = (1 << 0)  # End-of-conversion bitwise mask

# -- Internal flash (SST26VF016B) command bytes -----------------------------
_FLASH_CMD_READ = 0x03  # Read, 24-bit address, no dummy byte

# -- Flash memory map (Figure 10 of datasheet) ------------------------------
# NOTE: Section 15.2 quotes ThGrad "from 0x3B20 to 0x6270", which is neither
#       large enough for 19200 int16 values nor consistent with Figure 10.
#       The addresses below are taken from Figure 10, whose region sizes match
#       the element counts exactly (1600 resp. 19200 16-bit values).
_FLA_PIXCMIN            = 0x00000   # float32  minimum sensitivity coefficient
_FLA_PIXCMAX            = 0x00004   # float32  maximum sensitivity coefficient
_FLA_GRADSCALE          = 0x00008   # uint8    thermal gradient scaling exponent
_FLA_TABLENUMBER        = 0x0000B   # uint16 LE look-up table number (TN)
_FLA_EPSILON            = 0x0000D   # uint8    emissivity (100 = 1.00)
_FLA_MBITREFCAL_CALIB   = 0x0001A   # uint8    trim settings used at calibration
_FLA_BIASTOP_CALIB      = 0x0001B   # uint8
_FLA_CLK_CALIB          = 0x0001C   # uint8
_FLA_BIASBOT_CALIB      = 0x0001D   # uint8
_FLA_ARRAYTYPE          = 0x00022   # uint8
_FLA_VDDTH1             = 0x00026   # uint16 LE supply voltage at calib temp 1
_FLA_VDDTH2             = 0x00028   # uint16 LE supply voltage at calib temp 2
_FLA_PTAT_GRAD          = 0x00034   # float32
_FLA_PTAT_OFFSET        = 0x00038   # float32
_FLA_PTAT_TH1           = 0x0003C   # uint16 LE PTAT at calibration temp 1
_FLA_PTAT_TH2           = 0x0003E   # uint16 LE PTAT at calibration temp 2
_FLA_VDDSCGRAD          = 0x0004E   # uint8    VddComp gradient scaling exponent
_FLA_VDDSCOFF           = 0x0004F   # uint8    VddComp offset scaling exponent
_FLA_GLOBAL_OFF         = 0x00054   # int8     global object temperature offset
_FLA_GLOBAL_GAIN        = 0x00055   # uint16 LE
_FLA_MBITREFCAL_USER    = 0x00060   # uint8    user-settable trim copies
_FLA_BIASTOP_USER       = 0x00061   # uint8
_FLA_CLK_USER           = 0x00062   # uint8
_FLA_BIASBOT_USER       = 0x00063   # uint8
_FLA_DEVICEID           = 0x00074   # uint32 LE
_FLA_NR_DEF_PIX         = 0x0007F   # uint8    number of dead pixels (0-96)
_FLA_DEADPIX_ADR        = 0x00080   # 96 x uint16 LE dead pixel addresses
_FLA_DEADPIX_MASK       = 0x00140   # 96 x uint8     neighbour mask per dead pixel
_FLA_VDDCOMPGRAD        = 0x02100   # 1600 x int16 LE
_FLA_VDDCOMPOFF         = 0x02D80   # 1600 x int16 LE
_FLA_THGRAD             = 0x03A00   # 19200 x int16 LE thermal gradient per pixel
_FLA_THOFFSET           = 0x0D000   # 19200 x int16 LE thermal offset per pixel
_FLA_PIJ                = 0x16600   # 19200 x uint16 LE sensitivity coefficient

# -- Array geometry (Section 8) ---------------------------------------------
_ROWS       = 120
_COLS       = 160
_PIXELS     = _ROWS * _COLS         # 19200
_HALF       = int(_PIXELS // 2)     # 9600
_BLOCKS     = 12                    # multiplexed blocks per half
_ROWS_BLOCK = _ROWS // 2 // _BLOCKS # 5 rows per block and half
_BLOCK_PX   = _ROWS_BLOCK * _COLS   # 800 pixels per block and half
_N_ELOFF    = 2 * _BLOCK_PX         # 1600 electrical offsets
_N_VDDCOMP  = 2 * _BLOCK_PX         # 1600 VddComp coefficients
_MAX_DEFPIX = 96                    # Section 16.1

# Number of leading 16-bit words per half-array read: ATC, PTAT, VDD
_HEADER_WORDS = 3
# Bytes returned by a Read Data command (Tables 12-15)
_READ_BYTES   = 2 * (_HEADER_WORDS + _BLOCK_PX)   # 1606

# Read-out order and pixel order are mirrored in the bottom half; a read-out
# index in row R corresponds to the pixel in row (_ROW_MIRROR - R).
_ROW_MIRROR = _ROWS + _ROWS // 2 - 1              # 179

# -- Miscellaneous constants ------------------------------------------------
_PCSCALEVAL      = 1e8      # Section 15.5
_T_INTER_REG_MS  = 5        # min delay between writing trim registers
_T_WAKEUP_US     = 80       # wakeup time after WAKEUP command (Table 5)
_T_STARTUP_US    = 100      # startup time after power-on reset (Table 5)
_T_BUF_NS        = 200      # min time between STOP / START (Table 5)
_F_SPI_TYP       = 10000000  # Hz, typical SPI clock (Table 5)
_F_SPI_MAX       = 13000000  # Hz, max. SPI clock (Table 5)
_F_CLK_MIN       = 0.5      # MHz, min. internal clock frequency (Table 11)
_F_CLK_MAX       = 5.5      # MHz, max. internal clock frequency (Table 11)
_MAX_XFER_BYTES  = 4096     # default spidev buffer size (bufsiz module param)
_N_PTAT          = 2 * _BLOCKS  # 24 PTAT / VDD samples per frame


class SPI_Driver:
    pass


# ---------------------------------------------------------------------------
class SPI_HTPA160x120dR1Error(Exception):
    """Raised on SPI communication or data errors."""


# ---------------------------------------------------------------------------
class SPI_HTPA160x120dR1(SPI_Driver):
    """
    Full SPI driver for the HTPA160x120dR1L3.95/0.8 thermopile array sensor.

    Typical usage::

        from hspy.drivers.spi import SPI_HTPA160x120dR1

        with SPI_HTPA160x120dR1(bus=0, device=0) as sensor:
            frame = sensor.acquire_frame(applyCalib=True, calcdK=False)
            print(frame['Tamb'], frame['pix_comp'].reshape((120, 160)))
    """

    # -- Construction / context manager --------------------------------------

    def __init__(
        self,
        bus: int = 0,
        device: int = 0,
        max_speed_hz: int = _F_SPI_TYP,
        sensor_cs_high: bool = True,
        ee_enable: Optional[Callable[[bool], None]] = None,
        max_xfer_bytes: int = _MAX_XFER_BYTES,
        timeout_factor: float = 4.0,
    ):
        """
        Parameters
        ----------
        bus            : SPI bus number (e.g. 0 -> /dev/spidev0.x)
        device         : SPI chip select number (e.g. 0 -> /dev/spidev0.0)
        max_speed_hz   : SPI clock, 13 MHz max (Table 5)
        sensor_cs_high : If True the spidev chip select is driven HIGH while
                         addressing the sensor and LOW while addressing the
                         internal flash, matching the EE_Enable polarity of
                         Figures 8 and 9. Set to False if EE_Enable is
                         inverted in hardware.
        ee_enable      : Optional callable ``ee_enable(sensor: bool)`` that
                         drives the EE_Enable pin directly (e.g. via GPIO).
                         If given, the spidev chip select polarity is left
                         untouched and this callable is used instead.
        max_xfer_bytes : Largest single SPI transfer; must not exceed the
                         spidev buffer size (``bufsiz``, 4096 by default).
        timeout_factor : End-of-conversion timeout as a multiple of the
                         estimated block conversion time.
        """

        # Call constructor of parent class
        super(SPI_HTPA160x120dR1, self).__init__()

        if max_speed_hz > _F_SPI_MAX:
            raise ValueError(
                f"max_speed_hz {max_speed_hz} exceeds the specified maximum "
                f"SPI clock of {_F_SPI_MAX} Hz."
            )

        self._bus_num       = bus
        self._dev_num       = device
        self._speed         = max_speed_hz
        self._cs_high       = sensor_cs_high
        self._ee_enable     = ee_enable
        self._max_xfer      = max_xfer_bytes
        self._timeout_fac   = timeout_factor
        self._spi: Optional[SpiDev] = None

        # None = unknown, True = sensor selected, False = flash selected
        self._sel_sensor: Optional[bool] = None

        # Look-up-table, set via the LuT property
        self._LuT: Optional[LuT] = None

        # Calibration constants (populated by load_calibration / init)
        self._calib: dict = {}

        # Estimated end-of-conversion timeout in ms, set by init()
        self._timeout_ms: float = 100.0

        # Last converted electrical offsets; reused when acquire_frame() is
        # called with read_eloff=False (Section 15.3 recommends sampling the
        # electrical offsets only every 8th to 10th frame).
        self._eloff_last: Optional[np.ndarray] = None

        # Rolling stack buffers (depth 8) for stable averaging
        self._ptat_stack: List[float] = []
        self._vdd_stack:  List[float] = []
        self._stack_depth = 8

        # Open the SPI bus
        self.open()

        # Initialize the sensor
        self.init()

    def __enter__(self) -> "SPI_HTPA160x120dR1":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # --- class properties ---------------------------------------------------
    @property
    def LuT(self):
        return self._LuT

    @LuT.setter
    def LuT(self, lut: LuT):

        if not isinstance(lut, LuT):
            raise TypeError(f'LuT object is not type {type(LuT)} but {type(lut)}.')
        else:
            self._LuT = lut

    @property
    def calib(self) -> dict:
        """Calibration constants read from the internal flash."""
        return self._calib

    # -- Bus open / close ----------------------------------------------------
    def open(self) -> None:
        """Open the SPI device and apply mode, speed and bit order."""
        if self._spi is not None:
            return

        self._spi = SpiDev()
        self._spi.open(self._bus_num, self._dev_num)

        # Figures 8 and 9: SCLK idles low, MOSI is sampled on the rising edge
        self._spi.mode = 0b00
        self._spi.bits_per_word = 8
        self._spi.lsbfirst = False
        self._spi.max_speed_hz = self._speed

        self._sel_sensor = None
        self._select_sensor()

        # Wait for the sensor to finish its power-on reset (Table 5)
        time.sleep(_T_STARTUP_US * 1e-6)

    def close(self) -> None:
        """Put sensor to sleep, then release the SPI device."""
        if self._spi is not None:
            try:
                self._write_register(_CMD_CONFIG, 0x00)  # sleep
            except Exception:
                pass
            self._spi.close()
            self._spi = None
            self._sel_sensor = None

    # -- High-level public API -----------------------------------------------

    def init(self, use_calib_settings: bool = True) -> None:
        """
        Initialise the sensor (Section 14):
          1. Read all calibration data from the internal flash.
          2. Send WAKEUP.
          3. Write trim registers (using factory or user calibration values).

        Parameters
        ----------
        use_calib_settings : If True (recommended) use the MBIT/BIAS/CLK
                             values stored during factory calibration.
                             If False use the user-defined flash copies.
                             Section 15: the temperature calculation is only
                             valid with the settings used during calibration.
        """
        # Step 1 - calibration data
        self.load_calibration()

        # Step 2 - wake up sensor (Table 5)
        self._write_register(_CMD_CONFIG, _BIT_WAKEUP)
        time.sleep(_T_WAKEUP_US * 1e-6)

        # Step 3 - write trim registers
        key        = "calib" if use_calib_settings else "user"
        mbitrefcal = self._calib[f"mbitrefcal_{key}"]
        bias_top   = self._calib[f"biastop_{key}"]
        bias_bot   = self._calib[f"biasbot_{key}"]
        clk        = self._calib[f"clk_{key}"]

        for cmd, val in [
            (_CMD_TRIM1, mbitrefcal),
            (_CMD_TRIM2, bias_top),   # BIAS_TRIM_TOP
            (_CMD_TRIM3, bias_bot),   # BIAS_TRIM_BOT
            (_CMD_TRIM4, clk),
        ]:
            self._write_register(cmd, val)
            time.sleep(_T_INTER_REG_MS * 1e-3)

        # Step 4 - calculate approximative conversion time
        self._calc_conversion_time(use_calib_settings)

    def _calc_conversion_time(self, use_calib_settings: bool):
        """
        Estimate the conversion time of one block from CLK_TRIM and MBIT_TRIM
        (Table 11):

            F_CLK  = F_min + (F_max - F_min) / 63 * CLK_TRIM     [MHz]
            t_conv = 4 * (2^MBIT + 100) / F_CLK
        """
        key      = "calib" if use_calib_settings else "user"
        MBIT     = self._calib[f"mbit_{key}"]
        CLK_TRIM = self._calib[f"clktrim_{key}"]

        F_CLK = (_F_CLK_MIN + (_F_CLK_MAX - _F_CLK_MIN) / 63 * CLK_TRIM)

        # F_CLK in MHz -> t_block in ms
        t_block = 4 * (2 ** MBIT + 100) / (F_CLK * 1e3)

        self._calib["F_CLK"] = F_CLK
        self._calib["t_block"] = t_block
        self._timeout_ms = t_block * self._timeout_fac

    def sleep(self) -> None:
        """Put sensor into sleep state (~11 uA standby current)."""
        self._write_register(_CMD_CONFIG, 0x00)

    def wakeup(self) -> None:
        """Wake sensor from sleep state."""
        self._write_register(_CMD_CONFIG, _BIT_WAKEUP)
        time.sleep(_T_WAKEUP_US * 1e-6)

    # -- Single-call frame acquisition (for use in a thread's loop body) ------
    def acquire_frame(self,
                      read_eloff: bool = True,   # sample the electrical offsets
                      applyCalib: bool = True,   # apply calibration for Ud compensation
                      calcdK: bool = True) -> dict:      # apply LuT to Ud_comp
        """
        Acquire and fully process one frame: reads the raw pixel data and
        electrical offsets from the sensor over SPI, then converts/rearranges
        the raw bytes and (optionally) applies calibration and the LuT.

        This bundles the whole acquisition into a single blocking call, so it
        is intended to be called once per loop iteration by an externally
        owned thread class - analogous to how the I2C driver's
        ``acquire_frame()`` is used.

        Parameters
        ----------
        read_eloff : if True, trigger a BLIND conversion and read fresh
                     electrical offsets. If False, the offsets of the previous
                     call are reused. Section 15.3 recommends sampling them
                     every 8th to 10th frame only.
        applyCalib : if True, apply flash calibration (apply_calib)
        calcdK     : if True, apply the LuT to the calibrated voltages
                     (apply_LuT). Requires applyCalib=True and self.LuT to be set.

        Returns
        -------
        dict
            Raw + converted (+ calibrated / LuT-mapped) frame data. Does
            NOT contain a 'success' or 'image_id' key - the calling thread
            is responsible for adding those.
        """
        data = self.read_frame()                    # read pixels, atc, ptat, vdd

        if read_eloff:
            data.update(self.read_electrical_offsets())   # read electrical offsets

        data = self.convert_spi_data(data)          # convert and rearrange raw spi data

        if applyCalib:
            # Calculates compensated pixel voltages from raw pixel data given
            # the calibration data from the internal flash
            data = self.apply_calib(data)

        if calcdK:
            data = self.apply_LuT(data)

        return data

    def convert_spi_data(self, data: dict) -> dict:
        """
        Convert and rearrange the raw byte strings returned by read_frame()
        and read_electrical_offsets() into flat, image-ordered numpy arrays.

        Both halves are transmitted MSB first (big endian). The read-out order
        of the bottom half is mirrored with respect to the top half, so that
        the central rows are always read last (Section 8).

        Adds the keys 'pix', 'eloff', 'ptat', 'vdd' and 'atc' to *data*.
        """

        pix_array = np.zeros((_ROWS, _COLS), dtype=np.uint16)   # rearranged pixel data

        atc:  Dict[str, list] = {'top': [], 'bot': []}
        ptat: Dict[str, list] = {'top': [], 'bot': []}
        vdd:  Dict[str, list] = {'top': [], 'bot': []}

        # ------------- Convert pixels and atc/ptat/vdd ------------------------

        for block in range(len(data['pix_raw'])):
            top_raw = data['pix_raw'][block]['top']
            bot_raw = data['pix_raw'][block]['bot']

            top_int = np.frombuffer(top_raw, dtype='>u2')
            bot_int = np.frombuffer(bot_raw, dtype='>u2')
            # ---------------------------------
            data['pix_raw'][block]['top'] = top_int
            data['pix_raw'][block]['bot'] = bot_int
            # ---------------------------------
            # Word[0] = ATC, Word[1] = PTAT, Word[2] = VDD (Tables 12 and 13)
            atc['top'].append(top_int[0])
            atc['bot'].append(bot_int[0])
            ptat['top'].append(top_int[1])
            ptat['bot'].append(bot_int[1])
            vdd['top'].append(top_int[2])
            vdd['bot'].append(bot_int[2])

            pixels_top = top_int[_HEADER_WORDS:]
            pixels_bot = bot_int[_HEADER_WORDS:]

            # Rearrange top and bottom half according to Section 8: the top
            # half is read top-down, the bottom half bottom-up.
            pixels_top = pixels_top.reshape((_ROWS_BLOCK, _COLS))
            pixels_bot = np.flipud(pixels_bot.reshape((_ROWS_BLOCK, _COLS)))

            # Parse top and bottom half pixels to pix_array
            pix_array[block * _ROWS_BLOCK:(block + 1) * _ROWS_BLOCK, :] = pixels_top
            pix_array[_ROWS - (block + 1) * _ROWS_BLOCK:
                      _ROWS - block * _ROWS_BLOCK, :] = pixels_bot

        # ----------- Convert electrical offsets ------------------------------
        if 'eloff_raw' in data:
            top_raw = data['eloff_raw']['top']
            bot_raw = data['eloff_raw']['bot']

            top_int = np.frombuffer(top_raw, dtype='>u2')   # 803 words
            bot_int = np.frombuffer(bot_raw, dtype='>u2')   # 803 words
            # ---------------------------------
            data['eloff_raw']['top'] = top_int
            data['eloff_raw']['bot'] = bot_int
            # ---------------------------------
            # The BLIND read returns its own ATC / PTAT / VDD header. It is
            # kept separate because Section 15.1 averages the 24 PTAT samples
            # of the active conversion only.
            data['atc_blind']  = np.array([top_int[0], bot_int[0]], dtype=np.uint16)
            data['ptat_blind'] = np.array([top_int[1], bot_int[1]], dtype=np.uint16)
            data['vdd_blind']  = np.array([top_int[2], bot_int[2]], dtype=np.uint16)

            eloff_top = top_int[_HEADER_WORDS:]
            eloff_bot = bot_int[_HEADER_WORDS:]

            topbot = list(np.hstack([eloff_top, eloff_bot]))

            self._eloff_last = self._sort_perBlock_calibData(topbot)

        if self._eloff_last is None:
            raise SPI_HTPA160x120dR1Error(
                "No electrical offsets available. Call read_electrical_offsets() "
                "or acquire_frame(read_eloff=True) at least once."
            )

        # ------------- Append converted data to dict and return --------------
        data['pix']   = pix_array.flatten()
        data['eloff'] = self._eloff_last.flatten()
        data['ptat']  = np.array(ptat['top'] + ptat['bot'], dtype=np.uint16)
        data['vdd']   = np.array(vdd['top'] + vdd['bot'], dtype=np.uint16)
        data['atc']   = np.array(atc['top'] + atc['bot'], dtype=np.uint16)

        return data

    # -- Single Frame readout ------------------------------------------------

    def read_frame(self) -> dict:
        """
        Acquire one complete frame (12 blocks, top and bottom half).

        The sensor is divided into top and bottom halves, each split into
        12 blocks of 800 pixels. Block x of the top and the bottom half are
        measured at the same time and read separately (Section 8).

        Every Read Data command returns ATC, PTAT and VDD of the respective
        half in front of the pixel data (Tables 12 and 13), so no separate
        VDD measurement is required.

        Returns
        -------
        dict with keys:
          "pix_raw" - {block: {"top": bytes(1606), "bot": bytes(1606)}}
          "t"       - datetime.now() at function return
        """
        t_block = self._calib["t_block"]        # ms, block conversion time

        # --- Read in raw bytes of top and lower half from all 12 blocks ---
        raw_bytes: dict = {}

        for block in range(_BLOCKS):

            raw_bytes[block] = {}

            # Write to configuration register:
            # ---------------------------------------------------
            # | 7 | 6 | 5 | 4 |   3   |  2  |   1   |   0    |
            # |     BLOCK     | START | RFU | BLIND | WAKEUP |
            # ---------------------------------------------------
            config = _BIT_WAKEUP | _BIT_START | (block << _SHIFT_BLOCK)

            # Write to config register
            self._write_register(_CMD_CONFIG, config)

            # Pause 90 % of the approximate conversion time before checking
            # if end of conversion is reached
            time.sleep(0.9 * t_block * 1e-3)

            # Start checking for end of conversion bit
            self._wait_eoc()

            # Read block from register
            top_raw, bot_raw = self._read_both_halves()

            raw_bytes[block]['top'] = top_raw
            raw_bytes[block]['bot'] = bot_raw

        # ------ Return -------------------------------------------------------
        return {'pix_raw': raw_bytes, 't': datetime.now()}

    # -- Electrical Offset Readout -------------------------------------------
    def read_electrical_offsets(self) -> dict:
        """
        Trigger a BLIND conversion and read all 1600 electrical offset values
        (Tables 14 and 15).

        With the BLIND bit set the electrical offsets are sampled instead of
        the active pixels and the BLOCK setting is ignored, so a single pair
        of reads returns all offsets.

        Returns
        -------
        dict with key "eloff_raw" -> {"top": bytes(1606), "bot": bytes(1606)}
        """
        t_block = self._calib["t_block"]        # ms, block conversion time

        # Write to config register
        self._write_register(_CMD_CONFIG, _BIT_WAKEUP | _BIT_START | _BIT_BLIND)

        # Pause 90 % of the approximate conversion time before checking
        # if end of conversion is reached
        time.sleep(0.9 * t_block * 1e-3)

        # Start checking for end of conversion bit
        self._wait_eoc()

        # Read offsets from register
        top_raw, bot_raw = self._read_both_halves()

        return {'eloff_raw': {'top': top_raw, 'bot': bot_raw}}

    # -- Temperature calculation ---------------------------------------------
    def calculate_Tamb(self, data: dict) -> dict:
        """
        Calculate the ambient temperature Ta in dK from the averaged PTAT
        value (Section 15.1)::

            Ta = PTAT_av * PTAT_gradient + PTAT_offset

        Adds the keys 'ptat_avg' and 'Tamb' to *data*.
        """
        if not self._calib:
            raise SPI_HTPA160x120dR1Error(
                "Calibration not loaded. Call init() or load_calibration() first."
            )

        c = self._calib

        PTAT_avg = data['ptat'].astype(np.float64).mean()
        Tamb = PTAT_avg * c["ptat_gradient"] + c["ptat_offset"]

        data['ptat_avg'] = PTAT_avg
        data['Tamb'] = Tamb

        return data

    def apply_calib(self, data: dict) -> dict:
        """
        Convert a raw frame (from read_frame() / convert_spi_data()) into
        per-pixel compensated voltages in digits. Also computes the ambient
        temperature as a side-effect.

        Processing pipeline (Sections 15.1 - 15.5 and 16.1):
          1. Ambient temperature from PTAT
          2. Thermal offset compensation per pixel
          3. Electrical offset compensation per pixel
          4. VDD (supply voltage) compensation per pixel
          5. Sensitivity (PixC) compensation
          6. Dead-pixel masking

        Adds the key 'pix_comp' (and several intermediates) to *data*.
        """
        if not self._calib:
            raise SPI_HTPA160x120dR1Error(
                "Calibration not loaded. Call init() or load_calibration() first."
            )

        c = self._calib             # Calibration data dictionary

        # ------------- 15.1 Ambient Temperature ------------------------------
        data = self.calculate_Tamb(data)

        PTAT_avg = data['ptat_avg']

        # ------------- 15.2 Thermal Offset -----------------------------------
        Th_grad   = c["thgrad_arr"]
        Th_off    = c["thoffset_arr"]
        gradScale = c["gradscale"]

        V_ThComp = (Th_grad * PTAT_avg) / 2 ** (gradScale) + Th_off

        data['V_ThComp'] = V_ThComp         # store for debugging

        V_comp = data['pix'].astype(np.float64) - V_ThComp   # apply compensation

        # ------------- 15.3 Electrical Offset --------------------------------
        V_ElComp = data['eloff'].astype(np.float64)

        data['V_ElComp'] = V_ElComp         # store for debugging

        V_comp = V_comp - V_ElComp          # apply compensation

        # ------------- 15.4 Vdd Compensation ---------------------------------
        VddCompGrad = c["vddcompgrad_arr"]
        VddCompOff  = c["vddcompoff_arr"]
        VddScGrad   = c["vddscgrad"]
        VddScOff    = c["vddscoff"]

        VddTh1      = c["vddth1"]
        VddTh2      = c["vddth2"]
        PTAT_Th1    = c["ptat_th1"]
        PTAT_Th2    = c["ptat_th2"]

        VDD_avg = data['vdd'].astype(np.float64).mean()

        data['vdd_avg'] = VDD_avg

        V_VddComp = ((VddCompGrad * PTAT_avg) / (2 ** VddScGrad) + VddCompOff) / \
            (2 ** VddScOff) * \
            (VDD_avg - VddTh1 - ((VddTh2 - VddTh1) / (PTAT_Th2 - PTAT_Th1)) *
             (PTAT_avg - PTAT_Th1))

        data['V_VddComp'] = V_VddComp       # store for debugging

        V_comp = V_comp - V_VddComp         # apply compensation

        # ------------- 15.5 Object Temperature -------------------------------
        Pij         = c["pij_arr"]
        PixC_max    = c["pixcmax"]
        PixC_min    = c["pixcmin"]
        eps         = c["epsilon"]
        GlobalGain  = c["global_gain"]

        PixCij = (Pij * (PixC_max - PixC_min) / 65535 + PixC_min) * \
            eps / 100 * GlobalGain / 10000

        data['scale'] = _PCSCALEVAL / PixCij   # store for debugging

        pix_comp = V_comp * _PCSCALEVAL / PixCij

        # ------------- 16.1 Pixel Masking ------------------------------------
        pix_comp = self.mask_dead_pixels(pix_comp)

        # ------------- Write final value to dict and return ------------------
        data['pix_comp'] = pix_comp

        return data

    def apply_LuT(self, data: dict) -> dict:
        """
        Applies the look-up table to the compensated pixel voltages stored in
        data['pix_comp'] and writes the object temperatures in dK to
        data['pix_dK'] (Section 15.5).

        Parameters
        ----------
        data : dict
            Frame dictionary as returned by apply_calib().

        Returns
        -------
        dict
            The same dictionary with the key 'pix_dK' added.
        """

        # Arrange the compensated pixel voltages and the ambient temperature
        # in a np.ndarray
        Ud_Ta = np.hstack([data['pix_comp'].reshape((-1, 1)),
                           np.tile(data['Tamb'], (_PIXELS, 1)).reshape((-1, 1))])

        pix_dK = self.LuT.calc_To(Ud_Ta)

        # Section 16.1 masks the temperature values of the defect pixels
        pix_dK = self.mask_dead_pixels(np.asarray(pix_dK).flatten())

        data['pix_dK'] = pix_dK

        return data

    def get_ambient_temperature(self, frame: dict) -> float:
        """
        Return the ambient (sensor body) temperature in degrees Celsius
        computed from the on-chip PTAT sensor.
        """
        c = self._calib
        ta_dk = frame["ptat"].astype(np.float64).mean() * c["ptat_gradient"] \
            + c["ptat_offset"]
        return dk_to_celsius(ta_dk)

    def mask_dead_pixels(self, values: np.ndarray) -> np.ndarray:
        """
        Overwrite every defect pixel with the average of the neighbours
        selected by its DeadPixMask (Section 16.1).

        Parameters
        ----------
        values : flat array of _PIXELS entries in image order

        Returns
        -------
        np.ndarray
            The same array with the defect pixels replaced.
        """
        for dead_pix, mask in self._calib.get('dead_pix_sorted', {}).items():
            if len(mask) == 0:
                continue
            values[dead_pix] = values[mask].mean()

        return values

    # -- Calibration / internal flash ----------------------------------------

    def load_calibration(self) -> None:
        """
        Read all calibration constants from the internal flash (Figure 10)
        and cache them in self._calib. Called automatically by init().
        """
        c: dict = {}

        def u8(addr: int) -> int:
            return self._flash_read(addr, 1)[0]

        def i8(addr: int) -> int:
            v = u8(addr)
            return v if v < 128 else v - 256

        def u16(addr: int) -> int:
            return struct.unpack_from("<H", self._flash_read(addr, 2))[0]

        def u32(addr: int) -> int:
            return struct.unpack_from("<I", self._flash_read(addr, 4))[0]

        def f32(addr: int) -> float:
            return struct.unpack_from("<f", self._flash_read(addr, 4))[0]

        def i16_arr(addr: int, n: int) -> List[int]:
            raw = self._flash_read(addr, n * 2)
            return list(np.frombuffer(raw, dtype='<i2'))

        def u16_arr(addr: int, n: int) -> List[int]:
            raw = self._flash_read(addr, n * 2)
            return list(np.frombuffer(raw, dtype='<u2'))

        def unpack_mbit(mbit_raw: int) -> int:
            return mbit_raw & _BITMASK_MBIT

        def unpack_refcal(mbit_raw: int) -> int:
            return (mbit_raw & _BITMASK_REFCAL) >> 4

        def unpack_clktrim(clk_raw: int) -> int:
            return clk_raw & _BITMASK_CLKTRIM

        # Scalar values
        c["pixcmin"]       = f32(_FLA_PIXCMIN)
        c["pixcmax"]       = f32(_FLA_PIXCMAX)
        c["gradscale"]     = u8 (_FLA_GRADSCALE)
        c["table_number"]  = u16(_FLA_TABLENUMBER)
        c["epsilon"]       = u8 (_FLA_EPSILON)
        c["arraytype"]     = u8 (_FLA_ARRAYTYPE)
        c["device_id"]     = u32(_FLA_DEVICEID)

        # Factory calibration trim settings
        c["mbitrefcal_calib"]   = u8 (_FLA_MBITREFCAL_CALIB)
        c["mbit_calib"]         = unpack_mbit  (c["mbitrefcal_calib"])
        c["refcal_calib"]       = unpack_refcal(c["mbitrefcal_calib"])
        c["biastop_calib"]      = u8 (_FLA_BIASTOP_CALIB)
        c["biasbot_calib"]      = u8 (_FLA_BIASBOT_CALIB)
        c["clk_calib"]          = u8 (_FLA_CLK_CALIB)
        c["clktrim_calib"]      = unpack_clktrim(c["clk_calib"])

        # User trim settings
        c["mbitrefcal_user"]    = u8 (_FLA_MBITREFCAL_USER)
        c["mbit_user"]          = unpack_mbit  (c["mbitrefcal_user"])
        c["refcal_user"]        = unpack_refcal(c["mbitrefcal_user"])
        c["biastop_user"]       = u8 (_FLA_BIASTOP_USER)
        c["biasbot_user"]       = u8 (_FLA_BIASBOT_USER)
        c["clk_user"]           = u8 (_FLA_CLK_USER)
        c["clktrim_user"]       = unpack_clktrim(c["clk_user"])

        # PTAT / VDD calibration values
        c["ptat_gradient"] = f32(_FLA_PTAT_GRAD)
        c["ptat_offset"]   = f32(_FLA_PTAT_OFFSET)
        c["ptat_th1"]      = u16(_FLA_PTAT_TH1)
        c["ptat_th2"]      = u16(_FLA_PTAT_TH2)
        c["vddth1"]        = u16(_FLA_VDDTH1)
        c["vddth2"]        = u16(_FLA_VDDTH2)
        c["vddscgrad"]     = u8 (_FLA_VDDSCGRAD)
        c["vddscoff"]      = u8 (_FLA_VDDSCOFF)

        # Gain / offset
        c["global_off"]    = i8 (_FLA_GLOBAL_OFF)
        c["global_gain"]   = u16(_FLA_GLOBAL_GAIN)

        # Dead-pixels
        c["nr_def_pix"]    = u8 (_FLA_NR_DEF_PIX)
        c["dead_pix_adr"]  = u16_arr(_FLA_DEADPIX_ADR, _MAX_DEFPIX)
        c["dead_pix_mask"] = list(self._flash_read(_FLA_DEADPIX_MASK, _MAX_DEFPIX))

        c['dead_pix_sorted'] = self._determine_dead_pix_neighbors(c["nr_def_pix"],
                                                                  c["dead_pix_adr"],
                                                                  c["dead_pix_mask"])

        # Per-pixel arrays (19200 entries each)
        c["thgrad"]        = i16_arr(_FLA_THGRAD,   _PIXELS)
        c["thoffset"]      = i16_arr(_FLA_THOFFSET, _PIXELS)
        c["pij"]           = u16_arr(_FLA_PIJ,      _PIXELS)

        # Rearrange per-pixel arrays to correspond to actual pixel order
        c["thgrad_arr"]    = self._sort_perPixel_calibData(c["thgrad"])
        c["thoffset_arr"]  = self._sort_perPixel_calibData(c["thoffset"])
        c["pij_arr"]       = self._sort_perPixel_calibData(c["pij"])

        # VDD compensation arrays (1600 entries each)
        c["vddcompgrad"]   = i16_arr(_FLA_VDDCOMPGRAD, _N_VDDCOMP)
        c["vddcompoff"]    = i16_arr(_FLA_VDDCOMPOFF,  _N_VDDCOMP)

        # Reshape VDD calibration data to array of same dimension as pixels
        c["vddcompgrad_arr"] = self._sort_perBlock_calibData(c["vddcompgrad"])
        c["vddcompoff_arr"]  = self._sort_perBlock_calibData(c["vddcompoff"])

        self._calib = c

    def _sort_perBlock_calibData(self, block_data: list) -> np.ndarray:
        """
        Expand a block-wise data set (1600 entries: 800 top + 800 bottom) to a
        full, image-ordered array of _PIXELS entries.

        This applies to the electrical offsets in read-out order (Tables 14
        and 15 / Section 15.3) as well as to VddCompGrad and VddCompOff as
        stored in the flash (Figure 12): in both cases the first 800 entries
        describe the 5 rows of a top-half block in top-down order and the last
        800 entries the 5 rows of a bottom-half block in bottom-up order.
        """

        if not isinstance(block_data, list):
            raise TypeError(f'block_data is type {type(block_data)} instead of list.')

        # Check input size
        expected_len = _N_ELOFF

        if not len(block_data) == expected_len:
            raise ValueError(f'Length of block_data is {len(block_data)}, '
                             f'expected was {expected_len}.')

        top = block_data[0:_BLOCK_PX]
        bot = block_data[_BLOCK_PX::]

        top = np.array(top).reshape((_ROWS_BLOCK, _COLS))
        bot = np.array(bot).reshape((_ROWS_BLOCK, _COLS))

        # Flip the bottom readout
        bot = np.flipud(bot)

        # Repeat blocks using tile
        top = np.tile(top, (_BLOCKS, 1))
        bot = np.tile(bot, (_BLOCKS, 1))

        # Concatenate and return
        return np.vstack([top, bot]).flatten()

    def _sort_perPixel_calibData(self, calib: list) -> np.ndarray:
        """
        Rearrange a per-pixel data set stored in read-out order (ThGrad,
        ThOffset, Pij - Figure 11) into image order.

        The first half maps 1:1 onto the top half of the image; the second
        half is mirrored, i.e. read-out row 60 belongs to image row 119.
        """

        top = np.array(calib[0:_HALF]).reshape((int(_ROWS / 2), _COLS))
        bot = np.flipud(np.array(calib[_HALF::]).reshape((int(_ROWS / 2), _COLS)))

        # Concatenate and return
        return np.vstack([top, bot]).flatten()

    def _determine_dead_pix_neighbors(self,
                                      nr_def_pix: int,
                                      dead_pix_adr: List[np.uint16],
                                      dead_pix_mask: List[np.uint16]) -> dict:
        """
        Determines the neighbors of each defect pixel to be used for masking
        via averaging (Section 16.1).

        Parameters
        ----------
        nr_def_pix : number of valid entries in dead_pix_adr / dead_pix_mask
        dead_pix_adr : dead pixel addresses in read-out order
        dead_pix_mask : neighbour mask per dead pixel

        Returns
        -------
        dict
            {pixel index in image order: [indices of neighbours to average]}
        """

        if nr_def_pix > _MAX_DEFPIX:
            raise SPI_HTPA160x120dR1Error(
                f"Flash reports {nr_def_pix} defect pixels, but at most "
                f"{_MAX_DEFPIX} are allowed. Is the flash map correct?"
            )

        # Truncate the list with addresses of dead pixels and corresponding
        # masks to nr_def_pix elements
        dead_pix_adr = dead_pix_adr[0:nr_def_pix]
        dead_pix_mask = dead_pix_mask[0:nr_def_pix]

        # ------- Convert read-out-order indices to pixel array indices -------
        DefPix_idx = self._ReadOutIdx_to_PixIdx(dead_pix_adr)

        # -- Get a dictionary that maps each pixel to a dict of its neighbors--
        NeighMap = self._get_neighbors()

        # -------------- Keep only mapping for defect pixels ------------------
        NeighMap = {pix: NeighMap[pix] for pix in DefPix_idx}

        # -------------- Construct the final DeadPixelMask --------------------
        DeadPixelMask = {}

        # Compare the mapping to the mask given for each defect pixel
        for i in range(nr_def_pix):

            pix_idx = DefPix_idx[i]                 # index of dead pixel
            pix_mask = dead_pix_mask[i]             # pixel mask from flash
            pix_NeighMap = NeighMap[pix_idx]        # dict of all neighbours

            # Decode the mask for pixel pix_idx. pix_mask_decoded contains
            # positions of bits in pix_mask that are set to 1,
            # e.g. 129 -> 1000 0001 -> [0,7]
            pix_mask_decoded = self._decode_DeadPixMask(pix_mask,          # mask from flash
                                                        pix_idx < _HALF)  # pixel in upper half

            # Initialize list containing all neighbouring pixels that
            # are supposed to be used for masking
            DeadPixelMask[pix_idx] = []

            # Populate list
            for pos in pix_mask_decoded:
                if pos in pix_NeighMap:
                    DeadPixelMask[pix_idx].append(pix_NeighMap[pos])

        return DeadPixelMask

    def _get_neighbors(self) -> Dict[int, Dict[int, int]]:
        """
        Returns a dict mapping each pixel index to its dict of neighbors.
        Neighbors are the 8-directional adjacent elements with the following
        numbering

        7 -----     0     ----- 1

        6 ----- DeadPixel ----- 2

        5 -----     4     ----- 3

        """

        neighbors = {}

        for i in range(_PIXELS):
            row, col = divmod(i, _COLS)
            adjacent = {}
            if row > 0:                              adjacent[0] = i - _COLS      # N
            if row > 0         and col < _COLS - 1:  adjacent[1] = i - _COLS + 1  # NE
            if col < _COLS - 1:                      adjacent[2] = i + 1          # E
            if row < _ROWS - 1 and col < _COLS - 1:  adjacent[3] = i + _COLS + 1  # SE
            if row < _ROWS - 1:                      adjacent[4] = i + _COLS      # S
            if row < _ROWS - 1 and col > 0:          adjacent[5] = i + _COLS - 1  # SW
            if col > 0:                              adjacent[6] = i - 1          # W
            if row > 0         and col > 0:          adjacent[7] = i - _COLS - 1  # NW

            neighbors[i] = adjacent

        return neighbors

    def _ReadOutIdx_to_PixIdx(self, ReadOutIdx: List[np.uint16]) -> List[int]:
        """
        Convert pixel addresses given in read-out order into image order
        (Section 16.1)::

            adaptedAdr = 19200 + 9600 - DeadPixAdr + k * 2 - 160

        which is equivalent to mirroring the row index of the bottom half,
        i.e. read-out row R corresponds to image row (179 - R). The top half
        is not affected.

        Parameters
        ----------
        ReadOutIdx : addresses in read-out order

        Returns
        -------
        List[int]
            addresses in image order
        """

        PixIdx = []

        # Loop over all pixel addresses (given in read-out-order)
        for ro_idx in ReadOutIdx:

            ro_idx = int(ro_idx)

            # Read-out-order index and pixel index are identical in upper half
            if ro_idx < _HALF:
                pix_idx = ro_idx
            # In the lower half the read-out order counts from the centre
            # outwards while the pixel index counts from top to bottom
            else:
                row, col = divmod(ro_idx, _COLS)
                pix_idx = (_ROW_MIRROR - row) * _COLS + col

            PixIdx.append(int(pix_idx))

        return PixIdx

    def _decode_DeadPixMask(self,
                            mask: int,
                            upper_half: bool) -> List[int]:
        """
        Decodes an 8-bit mask integer into a list of neighbor indices to use,
        normalized to upper_half=True positions (Section 16.1).

        upper_half=True  position layout:   upper_half=False position layout:
          7  0  1                             5  4  3
          6  X  2                             6  X  2
          5  4  3                             7  0  1
        """
        active = [i for i in range(8) if (mask >> i) & 1]

        if not upper_half:
            remap = {0: 4, 1: 3, 2: 2, 3: 1, 4: 0, 5: 7, 6: 6, 7: 5}
            active = [remap[i] for i in active]

        return active

    def load_lut(self, lut: LuT) -> None:
        """
        Load the sensor-specific look-up table for object-temperature
        calculation (Section 16.2).

        Heimann Sensor provides this table in a separate file called "Table.c";
        the matching table is identified by the table number stored in the
        flash (``self.calib['table_number']``). Parse it into a
        :class:`hspy.LuT.LuT` and pass it here before calling apply_LuT().
        """
        self.LuT = lut

    # -- Trim register helpers -----------------------------------------------

    def set_trim_registers(
        self,
        mbit: int,
        bias_top: int,
        bias_bot: int,
        clk: int,
    ) -> None:
        """
        Write all four trim registers with explicit values (Tables 8 - 11).

        Parameters
        ----------
        mbit     : Trim Register 1 - REF_CAL (bits 5:4) | MBIT_TRIM (bits 3:0)
                   MBIT_TRIM m=4..12 -> ADC resolution (m+4) bits
        bias_top : BIAS_TRIM_TOP 0-255 -> 1-13 uA (ADC bias current, top half)
        bias_bot : BIAS_TRIM_BOT 0-255 -> 1-13 uA (ADC bias current, bottom half)
        clk      : CLK_TRIM 0-63 -> 0.5-5.5 MHz (use clk_trim_to_freq() to check)

        Note (Section 15): the temperature calculation is only valid if the
        same settings are used that have been set during calibration.
        """
        pairs = [
            (_CMD_TRIM1, mbit),
            (_CMD_TRIM2, bias_top),
            (_CMD_TRIM3, bias_bot),
            (_CMD_TRIM4, clk),
        ]
        for cmd, val in pairs:
            self._write_register(cmd, val)
            time.sleep(_T_INTER_REG_MS * 1e-3)

    @staticmethod
    def clk_trim_to_freq(clk_trim: int) -> float:
        """
        Convert a CLK_TRIM register value (0-63) to clock frequency in MHz
        (Table 11)::

            F_CLK = (F_min + (F_max - F_min) / 63 * CLK_TRIM) MHz
        """
        return _F_CLK_MIN + (_F_CLK_MAX - _F_CLK_MIN) / 63.0 * clk_trim

    @staticmethod
    def block_time_ms(clk_trim: int, mbit: int) -> float:
        """
        Estimated measurement time for one block (a twelfth of the array) in
        milliseconds. From Table 11::

            t_conv = 4 * (2^MBIT + 100) / F_CLK

        mbit here is the MBIT_TRIM nibble (0..12), NOT the full register byte.
        """
        f_clk_mhz = SPI_HTPA160x120dR1.clk_trim_to_freq(clk_trim)
        return 4.0 * (2 ** mbit + 100) / (f_clk_mhz * 1e3)

    # -- Status register -----------------------------------------------------

    def read_status(self) -> int:
        """Return the raw 8-bit status register value (Table 7)."""
        return self._read_command(_CMD_STATUS, 1)[0]

    # -- Private: low-level SPI ----------------------------------------------

    def _select_sensor(self) -> None:
        """Drive EE_Enable so that the sensor is addressed (Figures 8 and 9)."""
        if self._sel_sensor is True:
            return
        if self._ee_enable is not None:
            self._ee_enable(True)
        else:
            self._spi.cs_high = self._cs_high
        self._sel_sensor = True

    def _select_flash(self) -> None:
        """Drive EE_Enable so that the internal flash is addressed."""
        if self._sel_sensor is False:
            return
        if self._ee_enable is not None:
            self._ee_enable(False)
        else:
            self._spi.cs_high = not self._cs_high
        self._sel_sensor = False

    def _write_register(self, cmd: int, value: int) -> None:
        """
        SPI write (Figure 8):  EE_Enable | CMD | VALUE | EE_Enable
        """
        self._select_sensor()
        self._spi.xfer2([cmd & 0xFF, value & 0xFF])

    def _read_command(self, cmd: int, length: int) -> bytes:
        """
        SPI read (Figure 9): send the 8-bit command, then clock out *length*
        bytes while EE_Enable stays asserted.
        """
        self._select_sensor()

        if length + 1 > self._max_xfer:
            raise SPI_HTPA160x120dR1Error(
                f"Requested transfer of {length + 1} bytes exceeds the SPI "
                f"buffer size of {self._max_xfer} bytes. Increase the spidev "
                f"'bufsiz' module parameter or lower max_xfer_bytes."
            )

        rx = self._spi.xfer2([cmd & 0xFF] + [0x00] * length)

        return bytes(rx[1:])

    def _read_both_halves(self):
        """Read the top and the bottom half of the current block."""
        top_raw = self._read_command(_CMD_READ_TOP, _READ_BYTES)
        bot_raw = self._read_command(_CMD_READ_BOT, _READ_BYTES)
        return top_raw, bot_raw

    def _wait_eoc(self) -> None:
        """Poll the status register until EOC is set or the timeout expires."""
        deadline = time.monotonic() + self._timeout_ms / 1e3
        while True:
            if self.read_status() & _BIT_EOC:
                return
            if time.monotonic() > deadline:
                raise SPI_HTPA160x120dR1Error(
                    f"Timeout ({self._timeout_ms:.1f} ms) waiting for EOC."
                )

    # -- Private: internal flash access (SST26VF016B) ------------------------

    def _flash_read(self, mem_addr: int, length: int) -> bytes:
        """
        Read *length* bytes from the internal flash starting at the 24-bit
        memory address *mem_addr*.

        The flash is selected by driving EE_Enable low (Section 12); the
        SST26VF016B Read protocol is:

            CMD 0x03 | AddrH | AddrM | AddrL | D[0] ... D[n]

        Reads are split into chunks so that a single SPI transfer stays within
        the spidev buffer size.
        """
        self._select_flash()

        try:
            result = bytearray()
            max_chunk = self._max_xfer - 4      # 1 command + 3 address bytes

            if max_chunk <= 0:
                raise SPI_HTPA160x120dR1Error(
                    f"max_xfer_bytes ({self._max_xfer}) is too small for a "
                    f"flash read."
                )

            while length > 0:
                chunk = min(length, max_chunk)

                tx = [_FLASH_CMD_READ,
                      (mem_addr >> 16) & 0xFF,
                      (mem_addr >> 8) & 0xFF,
                      mem_addr & 0xFF] + [0x00] * chunk

                rx = self._spi.xfer2(tx)

                result += bytes(rx[4:])
                mem_addr += chunk
                length -= chunk
        finally:
            # Always hand the bus back to the sensor
            self._select_sensor()

        return bytes(result)

    # -- Private: stack buffer helpers ---------------------------------------

    def _update_stack(self, stack: list, value) -> None:
        """Append a value to a rolling FIFO of depth self._stack_depth."""
        stack.append(value)
        if len(stack) > self._stack_depth:
            stack.pop(0)


# -- Convenience utilities ---------------------------------------------------

def dk_to_celsius(dk: float) -> float:
    """Convert deci-Kelvin to degrees Celsius."""
    return (dk - 2732.0) / 10.0


def celsius_to_dk(celsius: float) -> float:
    """Convert degrees Celsius to deci-Kelvin."""
    return celsius * 10.0 + 2732.0


def frame_to_2d(flat) -> np.ndarray:
    """Reshape a flat 19200-element array into a 120x160 row-major grid."""
    return np.asarray(flat).reshape((_ROWS, _COLS))


def print_frame(temps, fmt: str = "{:6.1f}") -> None:
    """Pretty-print a temperature frame to stdout."""
    grid = frame_to_2d(temps)
    for row in grid:
        print(" ".join(fmt.format(v) for v in row))
