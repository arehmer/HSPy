"""
HTPA32x32dR2L1.7/0.8 Thermopile Array Sensor - Python I2C Interface
Based on Heimann Sensor datasheet Rev9 (02.03.2026)

Hardware:
  Pin 1 (SDA) - Serial Data,  100k internal pull-up
  Pin 2 (VSS) - Ground (0 V)
  Pin 3 (VDD) - 3.3 V - 3.6 V supply
  Pin 4 (SCL) - Serial Clock, 100k internal pull-up

I2C addresses:
  0x1A  - sensor configuration / data
  0x50  - internal EEPROM (24AA64)

Recommended external circuit (Section 6):
  4.7 kOhm pull-ups on SDA and SCL
  100 nF + 47 uF decoupling on VDD
  No inductors in the supply path
"""

import struct
import time
from typing import List, Optional
import smbus2
# ---------------------------------------------------------------------------
# Attempt to import smbus2; fall back to a stub so the module can be imported
# on non-Linux hosts for testing / offline use.
# ---------------------------------------------------------------------------
try:
    from smbus2 import SMBus, i2c_msg
    _SMBUS_AVAILABLE = True
except ImportError:
    _SMBUS_AVAILABLE = False

    class SMBus:  # type: ignore
        def __init__(self, bus: int):
            raise RuntimeError(
                "smbus2 is not installed. Run: pip install smbus2"
            )

    class i2c_msg:  # type: ignore
        @staticmethod
        def write(addr, data): ...
        @staticmethod
        def read(addr, length): ...


# ── I2C addresses ────────────────────────────────────────────────────────────
_ADDR_SENSOR = 0x1A   # 7-bit sensor address
_ADDR_EEPROM = 0x50   # 7-bit EEPROM address (24AA64, A2..A0 = 000)

# ── Sensor register / command bytes (Tables 6-16 of datasheet) ───────────────
_CMD_CONFIG   = 0x01  # Configuration Register  (write only)
_CMD_STATUS   = 0x02  # Status Register         (read only)
_CMD_TRIM1    = 0x03  # Trim Reg 1: REF_CAL + MBIT_TRIM
_CMD_TRIM2    = 0x04  # Trim Reg 2: BIAS_TRIM_TOP
_CMD_TRIM3    = 0x05  # Trim Reg 3: BIAS_TRIM_BOT
_CMD_TRIM4    = 0x06  # Trim Reg 4: CLK_TRIM
_CMD_TRIM5    = 0x07  # Trim Reg 5: BPA_TRIM_TOP
_CMD_TRIM6    = 0x08  # Trim Reg 6: BPA_TRIM_BOT
_CMD_TRIM7    = 0x09  # Trim Reg 7: PU_SDA/SCL_TRIM
_CMD_READ_TOP = 0x0A  # Read top-half array data  (258 bytes)
_CMD_READ_BOT = 0x0B  # Read bottom-half array data (258 bytes)

# ── Configuration register bit positions (Table 6) ────────────────────────────
_BIT_WAKEUP   = (1 << 0)  # 1 = on, 0 = sleep
_BIT_BLIND    = (1 << 1)  # 1 = sample electrical offsets
_BIT_VDD_MEAS = (1 << 2)  # 1 = measure VDD instead of PTAT
_BIT_START    = (1 << 3)  # trigger conversion

# ── Status register bit positions (Table 7) ───────────────────────────────────
_BIT_EOC = (1 << 0)  # End-of-conversion

# ── EEPROM memory map (Figure 12 of datasheet) ────────────────────────────────
_EEP_PIXCMIN      = 0x0000   # float32  minimum sensitivity coefficient
_EEP_PIXCMAX      = 0x0004   # float32  maximum sensitivity coefficient
_EEP_GRADSCALE    = 0x0008   # uint8    thermal gradient scaling exponent
_EEP_EPSILON      = 0x000D   # uint8    emissivity (100 = 1.00)
_EEP_MBIT_CALIB   = 0x001A   # uint8    MBIT used during factory calibration
_EEP_BIAS_CALIB   = 0x001B   # uint8
_EEP_CLK_CALIB    = 0x001C   # uint8
_EEP_BPA_CALIB    = 0x001D   # uint8
_EEP_PU_CALIB     = 0x001E   # uint8
_EEP_VDDTH1       = 0x0026   # uint16 LE  supply voltage at calibration temp 1
_EEP_VDDTH2       = 0x0028   # uint16 LE  supply voltage at calibration temp 2
_EEP_PTAT_GRAD    = 0x0034   # float32
_EEP_PTAT_OFFSET  = 0x0038   # float32
_EEP_PTAT_TH1     = 0x003C   # uint16 LE  PTAT at calibration temp 1
_EEP_PTAT_TH2     = 0x003E   # uint16 LE  PTAT at calibration temp 2
_EEP_VDDSCGRAD    = 0x004E   # uint8    VddComp gradient scaling exponent
_EEP_VDDSCOFF     = 0x004F   # uint8    VddComp offset scaling exponent
_EEP_GLOBAL_OFF   = 0x0054   # int8     global object temperature offset
_EEP_GLOBAL_GAIN  = 0x0055   # uint16 LE
_EEP_MBIT_USER    = 0x0060   # uint8    user-settable trim copies
_EEP_BIAS_USER    = 0x0061
_EEP_CLK_USER     = 0x0062
_EEP_BPA_USER     = 0x0063
_EEP_PU_USER      = 0x0064
_EEP_NR_DEF_PIX   = 0x007F   # uint8    number of dead pixels (0-5)
_EEP_DEADPIX_ADR  = 0x0080   # 5 x uint16 LE  dead pixel addresses
_EEP_DEADPIX_MASK = 0x008A   # 5 x uint8      neighbour mask per dead pixel
_EEP_VDDCOMPGRAD  = 0x0340   # 256 x int16 LE
_EEP_VDDCOMPOFF   = 0x0540   # 256 x int16 LE
_EEP_THGRAD       = 0x0740   # 1024 x int16 LE  thermal gradient per pixel
_EEP_THOFFSET     = 0x0F40   # 1024 x int16 LE  thermal offset per pixel
_EEP_PIJ          = 0x1740   # 1024 x uint16 LE sensitivity coefficient

# ── Array geometry ────────────────────────────────────────────────────────────
_ROWS      = 32
_COLS      = 32
_PIXELS    = _ROWS * _COLS     # 1024
_HALF      = _PIXELS // 2      # 512
_BLOCK_PX  = 128               # pixels per block per half

# ── Miscellaneous constants ───────────────────────────────────────────────────
_PCSCALEVAL      = 1e8    # Section 12.5
_T_INTER_REG_MS  = 5      # min delay between writing trim registers
_T_WAKEUP_US     = 80     # wakeup time after WAKEUP command (Table 5)
_T_POLL_MS       = 1      # EOC polling interval


# ─────────────────────────────────────────────────────────────────────────────
class HTPA32x32dError(Exception):
    """Raised on I2C communication or data errors."""


# ─────────────────────────────────────────────────────────────────────────────
class HTPA32x32d:
    """
    Full I2C driver for the HTPA32x32dR2L1.7/0.8 thermopile array sensor.

    Typical usage::

        from htpa32x32d import HTPA32x32d

        with HTPA32x32d(bus=1) as sensor:
            sensor.init()
            frame = sensor.read_frame()
            temps = sensor.calculate_temperatures(frame)
            ambient = sensor.get_ambient_temperature(frame)
            print(f"Ambient: {ambient:.1f} C  Max object: {max(temps):.1f} C")
    """

    # ── Construction / context manager ───────────────────────────────────────

    def __init__(
        self,
        bus: int = 1,
        i2c_addr: int = _ADDR_SENSOR,
        eeprom_addr: int = _ADDR_EEPROM,
        timeout_ms: int = 500,
    ):
        """
        Parameters
        ----------
        bus         : Linux I2C bus number (e.g. 1 -> /dev/i2c-1)
        i2c_addr    : 7-bit sensor address (default 0x1A)
        eeprom_addr : 7-bit EEPROM address (default 0x50)
        timeout_ms  : Maximum wait time for end-of-conversion
        """
        self._bus_num    = bus
        self._addr       = i2c_addr
        self._eep_addr   = eeprom_addr
        self._timeout_ms = timeout_ms
        self._bus: Optional[SMBus] = None

        # Calibration constants (populated by load_calibration / init)
        self._calib: dict = {}

        # Rolling stack buffers (depth 8) for stable averaging
        self._ptat_stack: List[float] = []
        self._vdd_stack:  List[float] = []
        self._el_offset_stack: List[List[int]] = []
        self._stack_depth = 8

    def __enter__(self) -> "HTPA32x32d":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ── Bus open / close ─────────────────────────────────────────────────────

    def open(self) -> None:
        """Open the I2C bus file descriptor."""
        self._bus = SMBus(self._bus_num)

    def close(self) -> None:
        """Put sensor to sleep, then release the bus."""
        if self._bus is not None:
            try:
                self._write_register(_CMD_CONFIG, 0x00)  # sleep
            except Exception:
                pass
            self._bus.close()
            self._bus = None

    # ── High-level public API ─────────────────────────────────────────────────

    def init(self, use_calib_settings: bool = True) -> None:
        """
        Initialise the sensor:
          1. Read all calibration data from EEPROM.
          2. Send WAKEUP.
          3. Write trim registers (using factory or user calibration values).

        Parameters
        ----------
        use_calib_settings : If True (recommended) use the MBIT/BIAS/CLK/BPA/PU
                             values stored during factory calibration.
                             If False use the user-defined EEPROM copies.
        """
        # Step 1 – calibration data
        self.load_calibration()

        # Step 2 – wake up sensor (Section 11.5)
        self._write_register(_CMD_CONFIG, _BIT_WAKEUP)
        time.sleep(_T_WAKEUP_US * 1e-6)

        # Step 3 – write trim registers
        key  = "calib" if use_calib_settings else "user"
        mbit = self._calib[f"mbit_{key}"]
        bias = self._calib[f"bias_{key}"]
        clk  = self._calib[f"clk_{key}"]
        bpa  = self._calib[f"bpa_{key}"]
        pu   = self._calib[f"pu_{key}"]

        for cmd, val in [
            (_CMD_TRIM1, mbit),
            (_CMD_TRIM2, bias),   # BIAS_TRIM_TOP
            (_CMD_TRIM3, bias),   # BIAS_TRIM_BOT  (same value)
            (_CMD_TRIM4, clk),
            (_CMD_TRIM5, bpa),    # BPA_TRIM_TOP
            (_CMD_TRIM6, bpa),    # BPA_TRIM_BOT  (same value)
            (_CMD_TRIM7, pu),
        ]:
            self._write_register(cmd, val)
            time.sleep(_T_INTER_REG_MS * 1e-3)

    def sleep(self) -> None:
        """Put sensor into sleep state (~9 µA standby current)."""
        self._write_register(_CMD_CONFIG, 0x00)

    def wakeup(self) -> None:
        """Wake sensor from sleep state."""
        self._write_register(_CMD_CONFIG, _BIT_WAKEUP)
        time.sleep(_T_WAKEUP_US * 1e-6)

    # ── Frame readout ─────────────────────────────────────────────────────────

    def read_frame(self, measure_vdd: bool = False) -> dict:
        """
        Acquire one complete frame (4 blocks + electrical offsets).

        The sensor is divided into top and bottom halves, each split into
        4 blocks of 128 pixels.  Each block is triggered and read separately.
        Electrical offsets are sampled every call (oversample every 8-10 frames
        in production for better performance).

        Parameters
        ----------
        measure_vdd : If True, simultaneously read the supply voltage; the
                      result is stored in the VDD stack for Vdd compensation.

        Returns
        -------
        dict with keys:
          "pixels"     - list[1024] raw ADC digits (uint16)
          "el_offsets" - list[256]  electrical offset digits (uint16)
          "ptat_av"    - PTAT average from the stack (float)
          "vdd_av"     - VDD average from the stack  (float or None)
        """
        raw_pixels: List[int] = [0] * _PIXELS
        ptat_samples: List[float] = []

        # ── Read four blocks ──────────────────────────────────────────────
        for block in range(4):
            config = _BIT_WAKEUP | _BIT_START | (block << 4)
            if measure_vdd:
                config |= _BIT_VDD_MEAS

            self._write_register(_CMD_CONFIG, config)
            self._wait_eoc()

            top = self._read_half(_CMD_READ_TOP)  # 129 words
            bot = self._read_half(_CMD_READ_BOT)  # 129 words

            # Word[0] = PTAT (or VDD if VDD_MEAS set)
            ptat_samples.append(float(top[0]))
            ptat_samples.append(float(bot[0]))

            if measure_vdd:
                vdd_sample = (float(top[0]) + float(bot[0])) / 2.0
                self._update_stack(self._vdd_stack, vdd_sample)

            # ── Store top-half pixels (Figure 5) ─────────────────────────
            # Top block pixels are contiguous:
            #   word[1] = pixel (0  + block*128)
            #   word[2] = pixel (1  + block*128)  ...
            #   word[128] = pixel (127 + block*128)
            for idx in range(_BLOCK_PX):
                raw_pixels[idx + block * _BLOCK_PX] = top[idx + 1]

            # ── Store bottom-half pixels (Table 16, mirrored readout) ─────
            # The bottom half is read in reverse row order so that the central
            # rows of the full array are always read last.
            # The 128 words [1..128] map to four groups of 32 pixels:
            #   words [1 ..32]  -> pixels (992 - block*128) .. (1023 - block*128)
            #   words [33..64]  -> pixels (960 - block*128) .. (991  - block*128)
            #   words [65..96]  -> pixels (928 - block*128) .. (959  - block*128)
            #   words [97..128] -> pixels (896 - block*128) .. (927  - block*128)
            #  ...and so on for next block
            for sub in range(4):
                base_pix = 992 - block * _BLOCK_PX - sub * _COLS
                for col_idx in range(_COLS):
                    word_pos = sub * _COLS + col_idx + 1
                    raw_pixels[base_pix + col_idx] = bot[word_pos]

        # ── Read electrical offsets (Section 12.3) ────────────────────────
        el_offsets = self._read_electrical_offsets()

        # ── Update PTAT stack ─────────────────────────────────────────────
        # Average of all 8 PTAT readings across the 4 blocks (2 per block)
        ptat_av_this_frame = sum(ptat_samples[:8]) / min(8, len(ptat_samples))
        self._update_stack(self._ptat_stack, ptat_av_this_frame)
        self._update_stack(self._el_offset_stack, el_offsets)

        ptat_av = sum(self._ptat_stack) / len(self._ptat_stack)
        vdd_av  = (sum(self._vdd_stack) / len(self._vdd_stack)
                   if self._vdd_stack else None)

        return {
            "pixels":     raw_pixels,
            "el_offsets": el_offsets,
            "ptat_av":    ptat_av,
            "vdd_av":     vdd_av,
        }

    def read_vdd(self) -> float:
        """
        Trigger a dedicated VDD measurement for supply-voltage compensation.
        Returns the averaged VDD digit value and updates the internal stack.
        """
        config = _BIT_WAKEUP | _BIT_START | _BIT_VDD_MEAS
        self._write_register(_CMD_CONFIG, config)
        self._wait_eoc()
        top = self._read_half(_CMD_READ_TOP)
        bot = self._read_half(_CMD_READ_BOT)
        vdd_av = (float(top[0]) + float(bot[0])) / 2.0
        self._update_stack(self._vdd_stack, vdd_av)
        return vdd_av

    # ── Temperature calculation ───────────────────────────────────────────────

    def calculate_temperatures(self, frame: dict) -> List[float]:
        """
        Convert a raw frame (from read_frame()) into per-pixel object
        temperatures in degrees Celsius.

        Processing pipeline (Sections 12.1 - 12.5):
          1. Ambient temperature from PTAT
          2. Thermal offset compensation per pixel
          3. Electrical offset compensation per pixel
          4. VDD (supply voltage) compensation per pixel  [if VDD available]
          5. Sensitivity (PixC) compensation
          6. Object temperature via LUT bilinear interpolation
          7. Dead-pixel masking

        Returns list[1024] of temperatures in °C.
        """
        if not self._calib:
            raise HTPA32x32dError(
                "Calibration not loaded. Call init() or load_calibration() first."
            )

        c       = self._calib
        raw     = frame["pixels"]
        ptat_av = frame["ptat_av"]
        vdd_av  = frame.get("vdd_av") or (
            float(sum(self._vdd_stack) / len(self._vdd_stack))
            if self._vdd_stack else None
        )

        # Use stored electrical offset stack if frame value unavailable
        if frame["el_offsets"]:
            el_off = frame["el_offsets"]
        elif self._el_offset_stack:
            el_off = self._el_offset_stack[-1]
        else:
            el_off = [0] * 256

        # ── Pre-compute per-pixel sensitivity coefficients ────────────────
        pix_c = self._calc_pixc(c)

        results: List[float] = []

        for pix_num in range(_PIXELS):
            row    = pix_num // _COLS
            col    = pix_num  % _COLS
            is_top = (row < _ROWS // 2)

            v = float(raw[pix_num])

            # ── Step 1: Thermal offset compensation (Section 12.2) ────────
            th_grad = c["thgrad"][pix_num]
            th_off  = c["thoffset"][pix_num]
            v_comp  = v - (th_grad * ptat_av / (2 ** c["gradscale"])) - th_off

            # ── Step 2: Electrical offset compensation (Section 12.3) ─────
            # Top half index: (col + row*32) % 128
            # Bottom half index: (col + row*32) % 128 + 128
            if is_top:
                el_idx = (col + row * _COLS) % _BLOCK_PX
            else:
                el_idx = (col + row * _COLS) % _BLOCK_PX + _BLOCK_PX
            v_comp -= el_off[el_idx]

            # ── Step 3: VDD compensation (Section 12.4) ───────────────────
            if vdd_av is not None:
                if is_top:
                    idx256 = (col + row * _COLS) % _BLOCK_PX
                else:
                    idx256 = (col + row * _COLS) % _BLOCK_PX + _BLOCK_PX

                vdd_cg = c["vddcompgrad"][idx256]
                vdd_co = c["vddcompoff"][idx256]

                vdd_num   = (vdd_cg * ptat_av / (2 ** c["vddscgrad"])) + vdd_co
                vdd_denom = 2 ** c["vddscoff"]

                ptat_th1 = c["ptat_th1"]
                ptat_th2 = c["ptat_th2"]
                vddth1   = c["vddth1"]
                vddth2   = c["vddth2"]

                ptat_slope  = (vddth2 - vddth1) / (ptat_th2 - ptat_th1)
                vdd_factor  = vdd_av - vddth1 - ptat_slope * (ptat_av - ptat_th1)
                v_comp     -= (vdd_num / vdd_denom) * vdd_factor

            # ── Step 4: Sensitivity (PixC) compensation (Section 12.5) ───
            pc = pix_c[pix_num] if pix_c[pix_num] != 0 else _PCSCALEVAL
            v_pixc = v_comp * _PCSCALEVAL / pc

            # ── Step 5: Object temperature via LUT ────────────────────────
            ta_dk   = ptat_av * c["ptat_gradient"] + c["ptat_offset"]
            t_dk    = self._lut_interpolate(v_pixc, ta_dk, c)
            t_dk   += c["global_off"]                      # GlobalOff trim
            results.append((t_dk - 2732.0) / 10.0)        # dK -> °C

        # ── Step 6: Replace dead pixels with neighbour average ────────────
        self._apply_pixel_masking(results, c)

        return results

    def get_ambient_temperature(self, frame: dict) -> float:
        """
        Return the ambient (sensor body) temperature in °C computed from the
        on-chip PTAT sensor.
        """
        c     = self._calib
        ta_dk = frame["ptat_av"] * c["ptat_gradient"] + c["ptat_offset"]
        return (ta_dk - 2732.0) / 10.0

    # ── Calibration / EEPROM ─────────────────────────────────────────────────

    def load_calibration(self) -> None:
        """
        Read all calibration constants from the sensor EEPROM and cache them
        in self._calib.  Called automatically by init().
        """
        c: dict = {}

        def u8(addr: int) -> int:
            return self._eeprom_read(addr, 1)[0]

        def i8(addr: int) -> int:
            v = u8(addr)
            return v if v < 128 else v - 256

        def u16(addr: int) -> int:
            return struct.unpack_from("<H", bytes(self._eeprom_read(addr, 2)))[0]

        def f32(addr: int) -> float:
            return struct.unpack_from("<f", bytes(self._eeprom_read(addr, 4)))[0]

        def i16_arr(addr: int, n: int) -> List[int]:
            raw = self._eeprom_read(addr, n * 2)
            return list(struct.unpack_from(f"<{n}h", bytes(raw)))

        def u16_arr(addr: int, n: int) -> List[int]:
            raw = self._eeprom_read(addr, n * 2)
            return list(struct.unpack_from(f"<{n}H", bytes(raw)))

        # Scalar values
        c["pixcmin"]       = f32(_EEP_PIXCMIN)
        c["pixcmax"]       = f32(_EEP_PIXCMAX)
        c["gradscale"]     = u8 (_EEP_GRADSCALE)
        c["epsilon"]       = u8 (_EEP_EPSILON)

        # Factory calibration trim settings
        c["mbit_calib"]    = u8 (_EEP_MBIT_CALIB)
        c["bias_calib"]    = u8 (_EEP_BIAS_CALIB)
        c["clk_calib"]     = u8 (_EEP_CLK_CALIB)
        c["bpa_calib"]     = u8 (_EEP_BPA_CALIB)
        c["pu_calib"]      = u8 (_EEP_PU_CALIB)

        # User trim settings
        c["mbit_user"]     = u8 (_EEP_MBIT_USER)
        c["bias_user"]     = u8 (_EEP_BIAS_USER)
        c["clk_user"]      = u8 (_EEP_CLK_USER)
        c["bpa_user"]      = u8 (_EEP_BPA_USER)
        c["pu_user"]       = u8 (_EEP_PU_USER)

        # PTAT / VDD calibration values
        c["ptat_gradient"] = f32(_EEP_PTAT_GRAD)
        c["ptat_offset"]   = f32(_EEP_PTAT_OFFSET)
        c["ptat_th1"]      = u16(_EEP_PTAT_TH1)
        c["ptat_th2"]      = u16(_EEP_PTAT_TH2)
        c["vddth1"]        = u16(_EEP_VDDTH1)
        c["vddth2"]        = u16(_EEP_VDDTH2)
        c["vddscgrad"]     = u8 (_EEP_VDDSCGRAD)
        c["vddscoff"]      = u8 (_EEP_VDDSCOFF)

        # Gain / offset
        c["global_off"]    = i8 (_EEP_GLOBAL_OFF)
        c["global_gain"]   = u16(_EEP_GLOBAL_GAIN)

        # Dead-pixel table
        c["nr_def_pix"]    = u8 (_EEP_NR_DEF_PIX)
        c["dead_pix_adr"]  = u16_arr(_EEP_DEADPIX_ADR, 5)
        c["dead_pix_mask"] = list(self._eeprom_read(_EEP_DEADPIX_MASK, 5))

        # Per-pixel arrays (1024 entries each)
        c["thgrad"]        = i16_arr(_EEP_THGRAD,   _PIXELS)
        c["thoffset"]      = i16_arr(_EEP_THOFFSET, _PIXELS)
        c["pij"]           = u16_arr(_EEP_PIJ,      _PIXELS)

        # VDD compensation arrays (256 entries each)
        c["vddcompgrad"]   = i16_arr(_EEP_VDDCOMPGRAD, 256)
        c["vddcompoff"]    = i16_arr(_EEP_VDDCOMPOFF,  256)

        self._calib = c
        
        print('Calibration constants read from EEPROM:')
        print(c)

    def load_lut(
        self,
        ta_cols: List[float],
        dig_rows: List[float],
        data: List[List[float]],
    ) -> None:
        """
        Load the sensor-specific look-up table for object-temperature
        calculation (Section 13.2).

        Heimann Sensor provides this table in a separate file called "Table.c".
        Parse the values from that file and pass them here before calling
        calculate_temperatures().

        Parameters
        ----------
        ta_cols  : Ambient temperature column headers in dK
                   e.g. [2782, 2882, 2982, 3082, 3182, 3282, 3382]
        dig_rows : Compensated pixel voltage row headers (digit values)
                   e.g. [-512, -448, ..., 0, ..., 8192]
        data     : 2-D list [row_index][col_index] of object temperatures in dK
        """
        self._calib["lut"] = {
            "ta_cols":  ta_cols,
            "dig_rows": dig_rows,
            "data":     data,
        }

    # ── Trim register helpers ─────────────────────────────────────────────────

    def set_trim_registers(
        self,
        mbit: int,
        bias_top: int,
        bias_bot: int,
        clk: int,
        bpa_top: int,
        bpa_bot: int,
        pu: int,
    ) -> None:
        """
        Write all seven trim registers with explicit values.

        Parameters
        ----------
        mbit     : Trim Register 1 – REF_CAL (bits 7:6) | MBIT_TRIM (bits 3:0)
                   MBIT_TRIM m=4..12 -> ADC resolution (m+4) bits
        bias_top : BIAS_TRIM_TOP 0-31 -> 1-13 µA  (ADC bias current, top half)
        bias_bot : BIAS_TRIM_BOT 0-31 -> 1-13 µA  (ADC bias current, bottom half)
        clk      : CLK_TRIM 0-63  -> 1-13 MHz  (use clk_trim_to_freq() to check)
        bpa_top  : BPA_TRIM_TOP 0-31 -> 0.2-4.0 µA  (preamplifier common mode)
        bpa_bot  : BPA_TRIM_BOT 0-31 -> 0.2-4.0 µA
        pu       : PU_SDA_TRIM (bits 7:4) | PU_SCL_TRIM (bits 3:0)
                   Encoding: 0x8 = 100kΩ, 0x4 = 50kΩ, 0x2 = 10kΩ, 0x1 = 1kΩ
                   Default 0x88 = 100 kΩ on both SDA and SCL
        """
        pairs = [
            (_CMD_TRIM1, mbit),
            (_CMD_TRIM2, bias_top),
            (_CMD_TRIM3, bias_bot),
            (_CMD_TRIM4, clk),
            (_CMD_TRIM5, bpa_top),
            (_CMD_TRIM6, bpa_bot),
            (_CMD_TRIM7, pu),
        ]
        for cmd, val in pairs:
            self._write_register(cmd, val)
            time.sleep(_T_INTER_REG_MS * 1e-3)

    @staticmethod
    def clk_trim_to_freq(clk_trim: int) -> float:
        """
        Convert a CLK_TRIM register value (0-63) to clock frequency in MHz.
        Formula added in datasheet Rev 9 (2025-03-02), Table 11:
            F_CLK = (F_min + (F_max - F_min) / 63 * CLK_TRIM) MHz
        """
        return 1.0 + (12.0 / 63.0) * clk_trim

    @staticmethod
    def quarter_frame_time_ms(clk_trim: int, mbit: int) -> float:
        """
        Estimated measurement time for one quarter frame in milliseconds.
        From Table 11:  t_fr4 = 32 * (2^MBIT + 4) / F_CLK
        mbit here is the MBIT_TRIM nibble (0..12), NOT the full register byte.
        """
        f_clk_mhz = HTPA32x32d.clk_trim_to_freq(clk_trim)
        return 32.0 * (2 ** mbit + 4) / (f_clk_mhz * 1e6) * 1e3

    # ── Status register ───────────────────────────────────────────────────────

    def read_status(self) -> int:
        """Return the raw 8-bit status register value."""
        write_msg = i2c_msg.write(_ADDR_SENSOR, [_CMD_STATUS])
        read_msg = i2c_msg.read(_ADDR_SENSOR, 1)
        
        self._bus.i2c_rdwr(write_msg, read_msg)
        
        status_byte = read_msg.buf[0]
      
        return status_byte

    def is_eoc(self) -> bool:
        """True when the End-of-Conversion flag is set."""
        return bool(self.read_status() & _BIT_EOC)

    # ── Private: low-level I2C ────────────────────────────────────────────────

    def _write_register(self, cmd: int, value: int) -> None:
        """
        I2C write:  S | ADDR W | CMD | VALUE | P     (Figure 10)
        """
        self._bus.write_byte_data(self._addr, cmd, value)

    def _read_bytes(self, cmd: int, length: int) -> List[int]:
        """
        I2C read with repeated start (Figure 11):
          S | ADDR W | CMD | Sr | ADDR R | D[0] ... D[n] nACK | P
        """
        msg_w = i2c_msg.write(self._addr, [cmd])
        msg_r = i2c_msg.read(self._addr, length)
        self._bus.i2c_rdwr(msg_w, msg_r)
        return list(msg_r)

    def _read_half(self, cmd: int) -> List[int]:
        """
        Read one 258-byte half-array response and decode into 129 uint16 words.
        Each pixel/PTAT value is transmitted MSB first then LSB (big-endian pair).

        Returns
        -------
        list[129]:  index 0 = PTAT (or VDD),  index 1..128 = pixel data
        """
        raw   = self._read_bytes(cmd, 258)
        words = [(raw[i] << 8) | raw[i + 1] for i in range(0, 258, 2)]
        return words

    def _wait_eoc(self) -> None:
        """Poll the status register until EOC is set or the timeout expires."""
        deadline = time.monotonic() + self._timeout_ms * 1e-3
        while not self.is_eoc():
            if time.monotonic() > deadline:
                raise HTPA32x32dError(
                    f"Timeout ({self._timeout_ms} ms) waiting for EOC."
                )
            time.sleep(_T_POLL_MS * 1e-3)

    # ── Private: electrical offsets ───────────────────────────────────────────

    def _read_electrical_offsets(self) -> List[int]:
        """
        Trigger a BLIND conversion and read all 256 electrical offset values.

        Returns list[256] of uint16 values:
          [0..127]   top-half electrical offsets
          [128..255] bottom-half electrical offsets
        """
        self._write_register(_CMD_CONFIG, _BIT_WAKEUP | _BIT_START | _BIT_BLIND)
        self._wait_eoc()

        top = self._read_half(_CMD_READ_TOP)  # words[1..128] = el_off[0..127]
        bot = self._read_half(_CMD_READ_BOT)  # words mirrored as per Table 18

        # Top half: direct mapping (Table 17)
        el = list(top[1:129])

        # Bottom half: mirrored readout order (Table 18)
        # words[1..32]   -> el_offset[224..255]
        # words[33..64]  -> el_offset[192..223]
        # words[65..96]  -> el_offset[160..191]
        # words[97..128] -> el_offset[128..159]
        bot_part = [0] * 128
        for sub in range(4):
            src = sub * 32 + 1
            dst = (3 - sub) * 32          # destination within bot_part
            for k in range(32):
                bot_part[dst + k] = bot[src + k]
        el += bot_part

        return el  # 256 values

    # ── Private: sensitivity coefficients ────────────────────────────────────

    def _calc_pixc(self, c: dict) -> List[float]:
        """
        Compute per-pixel sensitivity coefficient PixC_ij for all 1024 pixels.

        Formula (Section 12.5):
          PixC_ij = (P_ij * (PixCmax - PixCmin) / 65535 + PixCmin)
                    * epsilon/100
                    * GlobalGain/10000
        """
        pmin  = c["pixcmin"]
        pmax  = c["pixcmax"]
        eps   = c["epsilon"] / 100.0
        gg    = c["global_gain"] / 10000.0
        scale = (pmax - pmin) / 65535.0
        return [(c["pij"][n] * scale + pmin) * eps * gg for n in range(_PIXELS)]

    # ── Private: LUT bilinear interpolation (Section 13.2) ───────────────────

    def _lut_interpolate(
        self, v_pixc: float, ta_dk: float, c: dict
    ) -> float:
        """
        Return object temperature in dK via bilinear interpolation into the
        look-up table.

        If no LUT has been loaded the ambient temperature is returned as a
        best-effort fallback.
        """
        if "lut" not in c:
            return ta_dk   # no LUT – fall back to ambient temperature

        lut      = c["lut"]
        ta_cols  = lut["ta_cols"]
        dig_rows = lut["dig_rows"]
        data     = lut["data"]

        # Clamp to table extent
        ta  = max(ta_cols[0],  min(ta_cols[-1],  ta_dk))
        dig = max(dig_rows[0], min(dig_rows[-1], v_pixc))

        # Column index (ambient temperature axis)
        ci = 0
        for i in range(len(ta_cols) - 1):
            if ta_cols[i + 1] > ta:
                ci = i
                break
        else:
            ci = len(ta_cols) - 2

        # Row index (digit axis)
        ri = 0
        for i in range(len(dig_rows) - 1):
            if dig_rows[i + 1] > dig:
                ri = i
                break
        else:
            ri = len(dig_rows) - 2

        # Fractional positions
        dta  = ta_cols[ci + 1]  - ta_cols[ci]
        ddig = dig_rows[ri + 1] - dig_rows[ri]
        ta_f  = (ta  - ta_cols[ci])  / dta  if dta  != 0 else 0.0
        dig_f = (dig - dig_rows[ri]) / ddig if ddig != 0 else 0.0

        # Bilinear interpolation over the four surrounding LUT cells
        q11 = data[ri][ci]
        q12 = data[ri][ci + 1]
        q21 = data[ri + 1][ci]
        q22 = data[ri + 1][ci + 1]

        return (q11 * (1 - ta_f) * (1 - dig_f)
                + q12 * ta_f       * (1 - dig_f)
                + q21 * (1 - ta_f) * dig_f
                + q22 * ta_f       * dig_f)

    # ── Private: pixel masking (Section 13.1) ────────────────────────────────

    def _apply_pixel_masking(self, temps: List[float], c: dict) -> None:
        """
        Replace dead-pixel temperatures with the average of their nominated
        neighbours.  Operates in-place on *temps*.

        The neighbour layout from the datasheet (top half):
          128  1   2
           64  X   4
           32  16  8
        (bottom half layout is vertically mirrored)
        """
        nr = c["nr_def_pix"]
        if nr == 0:
            return

        # Neighbour (delta_row, delta_col) in bit order [7..0] for top half
        TOP_OFFSETS = [
            (-1, -1),  # bit 7 = 128
            (-1,  0),  # bit 0 = 1
            (-1, +1),  # bit 1 = 2
            ( 0, -1),  # bit 6 = 64
            ( 0, +1),  # bit 2 = 4
            (+1, -1),  # bit 5 = 32
            (+1,  0),  # bit 4 = 16
            (+1, +1),  # bit 3 = 8
        ]
        # Bottom half: vertically mirrored
        BOT_OFFSETS = [
            (+1, -1),  # bit 7
            (+1,  0),  # bit 0
            (+1, +1),  # bit 1
            ( 0, -1),  # bit 6
            ( 0, +1),  # bit 2
            (-1, -1),  # bit 5
            (-1,  0),  # bit 4
            (-1, +1),  # bit 3
        ]

        for i in range(nr):
            raw_adr  = c["dead_pix_adr"][i]
            mask_val = c["dead_pix_mask"][i]

            # Convert stored EEPROM address to actual pixel number (Section 13.1)
            if raw_adr < 0x0200:
                pix    = raw_adr
                is_top = True
            else:
                # Bottom half: adapted address formula from datasheet
                col_k = raw_adr % _COLS
                pix   = 1024 + 512 - raw_adr + col_k * 2 - 32
                is_top = False

            if not (0 <= pix < _PIXELS):
                continue

            row     = pix // _COLS
            col     = pix  % _COLS
            offsets = TOP_OFFSETS if is_top else BOT_OFFSETS

            neighbour_vals: List[float] = []
            for bit_pos, (dr, dc) in enumerate(offsets):
                if mask_val & (1 << (7 - bit_pos)):
                    nr_ = row + dr
                    nc_ = col + dc
                    if 0 <= nr_ < _ROWS and 0 <= nc_ < _COLS:
                        neighbour_vals.append(temps[nr_ * _COLS + nc_])

            if neighbour_vals:
                temps[pix] = sum(neighbour_vals) / len(neighbour_vals)

    # ── Private: stack buffer helpers ────────────────────────────────────────

    def _update_stack(self, stack: list, value) -> None:
        """Append a value to a rolling FIFO of depth self._stack_depth."""
        stack.append(value)
        if len(stack) > self._stack_depth:
            stack.pop(0)

    # ── Private: EEPROM access (24AA64 at I2C address 0x50) ──────────────────

    def _eeprom_read(self, mem_addr: int, length: int) -> List[int]:
        """
        Read *length* bytes from the internal 24AA64 EEPROM starting at the
        16-bit memory address *mem_addr*.

        The 24AA64 protocol:
          S | ADDR W | AddrMSB | AddrLSB | Sr | ADDR R | D[0]...D[n] nACK | P
        Reads are split into 128-byte chunks to stay within typical I2C limits.
        """
        result: List[int] = []
        MAX_CHUNK = 128

        while length > 0:
            chunk    = min(length, MAX_CHUNK)
            addr_msb = (mem_addr >> 8) & 0xFF
            addr_lsb =  mem_addr       & 0xFF
            
            # Construct message to send
            msg_w = i2c_msg.write(self._eep_addr, [addr_msb, addr_lsb])
            
            # Construct message object that answer is writting back to
            msg_r = i2c_msg.read(self._eep_addr, chunk)
            
            # Send message and read answer
            self._bus.i2c_rdwr(msg_w, msg_r)

            result   += list(msg_r)
            mem_addr += chunk
            length   -= chunk

        return result


# ── Convenience utilities ─────────────────────────────────────────────────────

def dk_to_celsius(dk: float) -> float:
    """Convert deci-Kelvin to degrees Celsius."""
    return (dk - 2732.0) / 10.0


def celsius_to_dk(celsius: float) -> float:
    """Convert degrees Celsius to deci-Kelvin."""
    return celsius * 10.0 + 2732.0


def frame_to_2d(flat: List[float]) -> List[List[float]]:
    """Reshape a flat 1024-element list into a 32x32 row-major grid."""
    return [flat[r * 32 : (r + 1) * 32] for r in range(32)]


def print_frame(temps: List[float], fmt: str = "{:6.1f}") -> None:
    """Pretty-print a temperature frame to stdout."""
    grid = frame_to_2d(temps)
    for row in grid:
        print(" ".join(fmt.format(v) for v in row))
