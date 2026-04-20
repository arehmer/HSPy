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
import threading
import queue
import numpy as np
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

# ── Configuration register bit positions (Table 6) ───────────────────────────
_BIT_WAKEUP   = (1 << 0)  # 1 = on, 0 = sleep
_BIT_BLIND    = (1 << 1)  # 1 = sample electrical offsets
_BIT_VDD_MEAS = (1 << 2)  # 1 = measure VDD instead of PTAT
_BIT_START    = (1 << 3)  # trigger conversion

# ── MBIT+REFCAL bit masks ───────────────────────────
_BITMASK_MBIT   = (1 << 4) - 1
_BITMASK_REFCAL = ((1 << 2) - 1) << 4

# ── Status register bit positions (Table 7) ───────────────────────────────────
_BIT_EOC = (1 << 0)  # End-of-conversion bitwise mask

# ── EEPROM memory map (Figure 12 of datasheet) ────────────────────────────────
_EEP_PIXCMIN            = 0x0000   # float32  minimum sensitivity coefficient
_EEP_PIXCMAX            = 0x0004   # float32  maximum sensitivity coefficient
_EEP_GRADSCALE          = 0x0008   # uint8    thermal gradient scaling exponent
_EEP_EPSILON            = 0x000D   # uint8    emissivity (100 = 1.00)
_EEP_MBITREFCAL_CALIB   = 0x001A   # uint8    MBIT used during factory calibration
_EEP_BIAS_CALIB         = 0x001B   # uint8s
_EEP_CLK_CALIB          = 0x001C   # uint8
_EEP_BPA_CALIB          = 0x001D   # uint8
_EEP_PU_CALIB           = 0x001E   # uint8
_EEP_VDDTH1             = 0x0026   # uint16 LE  supply voltage at calibration temp 1
_EEP_VDDTH2             = 0x0028   # uint16 LE  supply voltage at calibration temp 2
_EEP_PTAT_GRAD          = 0x0034   # float32
_EEP_PTAT_OFFSET        = 0x0038   # float32
_EEP_PTAT_TH1           = 0x003C   # uint16 LE  PTAT at calibration temp 1
_EEP_PTAT_TH2           = 0x003E   # uint16 LE  PTAT at calibration temp 2
_EEP_VDDSCGRAD          = 0x004E   # uint8    VddComp gradient scaling exponent
_EEP_VDDSCOFF           = 0x004F   # uint8    VddComp offset scaling exponent
_EEP_GLOBAL_OFF         = 0x0054   # int8     global object temperature offset
_EEP_GLOBAL_GAIN        = 0x0055   # uint16 LE
_EEP_MBITREFCAL_USER    = 0x0060   # uint8    user-settable trim copies
_EEP_BIAS_USER          = 0x0061
_EEP_CLK_USER           = 0x0062
_EEP_BPA_USER           = 0x0063
_EEP_PU_USER            = 0x0064
_EEP_NR_DEF_PIX         = 0x007F   # uint8    number of dead pixels (0-5)
_EEP_DEADPIX_ADR        = 0x0080   # 5 x uint16 LE  dead pixel addresses
_EEP_DEADPIX_MASK       = 0x008A   # 5 x uint8      neighbour mask per dead pixel
_EEP_VDDCOMPGRAD        = 0x0340   # 256 x int16 LE
_EEP_VDDCOMPOFF         = 0x0540   # 256 x int16 LE
_EEP_THGRAD             = 0x0740   # 1024 x int16 LE  thermal gradient per pixel
_EEP_THOFFSET           = 0x0F40   # 1024 x int16 LE  thermal offset per pixel
_EEP_PIJ                = 0x1740   # 1024 x uint16 LE sensitivity coefficient

# ── Array geometry ────────────────────────────────────────────────────────────
_ROWS      = 32
_COLS      = 32
_PIXELS    = _ROWS * _COLS     # 1024
_HALF      = int(_PIXELS // 2) # 512
_BLOCK_PX  = 128               # pixels per block per half
_BLOCKS_ = 4

# ── Miscellaneous constants ───────────────────────────────────────────────────
_PCSCALEVAL      = 1e8    # Section 12.5
_T_INTER_REG_MS  = 5      # min delay between writing trim registers
_T_WAKEUP_US     = 80     # wakeup time after WAKEUP command (Table 5)
_T_POLL_MS       = 1      # EOC polling interval
_Tbuf            = 0.5    # µs,  time between STOP / START  
_F_CLK_MIN       = 1      # MHz, min. clock frequency
_F_CLK_MAX       = 13     # MHz, max. clock frequency
_N_PTAT           = 8      # Number of PTATs


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

        # Threading necessities
        self.i2c_queue = queue.Queue(maxsize=10)       # bounded to avoid memory buildup
        self.i2c_stop = threading.Event()              # if event is set, threads are killed
        
        # Rolling stack buffers (depth 8) for stable averaging
        self._ptat_stack: List[float] = []
        self._vdd_stack:  List[float] = []
        self._el_offset_stack: List[List[int]] = []
        self._stack_depth = 8
        
        # Open the I2C bus
        self.open()
        
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
        mbitrefcal = self._calib[f"mbitrefcal_{key}"]
        bias       = self._calib[f"bias_{key}"]
        clk        = self._calib[f"clk_{key}"]
        bpa        = self._calib[f"bpa_{key}"]
        pu         = self._calib[f"pu_{key}"]

        for cmd, val in [
            (_CMD_TRIM1, mbitrefcal),
            (_CMD_TRIM2, bias),   # BIAS_TRIM_TOP
            (_CMD_TRIM3, bias),   # BIAS_TRIM_BOT  (same value)
            (_CMD_TRIM4, clk),
            (_CMD_TRIM5, bpa),    # BPA_TRIM_TOP
            (_CMD_TRIM6, bpa),    # BPA_TRIM_BOT  (same value)
            (_CMD_TRIM7, pu),
        ]:
            self._write_register(cmd, val)
            time.sleep(_T_INTER_REG_MS * 1e-3)
            
        # Step 4 - calculate approximative conversion time
        self._calc_conversion_time(use_calib_settings)
        
    def _calc_conversion_time(self,use_calib_settings: bool):
        
        key  = "calib" if use_calib_settings else "user"
        MBIT = self._calib[f"mbit_{key}"]
        CLK_TRIM = self._calib[f"clk_{key}"]
        
        F_CLK = (_F_CLK_MIN + (_F_CLK_MAX-_F_CLK_MIN) / 63 * CLK_TRIM)
        
        t_fr4 = 32*(2**MBIT + 4) / (F_CLK*1E3)
        
        self._calib["F_CLK"] = F_CLK
        self._calib["t_fr4"] = t_fr4

    def sleep(self) -> None:
        """Put sensor into sleep state (~9 µA standby current)."""
        self._write_register(_CMD_CONFIG, 0x00)

    def wakeup(self) -> None:
        """Wake sensor from sleep state."""
        self._write_register(_CMD_CONFIG, _BIT_WAKEUP)
        time.sleep(_T_WAKEUP_US * 1e-6)


    
    def read_thread(self):
        """
        Threadable function for i2c readouts 
        """
               
        while not self.i2c_stop.is_set():
            
            data_dict = self.read_frame(measure_vdd = True)                         # read pixels, ptat, vdd
            data_dict.update(self.read_electrical_offsets(measure_vdd = False))     # read electrical offsets
            
            self.i2c_queue.put(data_dict)                                        # put in queue, blocks if full (backpressure)

            
    def postprocessing_thread(self,applyCalib:bool=True):
        """
        Threadable function for converting raw i2c data into the appropriate 
        format (sorting, rearranging, conversion)
        """
        
        timeout = self._calib['t_fr4'] * 4 * 1.1                        # ms, timeout is frame conversion time + 10 %
        
        while not self.i2c_stop.is_set():
            try:
                raw_data = self.i2c_queue.get(timeout=timeout/1E3)         # blocks until data arrives, or times out
                proc_data = self.convert_i2c_data(raw_data)            # convert raw i2c data in place
                
                if applyCalib:
                    
                    # Calculcate Temperatures from pixel voltages
                    temp_data = self.apply_calib(proc_data)
                    
                
                print(temp_data['pixels_comp'][0:5])
            except queue.Empty:
                continue                                          # no data yet, loop and check stop_event
    
    # ── Continuous frame readout  ────────────────────────────────────────────
    def start_stream(self):
        """
        Acquire complete frames in a loop

        Returns
        -------
        None.

        """
        
        
        t_reader    = threading.Thread(target=self.read_thread)
        t_conv      = threading.Thread(target=self.postprocessing_thread)
        
        t_reader.start()
        t_conv.start()

        time.sleep(5)
        
        self.i2c_stop.set()
        
        t_reader.join()
        t_conv.join()

        
    def stop_stream(self):
        
        pass

    # ── Single Frame readout ─────────────────────────────────────────────────

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
          "time"       - time.time() at function return
        """
        pixels: dict[int, dict[str, list[int]]] = \
            {b: {'top': [0] * _BLOCK_PX, 'bot': [0] * _BLOCK_PX} \
             for b in range(4)}
                                        
        ptat: dict[str, list[int]] =  {'top': [] , 'bot': [] }

        vdd: dict[str, list[int]] =  {'top': [] , 'bot': [] }
        
        t_fr4 = self._calib["t_fr4"]                                           # ms, block conversion time

        # ── Read four blocks ──────────────────────────────────────────────
        for block in range(4):
            
            # Write to configuration register:
            # ------------------------------------------------------
            # | 7 | 6 | 5 | 4 |   3   |     2    |   1   |   0    |
            # |  RFU  | BLOCK | Start | VDD_MEAS | BLIND | WAKEUP 
            # ------------------------------------------------------| 
            config = _BIT_WAKEUP | _BIT_START | (block << 4)
            
            if measure_vdd:
                config |= _BIT_VDD_MEAS
            
            # Write to config register 
            self._write_register(_CMD_CONFIG, config)
            
            # Pause 90 % of the approximate conversion time before checking
            # if end of conversion is reached
            time.sleep(0.9*t_fr4*1E-3)
            
            # Start checking for end of conversion bit
            self._wait_eoc()

            # Read frame from register
            top = self._read_half(_CMD_READ_TOP)  # 129 words
            bot = self._read_half(_CMD_READ_BOT)  # 129 words
            
            # Word[0] = PTAT (or VDD if VDD_MEAS set)
            if measure_vdd:
                vdd['top'].append(top[0])
                vdd['bot'].append(bot[0])
            else:
                ptat['top'].append(top[0])
                ptat['bot'].append(bot[0])
            
            pixels[block]['top'] = top[1::]
            pixels[block]['bot'] = bot[1::]
            
        # ------ Average PTAT / VDD -------------------------------------------
        if measure_vdd:
            vdd['top'] = [int(np.round(sum (vdd['top']) / len (vdd['top'])))]
            vdd['bot'] = [int(np.round(sum (vdd['bot']) / len (vdd['bot'])))]
        else:
            ptat['top'] = [int(np.round(sum (ptat['top']) / len (ptat['top'])))]
            ptat['bot'] = [int(np.round(sum (ptat['bot']) / len (ptat['bot'])))]
        
        # ------ Return -------------------------------------------------------
        if measure_vdd:
            return {
                "pixels":     pixels,
                "vdd":        vdd,
                "t":          time.time()
            }
        else:
            return {
                "pixels":     pixels,
                "ptat":       ptat,
                "t":          time.time()
            }
    
    def convert_i2c_data(self,raw_data) -> dict :
        
        n_blocks = int(_PIXELS / 2 / _BLOCK_PX)
        n_ptat = 2
        n_vdd = 2
        
        pix_array = np.zeros((_ROWS,_COLS),dtype = np.uint16)                # zero array for storing rearranged pixel data 
        ptat_array = np.zeros((n_ptat,),dtype = np.uint16)                   # zero array for storing rearranged ptat data 
        vdd_array = np.zeros((n_vdd,),dtype = np.uint16)                     # zero array for storing rearranged ptat data 

        
        if 'pixels' in raw_data.keys():
            
            for block in range(4):
                    top = raw_data['pixels'][block]['top']      # block data from top half
                    bot = raw_data['pixels'][block]['bot']      # block data from bottom half
                    
                    for r in range(n_blocks):
                    
                        pix_array[block+r*n_blocks,:] = \
                            top[_COLS*r:_COLS*(r+1)]
                        pix_array[_ROWS-1-block-r*n_blocks,:] = \
                            bot[_COLS*r:_COLS*(r+1)]
                                
        if 'ptat' in raw_data.keys():
            
            top = raw_data['ptat']['top']                   # ptat data from top half
            bot = raw_data['ptat']['bot']                   # ptat data from bottom half
                           
            ptat_array[0] = top[0]
            ptat_array[1] = bot[0]
        
        if 'vdd' in raw_data.keys():
            
            top = raw_data['vdd']['top']                    # vdd data from top half
            bot = raw_data['vdd']['bot']                    # vdd data from bottom half
                           
            vdd_array[0] = top[0]
            vdd_array[1] = bot[0]      
                
                
        if 'eloff' in raw_data.keys():
            
            top = raw_data['eloff']['top']      # block data from top half
            bot = raw_data['eloff']['bot']      # block data from bottom half
            
            topbot = top + bot
            
            eloff_array = self._blocks_to_array(topbot)
        
        data = {}
        
        if 'pixels' in raw_data.keys():
            data['pixels'] = pix_array.flatten()
            
        if 'eloff' in raw_data.keys():
            data['eloff'] = eloff_array.flatten()

        if 'ptat' in raw_data.keys():
            data['ptat'] = ptat_array.flatten()

        if 'vdd' in raw_data.keys():
            data['vdd'] = vdd_array.flatten()            
        
        return data


    # ── Temperature calculation ──────────────────────────────────────────────
    def calculate_Tamb(self,data:dict) -> dict:
        """
        Calculates ambient temperature Tamb, if necessary calibration data 
        is available

        Parameters
        ----------
        data : dict
            DESCRIPTION.

        Returns
        -------
        dict
            DESCRIPTION.

        """
        
        
        
    

    def apply_calib(self, data: dict) -> List[float]:
        """
        Convert a raw frame (from read_frame()) into per-pixel compensated
        voltages in digits. Also computes ambient temperature as a side-effect.

        Processing pipeline (Sections 12.1 - 12.5):
          1. Ambient temperature from PTAT
          2. Thermal offset compensation per pixel
          3. Electrical offset compensation per pixel
          4. VDD (supply voltage) compensation per pixel  [if VDD available]
          5. Sensitivity (PixC) compensation
          6. Dead-pixel masking

        Returns list[1024] of temperatures in °C.
        """
        if not self._calib:
            raise HTPA32x32dError(
                "Calibration not loaded. Call init() or load_calibration() first."
            )

        
        c       = self._calib           # Calibration data dictionary

        # ------------- 12.1 Ambient Temperature ------------------------------
        PTAT_grad = c["ptat_gradient"]
        PTAT_off  = c["ptat_offset"]
        
        PTAT_avg = data['ptat'].mean()
        Tamb = PTAT_avg * PTAT_grad + PTAT_off
        
        data['ptat_avg'] = PTAT_avg
        data['Tamb'] = Tamb
        
        
        # ------------- 12.2 Thermal Offset -----------------------------------
        Th_grad = c["thgrad_arr"]
        Th_off  = c["thoffset_arr"]        
        gradScale = c["gradscale"]
      
        V_ThComp = + (Th_grad * PTAT_avg) / 2**(gradScale) + Th_off
        
        V_comp = data['pixels'] - V_ThComp
      
        # ------------- 12.3 Electrical Offset --------------------------------
        V_ElComp = data['eloff']
        
        V_comp = V_comp - V_ElComp
        
        # ------------- 12.4 Vdd Compensation- --------------------------------
        VddCompGrad = c["vddcompgrad_arr"]
        VddCompOff  = c["vddcompoff_arr"]
        VddScGrad   = c["vddscgrad"]
        VddScOff    = c["vddscoff"]
        
        VddTh1      = c["vddth1"]
        VddTh2      = c["vddth2"] 
        PTAT_Th1    = c["ptat_th1"]
        PTAT_Th2    = c["ptat_th2"]
               
        
        VDD_avg = data['vdd'].mean()
        
        
        V_VddComp = ((VddCompGrad * PTAT_avg)/(2**VddScGrad) + VddCompOff) / \
            (2**VddScOff) * \
                (VDD_avg - VddTh1 - ((VddTh2-VddTh1)/(PTAT_Th2-PTAT_Th1)) * \
                 (PTAT_avg-PTAT_Th1))
         
        V_comp = V_comp - V_VddComp
        
        # ------------- 12.5 Object Temperature -------------------------------
        Pij         = c["pij_arr"]
        PixC_max    = c["pixcmax"]
        PixC_min    = c["pixcmin"]
        eps         = c["epsilon"]
        GlobalGain  = c["global_gain"]
        
        PCSCELEVAL = 1E8
        
        PixCij = (Pij * (PixC_max-PixC_min) / 65535 +  PixC_min) * \
            eps/100 * GlobalGain/10000
        
            
        V_comp = V_comp * PCSCELEVAL / PixCij
            
        data['pixels_comp'] = V_comp
        

        # ── Step 6: Replace dead pixels with neighbour average ────────────
        # self._apply_pixel_masking(results, c)

        return data

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
        
        def unpack_mbit(mbit_raw:int) -> int:
            mbit = mbit_raw & _BITMASK_MBIT
            return mbit

        def unpack_refcal(mbit_raw:int) -> int:
            refcal = (mbit_raw & _BITMASK_REFCAL) >> 4
            return refcal
        
        # Scalar values
        c["pixcmin"]       = f32(_EEP_PIXCMIN)
        c["pixcmax"]       = f32(_EEP_PIXCMAX)
        c["gradscale"]     = u8 (_EEP_GRADSCALE)
        c["epsilon"]       = u8 (_EEP_EPSILON)

        # Factory calibration trim settings
        c["mbitrefcal_calib"]   = u8 (_EEP_MBITREFCAL_CALIB)
        c["mbit_calib"]         = unpack_mbit(c["mbitrefcal_calib"])
        c["refcal_calib"]       = unpack_refcal(c["mbitrefcal_calib"])
        c["bias_calib"]         = u8 (_EEP_BIAS_CALIB)
        c["clk_calib"]          = u8 (_EEP_CLK_CALIB)
        c["bpa_calib"]          = u8 (_EEP_BPA_CALIB)
        c["pu_calib"]           = u8 (_EEP_PU_CALIB)

        # User trim settings
        c["mbitrefcal_user"]    = u8 (_EEP_MBITREFCAL_USER)
        c["mbit_calib"]         = unpack_mbit(c["mbitrefcal_user"])
        c["refcal_calib"]       = unpack_refcal(c["mbitrefcal_user"])
        c["bias_user"]          = u8 (_EEP_BIAS_USER)
        c["clk_user"]           = u8 (_EEP_CLK_USER)
        c["bpa_user"]           = u8 (_EEP_BPA_USER)
        c["pu_user"]            = u8 (_EEP_PU_USER)

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
        
        
        
        
        # Rearrange per-pixel arrays to correspond to actual pixel order
        c["thgrad_arr"] =  np.hstack([c["thgrad"][0:_HALF],
                                     np.flip(c["thgrad"][_HALF::])]).reshape((_ROWS,_COLS)).flatten()
        c["thoffset_arr"] =  np.hstack([c["thoffset"][0:_HALF],
                                       np.flip(c["thoffset"][_HALF::])]).reshape((_ROWS,_COLS)).flatten()
        c["pij_arr"] =  np.hstack([c["pij"][0:_HALF],
                                  np.flip(c["pij"][_HALF::])]).reshape((_ROWS,_COLS)).flatten()

        # VDD compensation arrays (256 entries each)
        c["vddcompgrad"]   = i16_arr(_EEP_VDDCOMPGRAD, 256)
        c["vddcompoff"]    = i16_arr(_EEP_VDDCOMPOFF,  256)
        
        # Reshape VDD calibration data to array of same dimension as pixels
        c["vddcompgrad_arr"] = self._blocks_to_array(c["vddcompgrad"])
        c["vddcompoff_arr"] = self._blocks_to_array(c["vddcompoff"])
        

        self._calib = c
        
    def _blocks_to_array(self,block_data:list) -> np.ndarray:
        
        # Check input size 
        expected_len = _PIXELS / _BLOCKS_
        
        if not len(block_data) == expected_len:
            raise ValueError(f'Length of block_data is { len(block_data)}, ', 
                             f'expected was {expected_len}.')
        
        top = block_data[0:_COLS*_BLOCKS_]
        bot = block_data[_COLS*_BLOCKS_::]      
        
        top = np.array(top).reshape((_BLOCKS_,_COLS))
        bot = np.array(bot).reshape((_BLOCKS_,_COLS))
        
        # Flip bottom half
        bot = np.flipud(bot)
        
        # Repeat blocks using tile
        top = np.tile(top,(_BLOCKS_,1))
        bot = np.tile(bot,(_BLOCKS_,1))
        
        # Concatenate to one array
        array = np.vstack([top,bot]).flatten()
        
        return array
        

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
        
        status_byte = int.from_bytes(read_msg.buf[0])
      
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

    def read_electrical_offsets(self, measure_vdd: bool = False) -> List[int]:
        """
        Trigger a BLIND conversion and read all 256 electrical offset values.

        Returns list[256] of uint16 values:
          [0..127]   top-half electrical offsets
          [128..255] bottom-half electrical offsets
        """
        
        eloff: dict[str, list[int]] = {'top': [0] * _BLOCK_PX, 'bot': [0] * _BLOCK_PX}
                
        ptat: dict[str, list[int]] = {'top': [] , 'bot': [] }

        vdd: dict[str, list[int]] = {'top': [] , 'bot': [] }
        
        t_fr4 = self._calib["t_fr4"]                                           # ms, block conversion time

        # Write to config register 
        self._write_register(_CMD_CONFIG, _BIT_WAKEUP | _BIT_START | _BIT_BLIND)
        
        # Pause 90 % of the approximate conversion time before checking
        # if end of conversion is reached
        time.sleep(0.9*t_fr4*1E-3)
        
        # Start checking for end of conversion bit
        self._wait_eoc()
        
        # Read frame from register
        top = self._read_half(_CMD_READ_TOP)  # 129 words
        bot = self._read_half(_CMD_READ_BOT)  # 129 words
        
        if measure_vdd:
            vdd['top'] = [top[0]]
            vdd['bot'] = [bot[0]]
        else:
            ptat['top'] = [top[0]]
            ptat['bot'] = [bot[0]]
        
        eloff['top'] = top[1::]
        eloff['bot'] = bot[1::]
        
        if measure_vdd:
            return {
                "eloff":     eloff,
                "vdd":       vdd,
                "t":         time.time()
            }
        else:
            return {
                "eloff":     eloff,
                "ptat":      ptat,
                "t":         time.time()
            }

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
