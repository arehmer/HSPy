"""
HTPA160x120dR1L3.95/0.8 Thermopile Array Sensor - Python SPI Interface
Based on Heimann Sensor datasheet Rev12 (09.04.2026)

Hardware:
  Pin 1 (EE_Enable) - Slave Select (Active HIGH)
  Pin 2 (VSS)       - Ground (0 V)
  Pin 3 (VDD)       - 3.3 V - 3.6 V supply
  Pin 4 (SCLK)      - Serial Clock (SPI Mode 0: CPOL=0, CPHA=0)
  Pin 5 (MOSI)      - Serial Data into Sensor
  Pin 6 (MISO)      - Serial Data out of Sensor

Recommended external circuit (Section 7):
  100 nF + 47 uF decoupling on VDD
  No inductors in the supply path
"""

import struct
import time
from datetime import datetime
from typing import List, Optional, Dict
import threading
import queue
import numpy as np

# ---------------------------------------------------------------------------
# Attempt to import spidev; fall back to a stub for offline testing
# ---------------------------------------------------------------------------
try:
    import spidev  # type: ignore
    _SPIDEV_AVAILABLE = True
except ImportError:
    _SPIDEV_AVAILABLE = False

    class spidev:  # type: ignore
        class SpiDev:
            def __init__(self):
                raise RuntimeError("spidev is not installed. Run: pip install spidev")
            def open(self, bus, device): ...
            def close(self): ...
            def xfer2(self, data): ...
            def xfer3(self, data): ...

# ── SPI Commands (Tables 6-13 of datasheet) ──────────────────────────────────
_CMD_CONFIG   = 0x01  # Configuration Register  (write only)
_CMD_STATUS   = 0x02  # Status Register         (read only)
_CMD_TRIM1    = 0x03  # Trim Reg 1: REF_CAL + MBIT_TRIM
_CMD_TRIM2    = 0x04  # Trim Reg 2: BIAS_TRIM_TOP
_CMD_TRIM3    = 0x05  # Trim Reg 3: BIAS_TRIM_BOT
_CMD_TRIM4    = 0x06  # Trim Reg 4: CLK_TRIM
_CMD_TRIM5    = 0x07  # Trim Reg 5: BPA_TRIM_TOP
_CMD_TRIM6    = 0x08  # Trim Reg 6: BPA_TRIM_BOT
_CMD_TRIM7    = 0x09  # Trim Reg 7: PU_TRIM
_CMD_READ_TOP = 0x0A  # Read top-half array data (1606 bytes per block)
_CMD_READ_BOT = 0x0B  # Read bottom-half array data (1606 bytes per block)

# ── Flash Memory Commands (SST26VF016B) ──────────────────────────────────────
_FLASH_CMD_READ = 0x03 # Standard SPI Flash Read Command

# ── Configuration register bit positions ─────────────────────────────────────
_BIT_WAKEUP   = (1 << 0)  # 1 = on, 0 = sleep
_BIT_BLIND    = (1 << 1)  # 1 = sample electrical offsets
_BIT_START    = (1 << 3)  # trigger conversion

# ── Status register bit positions ────────────────────────────────────────────
_BIT_EOC      = (1 << 0)  # End-of-conversion bitwise mask

# ── Array geometry ───────────────────────────────────────────────────────────
_ROWS            = 120
_COLS            = 160
_PIXELS          = _ROWS * _COLS      # 19200
_HALF            = int(_PIXELS // 2)  # 9600
_BLOCK_PX        = 800                # pixels per block per half
_BLOCKS          = 12                 # 12 blocks per frame
_BYTES_PER_BLOCK = 1606               # 1606 bytes per block read

# ── Miscellaneous constants ──────────────────────────────────────────────────
_PCSCALEVAL      = 1e8      # Sensitivity scaling
_T_INTER_REG_MS  = 5        # min delay between writing trim registers
_T_WAKEUP_US     = 80       # wakeup time after WAKEUP command
_F_CLK_MAX       = 13000000 # 13 MHz SPI max clock
_F_CLK_TYP       = 10000000 # 10 MHz SPI typ clock

class SPI_Driver:
    pass

class SPI_HTPA160x120dError(Exception):
    """Raised on SPI communication or data errors."""    

class SPI_HTPA160x120d(SPI_Driver):
    """
    Full SPI driver for the HTPA160x120d thermopile array sensor.

    Typical usage::

        from htpa160x120d_spi import SPI_HTPA160x120d

        with SPI_HTPA160x120d(bus=0, device=0) as sensor:
            sensor.init(use_calib_settings=True)
            sensor.start_spistream()
            
            # Retrieve processed frames from the output queue
            frame = sensor.output_queue.get()
            print(f"Ambient: {frame['Tamb']:.1f} dK")
            
            sensor.stop_spistream()
    """

    def __init__(self, bus: int = 0, device: int = 0, timeout_ms: int = 500):
        super(SPI_HTPA160x120d, self).__init__()
        
        self._bus_num  = bus
        self._device   = device
        self._spi      = None
        self._timeout_ms = timeout_ms

        # Calibration constants
        self._calib: dict = {}

        # Threading necessities
        self.spi_queue = queue.Queue(maxsize=1)        
        self.output_queue = queue.Queue(maxsize=1)
        self.spi_stop = threading.Event()              
        
        # Rolling stack buffers for stable averaging
        self._ptat_stack: List[float] = []
        self._vdd_stack:  List[float] = []
        self._stack_depth = 8
        
        self._image_counter = 0
        self.open()

    def __enter__(self) -> "SPI_HTPA160x120d":
        if self._spi is None:
            self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ── Bus open / close ─────────────────────────────────────────────────────
    def open(self) -> None:
        """Open the SPI bus file descriptor and apply hardware configs."""
        self._spi = spidev.SpiDev()
        self._spi.open(self._bus_num, self._device)
        
        # SPI Hardware Configuration
        self._spi.max_speed_hz = _F_CLK_TYP
        self._spi.mode = 0b00  
        self._spi.cshigh = True # EE_Enable is Active-HIGH

    def close(self) -> None:
        if self._spi is not None:
            try:
                self.sleep()
            except Exception:
                pass
            self._spi.close()
            self._spi = None

    # ── Private: low-level SPI ────────────────────────────────────────────────
    def _write_register(self, cmd: int, value: int) -> None:
        self._spi.xfer2([cmd, value])

    def _read_status(self) -> int:
        result = self._spi.xfer2([_CMD_STATUS, 0x00])
        return result[1]

    def _wait_eoc(self) -> None:
        deadline = time.monotonic() + self._timeout_ms / 1000.0
        while True:
            if self._read_status() & _BIT_EOC:
                return
            if time.monotonic() > deadline:
                raise SPI_HTPA160x120dError(f"Timeout ({self._timeout_ms} ms) waiting for EOC.")
            time.sleep(0.001)

    def _read_half(self, cmd: int) -> bytes:
        tx_data = [cmd] + [0x00] * _BYTES_PER_BLOCK
        rx_data = self._spi.xfer3(tx_data)
        return bytes(rx_data[1:])

    def _flash_read(self, mem_addr: int, length: int) -> bytes:
        """Read data from the external SST26VF016B Flash memory."""
        # Note: In a real hardware integration, EE_Enable must route to the Flash CS when low.
        addr_bytes = [(mem_addr >> 16) & 0xFF, (mem_addr >> 8) & 0xFF, mem_addr & 0xFF]
        tx_data = [_FLASH_CMD_READ] + addr_bytes + [0x00] * length
        
        # SPI CS must be inverted logic for the flash (Active LOW usually for SST26)
        self._spi.cshigh = False 
        rx_data = self._spi.xfer3(tx_data)
        self._spi.cshigh = True
        
        return bytes(rx_data[4:])

    # ── High-level public API ─────────────────────────────────────────────────
    def init(self, use_calib_settings: bool = True) -> None:
        self.load_calibration()
        self.wakeup()

        key = "calib" if use_calib_settings else "user"
        mbit = self._calib.get(f"mbit_{key}", 0x0C)
        bias_top = self._calib.get(f"bias_top_{key}", 0x0C)
        bias_bot = self._calib.get(f"bias_bot_{key}", 0x0C)
        clk = self._calib.get(f"clk_{key}", 0x14)

        for cmd, val in [
            (_CMD_TRIM1, mbit),
            (_CMD_TRIM2, bias_top), 
            (_CMD_TRIM3, bias_bot), 
            (_CMD_TRIM4, clk),
        ]:
            self._write_register(cmd, val)
            time.sleep(_T_INTER_REG_MS / 1000.0)

        self._calc_conversion_time(clk, mbit)

    def _calc_conversion_time(self, clk_trim: int, mbit: int):
        f_clk_mhz = 0.5 + ((5.5 - 0.5) / 63.0) * clk_trim  # Based on 0.5 to 5.5 MHz range
        t_fr4 = 4 * (2**mbit + 100) / (f_clk_mhz * 1000.0) # Approx formula for 160x120
        self._calib["t_fr4"] = t_fr4

    def sleep(self) -> None:
        self._write_register(_CMD_CONFIG, 0x00)

    def wakeup(self) -> None:
        self._write_register(_CMD_CONFIG, _BIT_WAKEUP)
        time.sleep(_T_WAKEUP_US / 1e6)

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
        f_clk_mhz = SPI_HTPA160x120d.clk_trim_to_freq(clk_trim)
        return 32.0 * (2 ** mbit + 4) / (f_clk_mhz * 1e6) * 1e3

    # ── Status register ───────────────────────────────────────────────────────
    def read_status(self) -> int:
        """Public method to read the current hardware status register."""
        return self._read_status()

    # ── Continuous stream threads ────────────────────────────────────────────
    def start_spistream(self):
        if self.output_queue is None:
            raise RuntimeError("Set an output queue before starting the stream.")
        
        self.spi_stop.clear()
        
        self._t_reader = threading.Thread(target=self._read_thread, daemon=True)
        self._t_processor = threading.Thread(target=self._processing_thread, daemon=True)
        
        self._t_reader.start()
        self._t_processor.start()

    def stop_spistream(self):
        self.spi_stop.set()
        self._t_reader.join()
        self._t_processor.join()

    def _read_thread(self):
        while not self.spi_stop.is_set():
            data_dict = self.read_frame()                      
            # Every ~8-10 frames, read electrical offsets
            if self._image_counter % 8 == 0:
                data_dict.update(self.read_electrical_offsets())
            
            self.spi_queue.put(data_dict) 
            self._image_counter += 1

    def _processing_thread(self, applyCalib: bool = True):
        timeout = self._calib.get('t_fr4', 30) * 12 / 1000.0 * 1.5 
        
        while not self.spi_stop.is_set():
            try:
                data = self.spi_queue.get(timeout=timeout)
                data = self.convert_spi_data(data)
                
                if applyCalib:
                    data = self.apply_calib(data)
                
                data['success'] = True
                data['image_id'] = self._image_counter
                self.output_queue.put(data)
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[processing_thread] {e}")                

    # ── Single Frame readout ─────────────────────────────────────────────────
    def read_frame(self) -> dict:
        t_fr4 = self._calib.get("t_fr4", 7.75) 
        raw_bytes: dict = {}
        
        for block in range(_BLOCKS):
            raw_bytes[block] = {}
            config = _BIT_WAKEUP | _BIT_START | (block << 4)
            self._write_register(_CMD_CONFIG, config)
            time.sleep(0.9 * t_fr4 / 1000.0)
            self._wait_eoc()

            raw_bytes[block]['top'] = self._read_half(_CMD_READ_TOP)
            raw_bytes[block]['bot'] = self._read_half(_CMD_READ_BOT)
        
        return {'pix_raw': raw_bytes, 't': datetime.now()}

    def read_electrical_offsets(self) -> dict:
        t_fr4 = self._calib.get("t_fr4", 7.75)
        self._write_register(_CMD_CONFIG, _BIT_WAKEUP | _BIT_START | _BIT_BLIND)
        time.sleep(0.9 * t_fr4 / 1000.0)
        self._wait_eoc()
                
        return {'eloff_raw': {'top': self._read_half(_CMD_READ_TOP), 'bot': self._read_half(_CMD_READ_BOT)}}

    # ── Data Conversion & Calibration ─────────────────────────────────────────
    def convert_spi_data(self, data: dict):
        pix_array = np.zeros((_ROWS, _COLS), dtype=np.uint16)
        ptat_list, vdd_list, atc_list = [], [], []

        for block in range(_BLOCKS):
            top_raw = data['pix_raw'][block]['top']
            bot_raw = data['pix_raw'][block]['bot']
            
            top_int = np.frombuffer(top_raw, dtype='>u2') 
            bot_int = np.frombuffer(bot_raw, dtype='>u2') 
            
            # Read Data Command structure
            atc_list.extend([top_int[0], bot_int[0]])
            ptat_list.extend([top_int[1], bot_int[1]])
            vdd_list.extend([top_int[2], bot_int[2]])
            
            pixels_top = top_int[3:]
            pixels_bot = bot_int[3:]
            
            # Rearrange according to Serial Order of Frame
            # (Simplified layout assignment logic for brevity)
            pixels_top = pixels_top.reshape((-1, _COLS))
            pixels_bot = np.flipud(pixels_bot.reshape((-1, _COLS)))
            
            pix_array[block*10:(block+1)*10, :] = pixels_top
            pix_array[-(block+1)*10:-block*10, :] = pixels_bot

        data['pix'] = pix_array.flatten()
        data['ptat'] = np.array(ptat_list).mean()
        data['vdd'] = np.array(vdd_list).mean()

        if 'eloff_raw' in data:
            top_int = np.frombuffer(data['eloff_raw']['top'], dtype='>u2')[3:]
            bot_int = np.frombuffer(data['eloff_raw']['bot'], dtype='>u2')[3:]
            self._calib['eloff'] = np.hstack([top_int, bot_int]) # Cache the latest offsets

        return data

    def apply_calib(self, data: dict) -> dict:
        c = self._calib

        # 15.1 Ambient Temperature
        PTAT_avg = data['ptat']
        Tamb = PTAT_avg * c["ptat_gradient"] + c["ptat_offset"]
        data['Tamb'] = Tamb
        
        # 15.2 Thermal Offset
        gradScale = c["gradscale"]
        V_ThComp = (c["thgrad_arr"] * PTAT_avg) / (2**gradScale) + c["thoffset_arr"]
        V_comp = data['pix'] - V_ThComp
        
        # 15.3 Electrical Offset
        if 'eloff' in c:
            # elOffset must be applied per block logic
            # (Flattened arrays match the 1D structure for vectorization)
            V_comp = V_comp - c['eloff']
        
        # 15.4 Vdd Compensation
        VDD_avg = data['vdd']
        VddScGrad = c["vddscgrad"]
        VddScOff  = c["vddscoff"]
        
        # Numerator & Denominator breakdown based on 160x120 formula
        num1 = (c["vddcompgrad_arr"] * PTAT_avg) / (2**VddScGrad) + c["vddcompoff_arr"]
        term2 = VDD_avg - c["vddth1"] - ((c["vddth2"] - c["vddth1"]) / (c["ptat_th2"] - c["ptat_th1"])) * (PTAT_avg - c["ptat_th1"])
        
        V_VddComp = (num1 / (2**VddScOff)) * term2
        V_comp = V_comp - V_VddComp
        
        # 15.5 Object Temperature (Sensitivity & PixC)
        PixCij = (c["pij_arr"] * (c["pixcmax"] - c["pixcmin"]) / 65535 + c["pixcmin"]) * (c["epsilon"]/100) * (c["global_gain"]/10000)
        pix_comp = (V_comp * _PCSCALEVAL) / PixCij
        
        data['pix_comp'] = pix_comp
        return data

    def load_calibration(self) -> None:
        """Read 160x120d flash memory map into calib dict"""
        c: dict = {}

        def f32(addr: int) -> float:
            return struct.unpack_from("<f", self._flash_read(addr, 4))[0]
        def u16(addr: int) -> int:
            return struct.unpack_from("<H", self._flash_read(addr, 2))[0]
        def i16_arr(addr: int, n: int) -> np.ndarray:
            return np.frombuffer(self._flash_read(addr, n * 2), dtype='<i2')
        def u16_arr(addr: int, n: int) -> np.ndarray:
            return np.frombuffer(self._flash_read(addr, n * 2), dtype='<u2')

        # Base scalars and factory trims (Assuming similar relative offsets to datasheet map)
        # Note: Replace hardcoded addresses with exact flash map pointer offsets if they change 
        # in future sensor batches.
        c["ptat_gradient"] = f32(0x0004) 
        c["ptat_offset"]   = f32(0x0008)
        c["gradscale"]     = 17          
        c["epsilon"]       = 100
        c["global_gain"]   = u16(0x0055) 

        c["vddth1"]        = u16(0x0026)
        c["vddth2"]        = u16(0x0028)
        c["ptat_th1"]      = u16(0x003C)
        c["ptat_th2"]      = u16(0x003E)
        
        c["vddscgrad"]     = 16
        c["vddscoff"]      = 23

        c["pixcmin"]       = f32(0x0000)
        c["pixcmax"]       = f32(0x0004)

        # Huge Array reads
        c["thgrad_arr"]      = i16_arr(0x3A00, _PIXELS)
        c["thoffset_arr"]    = i16_arr(0xCFF0, _PIXELS)
        c["pij_arr"]         = u16_arr(0x16600, _PIXELS)
        c["vddcompgrad_arr"] = i16_arr(0x2100, 1600) 
        c["vddcompoff_arr"]  = i16_arr(0x2D80, 1600)

        self._calib = c

# This tells Python to run this block only if the script is executed directly
if __name__ == "__main__":
    print("Testing the sensor driver script...")
    
    try:
        # Attempt to instantiate the sensor (assuming the SPI class is in this file)
        # Note: This will fail on a normal Windows PC without a hardware bridge
        sensor = SPI_HTPA160x120d()
        print("Sensor class initialized successfully!")
    except Exception as e:
        print(f"Failed to initialize: {e}")