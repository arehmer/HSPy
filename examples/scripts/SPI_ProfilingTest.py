import time
from hspy.drivers.spi import SPI_HTPA160x120dR1

# gpiozero is only defined fo Raspberry Pis
try:
    from gpiozero import DigitalOutputDevice
    from gpiozero.exc import BadPinFactory
    _GPIOZERO_AVAILABLE = True
except ImportError:
    _GPIOZERO_AVAILABLE = False

# ... construct spi_driver exactly as in the example ...
# Returns a callable that drives the GPIO pin to enable/disable the EEPROM on the HTPA board
def make_ee_enable(pin: int = 8):
    """
    Return an ee_enable(level) callable for SPI_HTPA160x120dR1, driving
    BCM GPIO `pin`. Raises RuntimeError with a clear message on machines
    without gpiozero or without GPIO hardware.
    """
    if not _GPIOZERO_AVAILABLE:
        raise RuntimeError("gpiozero is not installed. Run: pip install gpiozero")
    try:
        ee = DigitalOutputDevice(pin, initial_value=True)   # idle HIGH
    except BadPinFactory as e:
        raise RuntimeError(
            "No GPIO hardware found: gpiozero only drives pins on a "
            "Raspberry Pi (or with a mock pin factory for testing)."
        ) from e

    def ee_enable(level: bool) -> None:
        ee.value = level

    ee_enable.device = ee        # keep a handle, e.g. for ee_enable.device.close()
    return ee_enable

spi_driver = SPI_HTPA160x120dR1(bus = 0,
                                device = 0,
                                ee_enable = make_ee_enable(pin=8))


d = spi_driver
print('t_block est. [ms]:', d.calib['t_block'], ' F_CLK [MHz]:', d.calib['F_CLK'])

N = 1000
t = time.perf_counter()
for _ in range(N): d._ee_enable(True)
print('EE_Enable write [us]:', (time.perf_counter() - t) / N * 1e6)

t = time.perf_counter()
for _ in range(N): d.read_status()
print('status poll [us]:', (time.perf_counter() - t) / N * 1e6)

t = time.perf_counter()
for _ in range(100): d._read_both_halves(d._io.xf_frame[0])
print('read both halves [ms]:', (time.perf_counter() - t) / 100 * 1e3)

# Actual conversion time vs. the estimate
for _ in range(5):
    d._write_register(0x01, 0x09)          # WAKEUP | START, block 0
    t0 = time.perf_counter(); n = 0
    while not d.read_status() & 1: n += 1
    print(f'conversion [ms]: {(time.perf_counter()-t0)*1e3:.2f}, polls: {n}')