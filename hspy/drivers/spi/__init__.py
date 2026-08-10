import sys
import warnings

if sys.platform == 'win32':
    warnings.warn('SPI drivers (spidev) are not available natively on Windows')

if sys.platform == 'linux':
    from .drivers import SPI_Driver
    from .drivers import SPI_HTPA160x120d