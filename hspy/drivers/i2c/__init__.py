import sys
import warnings

if sys.platform == 'win32':
    pass
    # warnings.warn('I2C drivers not available on windows')

if sys.platform == 'linux':
    from .drivers import I2C_Driver
    from .drivers import I2C_HTPA32x32d