import os
import warnings

if os.name == 'nt':
    warnings.warn('I2C drivers not available on windows')

if os.name == 'posix':
    from .drivers import I2C_Driver
    from .drivers import I2C_HTPA32x32d