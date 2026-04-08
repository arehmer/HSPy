import os
import warnings

if os.name == 'nt':
    warnings.warn('I2C drivers not available on windows')

if os.name == 'posix':
    from .htpa32x32d import HTPA32x32d