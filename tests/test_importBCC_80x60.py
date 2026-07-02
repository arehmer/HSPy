# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 09:07:54 2024
@author: Rehmer
"""
from pathlib import Path
from hspy.tparray import TPArray, ArrayTypes

# %% Specify path
bcc = Path.cwd() / 'BCCs' / '80x60.BCC'

# %% Create TPArray and import BCC
tparray = TPArray(ArrayType=ArrayTypes['HTPA80x60d'])
BCC = tparray.import_BCC(bcc)

print('calibVersion:', tparray.calibVersion)

# %% Inspect which calibration fields are actually available
print('BCC keys:', list(BCC.keys()))

# Check whether the top/bot fields used by _calc_Tamb0_CalibV4 exist
for key in ['ptatGrad_top', 'ptatOffset_top', 'ptatGrad_bot', 'ptatOffset_bot']:
    print(f'  {key:18s} present:', key in BCC)

# %% Verify the actual values against the datasheet table
# Expected: grad_top ~0.026852544 | off_top ~2047.300537
#           grad_bot ~0.02742737  | off_bot ~2107.665039
for key in ['ptatGrad_top', 'ptatOffset_top', 'ptatGrad_bot', 'ptatOffset_bot']:
    if key in BCC:
        print(f'  {key:18s} =', BCC[key])