# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 15:56:46 2024

@author: rehmer
"""

import matplotlib.pyplot as plt
from pathlib import Path


from hspytools.readers import HTPAdGUI_FileReader
from hspytools.tparray import TPArray, SensorTypes

from hspytools.LuT import LuT




# %% path to file as Path object

# %% All implemented sensor types are in the dictionary SensorTypes
print(SensorTypes.keys())

#%% Initialize a TPArray instance providing the proper Sensor Type
tparray = TPArray(SensorType = SensorTypes['HTPA60x40D_L1K9_0K8'])
# tparray = TPArray(width = 60, height = 40)


# %% Load the raw data you want to convert from raw digits to dK using the
# HTPAdGUI_FileReader
file_path = Path.cwd() / 'data_samples' / '60x40L1k9_Ta25_To100_DevID1172.TXT'

reader = HTPAdGUI_FileReader(tparray)
raw_data,header = reader.read_htpa_video(file_path)

#%% Use only the first 100 rows to reduce processing time of this example 
raw_data = raw_data.loc[0:10]
# %% Plot any frame from the raw data
frame10_raw = raw_data.loc[10,tparray._pix]
frame10_raw = frame10_raw.values.reshape(tparray._npsize)

plt.figure('Frame 10 raw digits')
plt.imshow(frame10_raw)


# %% Import a BCC File to the TPArray instance
bcc_file = Path.cwd() / 'data_samples' / 'DevID1172.BCC'
tparray.import_BCC(bcc_file)

# %% Use the tparray class to apply calibration to raw_data
comp_data = tparray.rawmeas_comp(raw_data)

# %% Plot the compensated frame
frame10_comp = comp_data.loc[10,tparray._pix]
frame10_comp = frame10_comp.values.reshape(tparray._npsize)

plt.figure('Frame 10 compensated digits')
plt.imshow(frame10_comp)

# %% Next initialize a LuT object
LuT150 = LuT()
try:
    LuT150.from_HTPAxls('60x40dL1k9')
except Exception as e:
    print(f'Seems as you are not within Heimann Sensor premises! Exception: {e}')

# %% Import the LuT to the TPArray object
tparray.import_LuT(LuT150)

# %% Use 
dK_data = tparray.rawmeas_to_dK(raw_data)

# %% Plot the dK frame
frame10_dK = dK_data.loc[10,tparray._pix]
frame10_dK = frame10_dK.values.reshape(tparray._npsize)

plt.figure('Frame 10 dK')
plt.imshow(frame10_dK)

#%% Für Dennis
dK_data.to_excel('ConvertedData.xlsx')
