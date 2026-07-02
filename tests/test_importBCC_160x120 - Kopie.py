# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 09:07:54 2024

@author: Rehmer
"""


from pathlib import Path

from hspytools.tparray import TPArray
import matplotlib.pyplot as plt

# %% Specify paths
bcc_31 = Path.cwd() / 'BCCs'  / '60x40_L1k9_31.BCC'



# %% 

tparray = TPArray(60,40)

BCC = tparray.import_BCC(bcc_31)

# %%

plt.imshow(BCC['pij'])
