# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 15:57:00 2026

@author: rehmer
"""
import pickle as pkl
import matplotlib.pyplot as plt


frame = pkl.load(open('raspi_frame.pkl','rb'))


frame_calib = frame['pixels'] - frame['V_ElComp'] - frame['V_ThComp'] - frame['V_VddComp']

plt.imshow(frame_calib.reshape((32,32)))