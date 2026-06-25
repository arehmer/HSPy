# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 11:16:48 2023

@author: Rehmer
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import json
from pathlib import Path
import struct
import time

import warnings

from dataclasses import dataclass, fields
from collections.abc import Sequence, Iterator
from typing import Dict, List, Mapping, Any


from .LuT import LuT

import warnings

# This needs to be an exact copy of the enum from TPArray.hpp
SensorTypes = {'HTPA60x40D_L1K9_0K8':0,
               'HTPA120x84DR2_L3K95_0K8':1,
               'HTPA160x120DR1_L3K95_0K8':2,
               'HTPA8x8DR1_L0K8_0K8':3,
               'HTPA32x32dR2_L1k9_0k8':4,
               'HTPA80x64dR2_L10k5_0k95_F7k7':5,
               'HTPA32x32dR2_L1k7_0k8':6,
               'HTPA32x32dR2_L1k7_0k8_THiC_Si':7,
               'SENSOR_TYPE_NONE' : 99}

ArrayTypes = {'HTPA8x8' : 0,
              'HTPA16x16' : 1,
              'HTPA32x16' : 2,
              'HTPA32x31' : 3,
              'Zeile64' : 4,
              'HTPA64x62' : 5,
              'HTPA16x4' : 6,
              'HID' : 7,
              'HTPA106x52' : 8,
              'HTPA82x62' : 9,
              'HTPA32x32d' : 10,
               'HTPA32x32dR2' : 10, 
              'HTPA80x64d'	: 11,
              'HTPA120x84d' : 12,
              'HTPA84x60d' :	13,
              'HTPA60x40d'	: 14,
              'HTPA160x120d' :	15,
              'HTPA120x84dR2' : 16,
              'HTPA16x16dR3' : 17,
              'HTPA160x120dR1' : 18,
              'HTPA80x60d' : 19,
              'HTPA60x40dR2' : 20,
              'HTPA50x50d' : 21}

class TPArray():
    """
    Class contains hard-coded properties of Thermopile-Arrays relevant
    for reading from Bytestream
    """
    
    def __init__(self,**attr_dict):
        
        self._SensorType = attr_dict.pop('SensorType',None)
        self._ArrayType = attr_dict.pop('ArrayType',None)
        
        width = attr_dict.pop('w',None)
        height = attr_dict.pop('h',None)
        
        self.DevConst = {}
        
        if self._SensorType is not None:
            self._init_by_SensorType()
            self._init_by_ArrayType()
        elif self._ArrayType is not None:
            self._init_by_ArrayType()
        elif width is not None and height is not None:
            self._init_by_Resolution(width,height)
            self._init_by_ArrayType()
        else:
            raise Exception('Provide either SensorType or ArrayType or Resolution!')
        
        # Init basic attributes and DevConst by resolution
        self._init_DevConst()
        
        # Init reamining properties, otherwise save() method will fail
        self.BCC = attr_dict.pop('BCC',None)

    @property
    def SensorType(self):
        return self._SensorType
    @SensorType.setter
    def SensorType(self,sensorType:int):
        self._SensorType = sensorType
        
    @property
    def ArrayType(self):
        return self._ArrayType
    @ArrayType.setter
    def ArrayType(self,ArrayType:int):
        self._ArrayType = ArrayType
        
    @property
    def DesignGen(self):
        return self._DesignGen
    @DesignGen.setter
    def DesignGen(self,DesignGen:int):
        self._DesignGen = DesignGen
        
    @property
    def DevConst(self):
        return self._DevConst
    @DevConst.setter
    def DevConst(self,DevConst:dict):
        self._DevConst = DevConst
        
    @property
    def DataCols(self):
        return self._DataCols
    @DataCols.setter
    def DataCols(self,DataCols:DataCols):
        self._DataCols = DataCols
        
    @property
    def width(self):
        return self._width
    @width.setter
    def width(self,w):
        self._width = w
    
    @property
    def height(self):
        return self._height
    @height.setter
    def height(self,h):
        self._height = h
        
    @property
    def BCC(self):
        return self._BCC
    @BCC.setter
    def BCC(self,BCC):
        self._BCC = BCC
        
    # ------ Compatibility attributes. To be removed in future releases ------
    @property
    def _pix(self):
        warnings.warn(
        "TPArray._pix is deprecated and will be removed. "
        "Use TPArray.DataCols.pix instead.",
        DeprecationWarning,
        stacklevel=2)
        return self.DataCols.pix
    @property
    def _e_off(self):
        warnings.warn(
        "TPArray._e_off is deprecated and will be removed. "
        "Use TPArray.DataCols.e_off instead.",
        DeprecationWarning,
        stacklevel=2)
        return self.DataCols.e_off
    @property
    def _vdd(self):
        warnings.warn(
        "TPArray._vdd is deprecated and will be removed. "
        "Use TPArray.DataCols.vdd instead.",
        DeprecationWarning,
        stacklevel=2)
        return self.DataCols.vdd    
    @property
    def _T_amb(self):
        warnings.warn(
        "TPArray._T_amb is deprecated and will be removed. "
        "Use TPArray.DataCols.T_amb instead.",
        DeprecationWarning,
        stacklevel=2)
        return self.DataCols.T_amb       
    @property
    def _PTAT(self):
        warnings.warn(
        "TPArray._PTAT is deprecated and will be removed. "
        "Use TPArray.DataCols.PTAT instead.",
        DeprecationWarning,
        stacklevel=2)
        return self.DataCols.PTAT
    @property
    def _ATC(self):
        warnings.warn(
        "TPArray.ATC is deprecated and will be removed. "
        "Use TPArray.DataCols.ATC instead.",
        DeprecationWarning,
        stacklevel=2)
        return self.DataCols.ATC
    @property
    def _serial_data_order(self):
        warnings.warn(
        "TPArray._serial_data_order is deprecated and will be removed. "
        "Use TPArray.DataCols.all() instead.",
        DeprecationWarning,
        stacklevel=2)
        return self.DataCols.all()    

    # -------------------------------------------------------------------------

    def _init_by_SensorType(self):
        
        if (self.SensorType == SensorTypes['HTPA60x40D_L1K9_0K8']):
            self.ArrayType = ArrayTypes['HTPA60x40d']
        elif (self.SensorType == SensorTypes['HTPA120x84DR2_L3K95_0K8']):
            self.ArrayType = ArrayTypes['HTPA120x84dR2']
        elif (self.SensorType == SensorTypes['HTPA160x120DR1_L3K95_0K8']):
            self.ArrayType = ArrayTypes['HTPA160x120dR1']
        elif (self.SensorType == SensorTypes['HTPA8x8DR1_L0K8_0K8']):
            self.ArrayType = ArrayTypes['HTPA8x8']
        elif (self.SensorType == SensorTypes['HTPA32x32dR2_L1k9_0k8']):
            self.ArrayType = ArrayTypes['HTPA32x32d']
            self._NETD = 60 # Tuned on Archesens-Data
        elif (self.SensorType == SensorTypes['HTPA32x32dR2_L1k7_0k8']):
            self.ArrayType = ArrayTypes['HTPA32x32dR2']
            self._NETD = 152 # from datasheet   
            self.FocalLength = 1.7  # mm
        elif (self.SensorType == SensorTypes['HTPA32x32dR2_L1k7_0k8_THiC_Si']):
            self.ArrayType = ArrayTypes['HTPA32x32dR2']
        elif self.SensorType is None:
            self.ArrayType = None
        else:
            raise NotImplementedError('SensorType not implemented or not known!')
            
    def _init_by_ArrayType(self):
        
        if (self.ArrayType == ArrayTypes['HTPA8x8']):
            self.width = 8
            self.height = 8
            self.DesignGen = 0
            self.UDP_PackageIndex = 0
        elif (self.ArrayType == ArrayTypes['HTPA16x16']):
            self.width = 16
            self.height = 16
            self.DesignGen = 0
            self.UDP_PackageIndex = 0
        elif (self.ArrayType == ArrayTypes['HTPA32x32d']):
            self.width = 32
            self.height = 32
            self.pixpitch = 90 # µm
            self.DesignGen = 0
            self.UDP_PackageIndex = 0
        elif (self.ArrayType == ArrayTypes['HTPA80x64d']):
            self.width = 80
            self.height = 64
            self.DesignGen = 0
            self.UDP_PackageIndex = 1
        elif (self.ArrayType == ArrayTypes['HTPA120x84d']):
            self.width = 120
            self.height = 84
            self.DesignGen = 0
            self.UDP_PackageIndex = 1
        elif (self.ArrayType == ArrayTypes['HTPA84x60d']):
            self.width = 84
            self.height = 60
            self.DesignGen = 1
            self.UDP_PackageIndex = 1
        elif (self.ArrayType == ArrayTypes['HTPA60x40d']):
            self.width = 60
            self.height = 40
            self.DesignGen = 1
            self.UDP_PackageIndex = 1
        elif (self.ArrayType == ArrayTypes['HTPA160x120d']):
            self.width = 160
            self.height = 120
            self.DesignGen = 1
            self.UDP_PackageIndex = 1
        elif (self.ArrayType == ArrayTypes['HTPA120x84dR2']):
            self.width = 120
            self.height = 84
            self.DesignGen = 1
            self.UDP_PackageIndex = 1
        elif (self.ArrayType == ArrayTypes['HTPA16x16dR3']):
            self.width = 16
            self.height = 16
            self.DesignGen = 2
            self.UDP_PackageIndex = 0
        elif (self.ArrayType == ArrayTypes['HTPA160x120dR1']):
            self.width = 160
            self.height = 120
            self.DesignGen = 1
            self.UDP_PackageIndex = 1
        elif (self.ArrayType == ArrayTypes['HTPA80x60d']):
            self.width = 80
            self.height = 60
            self.DesignGen = 3
            self.UDP_PackageIndex = 1
        elif (self.ArrayType == ArrayTypes['HTPA60x40dR2']):
            self.width = 60
            self.height = 40 
            self.DesignGen = 1
            self.UDP_PackageIndex = 1
        elif (self.ArrayType == ArrayTypes['HTPA50x50d']):
            self.width = 50
            self.height = 50      
            self.DesignGen = 4
            self.UDP_PackageIndex = 1
        elif self.ArrayType is None:
            self.width = None
            self.height = None
        else:
            raise NotImplementedError('ArrayType not implemented or not known!')

    def _init_by_Resolution(self,width,height):
        
        if width == 8 and height == 8:
            self.ArrayType = ArrayTypes['HTPA8x8']
        elif width == 16 and height == 16:
            self.ArrayType = ArrayTypes['HTPA16x16dR3']
        elif width == 32 and height == 32:
            self.ArrayType = ArrayTypes['HTPA32x32d']
        elif width == 80 and height == 64:
            self.ArrayType = ArrayTypes['HTPA80x64d']            
        elif width == 120 and height == 84:
            self.ArrayType = ArrayTypes['HTPA120x84dR2']
        elif width == 84 and height == 60:
            self.ArrayType = ArrayTypes['HTPA84x60d']
        elif width == 60 and height == 40:
            self.ArrayType = ArrayTypes['HTPA60x40dR2']
        elif width == 160 and height == 120:
            self.ArrayType = ArrayTypes['HTPA160x120dR1']
        elif width == 80 and height == 60:
            self.ArrayType = ArrayTypes['HTPA80x60d']
        elif width == 50 and height == 50:
            self.ArrayType = ArrayTypes['HTPA50x50d']
        else:
            raise NotImplementedError(f'No Array with w x h {width}x{height} known!')

    def _init_DevConst(self):
        
        self._size = (self.width,self.height)
        self._npsize = (self.height,self.width)
        
        
        if self.ArrayType == ArrayTypes['HTPA8x8']:
            self.DevConst['NROFATC']=0
            self.DevConst['NROFBLOCKS']=1
            self.DevConst['NROFPTAT']=1
            
            self._package_num = 1
            self._package_size = 262
            self._fs = 160
            self._NETD = 100
            self.Pitch = 90.0e-6
            self.Ampl = 40
            
            self._mask = np.ones(self._npsize)

            # path to array data
            path = Path(__file__).parent / 'arraytypes' / '8x8.json'
            # Load calibration data from file
            self._load_calib_json(path)  
            
        elif self.ArrayType in [ArrayTypes['HTPA16x16'],
                                ArrayTypes['HTPA16x16dR3']]:
            self.DevConst['NROFATC']=2
            self.DevConst['NROFBLOCKS']=2
            self.DevConst['NROFPTAT']=2
            
            self._package_num = 1
            self._package_size = 780
            self._fs = 70
            self._NETD = 130
            self.Pitch = 90.0e-6    #equal for r3
            self.Ampl = 40          #equal for r3  
            
            self._mask = np.ones(self._npsize)

            # path to array data
            path = Path(__file__).parent / 'arraytypes' / '16x16.json'
            # Load calibration data from file
            self._load_calib_json(path)  
        
        elif self.ArrayType == ArrayTypes['HTPA32x32d']:
            self.DevConst['NROFATC']=0
            self.DevConst['NROFBLOCKS']=4
            self.DevConst['NROFPTAT']=2

            self._package_num = 2
            self._package_size = 1292
            self._fs = 27
            self.Pitch = 90.0e-6
            self.Ampl = 40            
            
            # The mask is generated by considering the radiometric radius
            self._radio_r = 14
            self._mask = self._binary_mask(self._radio_r) 
            
            # path to array data
            path = Path(__file__).parent / 'arraytypes' / '32x32.json'
            # Load calibration data from file
            self._load_calib_json(path)  
            
        elif self.ArrayType == ArrayTypes['HTPA80x64d']:
            self.DevConst['NROFATC']=0
            self.DevConst['NROFBLOCKS']=4
            self.DevConst['NROFPTAT']=2

            self._package_num = 10
            self._package_size = 1283
            self._fs = 41
            self._NETD = 70
            
            self._mask = np.ones(self._npsize)
            
            # path to array data
            path = Path(__file__).parent / 'arraytypes' / '80x64.json'
            # Load calibration data from file
            self._load_calib_json(path)  
            
        elif self.ArrayType == ArrayTypes['HTPA84x60d']:
            self.DevConst['NROFBLOCKS']=7
            self.DevConst['NROFPTAT']=2
            self.DevConst['NROFATC']= 2

            self._package_num = 10
            self._package_size = 1283
            self._fs = 41
            self._NETD = 70
            self.Pitch = 60.0e-6
            self.Ampl = 60               
           
            self._mask = np.ones(self._npsize)
            
            # path to array data
            path = Path(__file__).parent / 'arraytypes' / '60x84.json'
            # Load calibration data from file
            self._load_calib_json(path)  
            
        elif self.ArrayType in [ArrayTypes['HTPA60x40d'],
                                ArrayTypes['HTPA60x40dR2']]:
            self.DevConst['NROFATC']=2
            self.DevConst['NROFBLOCKS']=5
            self.DevConst['NROFPTAT']=2

            self._package_num = 5
            self._package_size = 1159
            self._fs = 47
            self._NETD = 90
            self.Pitch = 45.0e-6
            self.Ampl = 60               
            
            self._mask = np.ones(self._npsize)
            
            # path to array data
            path = Path(__file__).parent / 'arraytypes' / '60x40.json'
            # Load calibration data from file
            self._load_calib_json(path)  

        elif self.ArrayType in [ArrayTypes['HTPA120x84d'],
                                ArrayTypes['HTPA120x84dR2']]:
            self.DevConst['NROFATC']=0
            self.DevConst['NROFBLOCKS']=6

            self._package_num = 17
            self._package_size = 1401
            self._fs = 20
            self._NETD = 130
            self.Pitch = 60.0e-6
            self.Ampl = 60               
            r_lim = 60-4
            
            self._mask = self._binary_mask(r_lim) 
            
            # path to array data
            path = Path(__file__).parent / 'arraytypes' / '120x84.json'
            # Load calibration data from file
            self._load_calib_json(path)
            
        elif self.ArrayType in [ArrayTypes['HTPA160x120d'],
                                ArrayTypes['HTPA160x120dR1']]:
            self.DevConst['NROFATC'] = 2
            self.DevConst['NROFBLOCKS'] = 12
            self.DevConst['NROFPTAT'] = 2

            self._package_num = 30
            self._package_size = 1401
            self._fs = 25
            self._NETD = 110
            self.Pitch = 45.0e-6        #equal for r1
            self.Ampl = 60              #40 for r1         
            
            self._mask = np.ones(self._npsize)
            
            # path to EEPROM map
            path = Path(__file__).parent / 'arraytypes' / '160x120.json'
            
            # Load calibration data from file
            self._load_calib_json(path)
            
        elif self.ArrayType == ArrayTypes['HTPA80x60d']:
            self.DevConst['NROFATC'] = 2
            self.DevConst['NROFBLOCKS'] = 6
            self.DevConst['NROFPTAT'] = 2
            
            warnings.warn('ArrayType not fully implemented!')
            
            # path to EEPROM map
            path = Path(__file__).parent / 'arraytypes' / '80x60.json'
            
            # Load calibration data from file
            self._load_calib_json(path)
            
            
        elif self.ArrayType == ArrayTypes['HTPA50x50d']:
            self.DevConst['NROFATC'] = 0
            self.DevConst['NROFPTAT'] = 1
            self.DevConst['NROFBLOCKS'] = 50
            
            
            warnings.warn("50x50.json is a copy of 32x32.json. Validate/correct in future!")
            path = Path(__file__).parent / 'arraytypes' / '50x50.json'
            self._load_calib_json(path)  

        else:
            raise Exception('This Thermopile Array is not known.') 
        
        # From the hard-coded constants, the remaining useful information can
        # be derived
        if self.DesignGen <= 3:
            self._init_DerivedConstants_3()
        elif self.DesignGen == 4:
            self._init_DerivedConstants_4()
        else:
            raise ValueError('DesignGen is not set or value not known.')
    
    def _init_DerivedConstants_3(self):
        
        # For convenience
        DevConst = self.DevConst
        
        # Remaining DevConst can be derived
        DevConst['VDDaddr'] = \
            int(self.width*self.height+self.height/DevConst['NROFBLOCKS']*self.width)   
            
        DevConst['TAaddr']=DevConst['VDDaddr'] + 1
        DevConst['PTaddr']=DevConst['TAaddr'] + 1
            
        self.DevConst = DevConst        
        self._rowsPerBlock = int(self.height/DevConst['NROFBLOCKS'] / 2)
        self._pixelPerBlock = int(self._rowsPerBlock * self.width)
        self._PCSCALEVAL = 100000000
        
        # Derive order of serial data from DevConst
        # pixels
        pix = ['pix'+str(p) for p in range(0,self.width*self.height)]
        
        # electrical offsets
        no_e_off = int(self.height/DevConst['NROFBLOCKS'] * self.width)
        e_off = ['e_off'+str(e) for e in range(0,no_e_off)]
        
        # voltage
        vdd = ['Vdd'+str(v) for v in range(0,
                                    DevConst['TAaddr']-DevConst['VDDaddr'])]
        
        # ambient temperature
        T_amb = ['Tamb'+str(t) for t in range(0,
                                      DevConst['PTaddr']-DevConst['TAaddr'])]
        
        # PTAT
        no_ptat = int(DevConst['NROFBLOCKS']*DevConst['NROFPTAT'])
        PTAT = ['PTAT'+str(t) for t in range(0,no_ptat)]
        
        # ATC            
        ATC = ['ATC'+str(a) for a in range(0,DevConst['NROFATC'])]
                
        # Store these headers in the class attribute DataCols
        self.DataCols = DataCols(pix=pix,
                                 e_off=e_off,
                                 vdd=vdd,
                                 T_amb=T_amb,
                                 PTAT=PTAT,
                                 ATC=ATC)

    def _init_DerivedConstants_4(self):
        
        # For convenience
        DevConst = self.DevConst
        #put here VDDadr, etc.
        self.DevConst = DevConst        
        self._rowsPerBlock = int(1)
        self._pixelPerBlock = int(self.width)
                
        
        # From DevConst, derive the column headers for .bds / .txt files 
        
        # Pixel headers
        pix = ['pix'+str(p) for p in range(0,self.width*self.height)]
        
        # Electrical Offset headers
        e_off = ['e_off'+str(e) for e in range(0,self.width)]
        
        # Vdd headers (there's only one)
        vdd = ['Vdd0']
        
        # Ambient temperature header (there's only one)
        T_amb = ['Tamb0']
        
        # PTAT header (there's only one)
        PTAT = ['PTAT0']
        
        # ATC header (none exists)
        ATC = []
        
        # Store these headers in the class attribute DataCols
        self.DataCols = DataCols(pix=pix,
                                 e_off=e_off,
                                 vdd=vdd,
                                 T_amb=T_amb,
                                 PTAT=PTAT,
                                 ATC=ATC)
                
    def _load_calib_json(self, path:Path):
        
        with open(path,'r') as file:
            eeprom_adresses = json.load(file)
        
        self._eeprom_adresses =  eeprom_adresses
   
    def get_DevConst(self):
        return self._DevConst
    
    def get_serial_data_order(self):
        return self._serial_data_order
        
    def get_eeprom_adresses(self):
        return self._eeprom_adresses
    
    def set_LuT(self,LuT):
        self._LuT = LuT
    
    def import_LuT(self,LuT:LuT):
       
        self._LuT = LuT
        
        return None    
        
    def import_BCC(self,bcc_path):
        """
        This is a copy of Read_BccData.py by CK
        

        Parameters
        ----------
        bcc_path : pathlib.Path()
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        # Shorthand for EEPROM Adresses
        ee = self.get_eeprom_adresses()['EEPROM']
        
        # Initialize empty dict for return results
        bcc = {}
        
        ########################################
        # get all relevant data from .bcc file #
        ########################################
        
        # read hex data in bit by bit
        bcc_raw = self._stable_read(bcc_path)
        
        # Read and convert data according to provided json file
        for key in ee.keys():

            # Ge start and stop indices from addresses
            idx_start = int(ee[key]['adr_start'],0)
            idx_stop = int(ee[key]['adr_stop'],0)+1
            
            # Get raw value
            raw_val = bcc_raw[idx_start:idx_stop]
                        
            # Convert raw value 
            bcc[key] = self._convert_raw_bcc(raw_val,ee[key]['dtype'])

        
        # Convert all EEPROM values from lists to numpy array
        for key in bcc.keys():
            bcc[key] = np.array(bcc[key]) 
            
            
        # Special case for 16x16 Arrays because of different EEPROM 
        # if self.ArrayType == ArrayTypes['HTPA16x16']
        
        # Derive calibration settings from raw values
        bcc = self._derive_calib_settings(bcc)


        # Convert all arrays to appropriate shape and flip them
        # properly
        bcc['pij'] = np.array(bcc['pij']).reshape(self._npsize)
        bcc['thGrad'] = np.array(bcc['thGrad']).reshape(self._npsize)
        bcc['thOff'] = np.array(bcc['thOff']).reshape(self._npsize)
        
        # Only 8x8 Arrays don't have vdd calibration data and pij, thGrad and 
        # thOff are not flipped
        if not (self.width,self.height) == (8,8):
            
            NROFBLOCKS = self.get_DevConst()['NROFBLOCKS']
            vdd_size = (int(self.height/NROFBLOCKS),self._width)
                        
            # The lower half needs to be flipped vertically
            bcc['pij'][int(self.height/2):,::] = \
                np.flipud(bcc['pij'][int(self.height/2):,::])
            
            bcc['thGrad'][int(self.height/2):,::] = \
                np.flipud(bcc['thGrad'][int(self.height/2):,::])
                
            bcc['thOff'][int(self.height/2):,::] = \
                np.flipud(bcc['thOff'][int(self.height/2):,::])
            
            bcc['vddCompGrad'] = np.array(bcc['vddCompGrad']).reshape(vdd_size)
            bcc['vddCompOff'] = np.array(bcc['vddCompOff']).reshape(vdd_size)
            
            bcc['vddCompGrad'][int(vdd_size[0]/2):,::] = \
                np.flipud(bcc['vddCompGrad'][int(vdd_size[0]/2):,::])
            
            bcc['vddCompOff'][int(vdd_size[0]/2):,::] = \
                np.flipud(bcc['vddCompOff'][int(vdd_size[0]/2):,::])
        
        self.BCC = bcc

        self._checkBCC(bcc)
        
        return bcc
    
    def _checkBCC(self,bcc):
        """
        Performs a sanity check on the imported BCC. Mostly standard values
        are checked for and a warning issued to the user if found

        Returns
        -------
        None.

        """
        # print('after: ' + str(bcc['pij'][0,0]))
        type_max = {'uint16':65535}
        
        # Check if pixel constants are on standard value
        pij = bcc['pij']
        
        if (pij == type_max['uint16']).all():
            warnings.warn('Pixel constants have not yet been set for this device!')

    def _stable_read(self, path, timeout=2.0, interval=0.15):
        # path = Path(path)
        end = time.monotonic() + timeout
        prev = None
        while True:
            with open(path, "rb") as f:
                data1 = f.read()
                
            time.sleep(interval)  # tiny pause
            
            with open(path, "rb") as f:
                data2 = f.read()
                                
            if data1 == data2:           # stable content, not just stable size
                return data2
    
            if time.monotonic() >= end:
                # last attempt; surface the issue
                raise TimeoutError("File content didn't stabilize within timeout.")

    def _convert_raw_bcc(self,raw_val:list,dtype:str):
        """
        Link to documentation of struct lybrary
        https://docs.python.org/3/library/struct.html#struct-format-strings
        """
        
        if dtype == 'float16':
            b_idx = np.arange(0,len(raw_val),2)
            conv_val = [struct.unpack('e',raw_val[b:b+2])[0] for b in  b_idx]
        
        if dtype == 'float32':
            b_idx = np.arange(0,len(raw_val),4)
            conv_val = [struct.unpack('f',raw_val[b:b+4])[0] for b in  b_idx]
            
        elif dtype == 'uint8':            
            b_idx = np.arange(0,len(raw_val),1)
            conv_val = [struct.unpack('B',raw_val[b:b+1])[0] for b in  b_idx] 
            
        elif dtype == 'uint16':
            b_idx = np.arange(0,len(raw_val),2)
            conv_val = [struct.unpack('<H',raw_val[b:b+2])[0] for b in  b_idx] 

        elif dtype == 'int8':
            b_idx = np.arange(0,len(raw_val),1)
            conv_val = [struct.unpack('b',raw_val[b:b+1])[0] for b in  b_idx] 

        elif dtype == 'int12':
            b_idx = np.arange(0,len(raw_val),1)
            conv_val = self._extract_signed_12bit(raw_val)

        elif dtype == 'int16':
            b_idx = np.arange(0,len(raw_val),2)
            conv_val = [struct.unpack('<h',raw_val[b:b+2])[0] for b in  b_idx] 
            
        elif dtype == 'uint32':
            b_idx = np.arange(0, len(raw_val), 4)
            conv_val = [struct.unpack('<I', raw_val[b:b+4])[0] for b in b_idx]
            
        else:
            Exception('Unknown datatype')
            conv_val = None
        
        return conv_val
    
    def _derive_calib_settings(self,bcc:dict) -> dict:
        
        if 'MBIT(calib)' in bcc.keys():
            
            # Extract int from bcc
            MBIT_calib = bcc['MBIT(calib)'].item()
            
            if MBIT_calib != 0 and MBIT_calib != 255:
    
                # Shift right 4 positions and mask to extract REFCAL_calib
                REFCAL_calib = (MBIT_calib>>4) & 0b11
            
            else:
                REFCAL_calib = np.nan
                
        if 'MBIT(user)' in bcc.keys():
            
            # Extract int from bcc
            MBIT_user = bcc['MBIT(user)'].item()
            
            if MBIT_user != 0 and MBIT_user != 255:
    
                # Shift right 4 positions and mask to extract REFCAL_user
                REFCAL_user = (MBIT_user>>4) & 0b11
            
            else:
                REFCAL_user = np.nan        
                
            # bcc['REFCAL(user)'] = REFCAL_user

        return bcc

    def _extract_signed_12bit(self,raw_val):
        conv_val = []
        
        # Iterate over bytes in steps of 1.5 bytes (2 bytes per 12-bit pair)
        for i in range(0, len(raw_val) - 1, 3):  
            
            if i + 2 >= len(raw_val):
                break  # Ensure we have a full 3-byte set
            
            b1, b2, b3 = raw_val[i], raw_val[i+1], raw_val[i+2]
    
            # Reconstruct two 12-bit values
            val1 = ((b2 & 0x0F) << 8) | b1  # First 12-bit value (lower byte first)
            val2 = (b3 << 4) | (b2 >> 4)    # Second 12-bit value    
            
            # Convert to signed 12-bit integers
            val1 = val1 - 2048 # if val1 >= 2048 else val1
            val2 = val2 - 2048 # if val2 >= 2048 else val2
    
            conv_val.extend([val1, val2])
    
        return conv_val

    def _comp_thermal_offset(self,df_meas:pd.Series):
        warnings.warn(
        "TPArray._comp_thermal_offset is deprecated and will be removed. "
        "Use TPArray.comp_thermal_offset instead.",
        DeprecationWarning,
        stacklevel=2)
        return self.comp_thermal_offset(df_meas)

    def comp_thermal_offset(self,df_meas:pd.Series):
        
        # Check type
        if not isinstance(df_meas,pd.Series):
            raise TypeError('df_meas must be pd.Series type')
        
        ''' Thermal offset compensation '''
        if self.DesignGen <= 3:
            return self._comp_thermal_offset_3(df_meas)
        elif self.DesignGen == 4:
            return self._comp_thermal_offset_4(df_meas)
        else:
            raise ValueError('DesignGen is not set or value not known.')


    def _comp_thermal_offset_3(self,df_meas:pd.Series):
        
        ''' Thermal offset compensation '''
        
        # Only for this function reverse self._size for easy use in numpy
        size = (self._size[1],self._size[0])
        
        Pixel = df_meas[self._pix] 
        pixel_dtype = df_meas[self._pix].dtypes
        
        # Get stuff for calculation
        ThGrad = self.BCC['thGrad'].reshape(size)
        # avgPtat = df_meas[self._PTAT].mean().item()
        gradScale = self.BCC['gradScale']
        ThOffset = self.BCC['thOff'].reshape(size)
        
        
        if (self.width,self.height) == (8,8):
            T_depend = df_meas[self._T_amb].item()
        else:
            T_depend = df_meas[self._PTAT].mean().item()
            
        V_th_comp = Pixel.values.reshape(size) -\
            (ThGrad*T_depend) / np.power(2*np.ones(size),gradScale) -\
                ThOffset
         
        df_meas.loc[self._pix] = V_th_comp.flatten().astype(pixel_dtype)
        
        return df_meas
    
    def _comp_thermal_offset_4(self,df_meas:pd.Series):
        
        ''' Thermal offset compensation '''
        
        raise NotImplementedError('Thermal offset compensation not implemented '
                                  'for this htpa device')
        
        return df_meas

    def _comp_electrical_offset(self,df_meas:pd.Series):
        warnings.warn(
        "TPArray._comp_electrical_offset is deprecated and will be removed. "
        "Use TPArray.comp_electrical_offset instead.",
        DeprecationWarning,
        stacklevel=2)
        return self.comp_electrical_offset(df_meas)
    
    def comp_electrical_offset(self,df_meas:pd.Series | pd.DataFrame):
        
        # Check type
        if not (isinstance(df_meas,pd.Series) or isinstance(df_meas,pd.DataFrame)):
            raise TypeError('df_meas must be pd.Series or pd.DataFrame')
        
        ''' Thermal offset compensation '''
        if self.DesignGen <= 3:
            return self._comp_electrical_offset_3(df_meas)
        elif self.DesignGen == 4:
            return self._comp_electrical_offset_4(df_meas)
        else:
            raise ValueError('DesignGen is not set or value not known.')
        
    def _comp_electrical_offset_3(self,df_meas:pd.Series | pd.DataFrame):
        """
        

        Parameters
        ----------
        df_meas : pd.Series
            Single measurement as pandas Series.

        Returns
        -------
        df_meas : pd.Series
            Measurement compensated by electrical offsets.

        """
        
        if not (isinstance(df_meas,pd.Series) or isinstance(df_meas,pd.DataFrame)):
            raise TypeError('df_meas must be pd.Series or pd.DataFrame')
            return None
        
        
        def comp_electrical_offset_row(df_row : pd.Series):
        
            ''' Electrical offset compensation '''
            ElOff = df_row[self._e_off]
            
          
            Pixel = df_row[self._pix] 
            
            
            # Replicate electrical offsets corresponding to their pixels
            if self._DevConst['NROFPTAT']==2:
                ElOff_upper_half = ElOff.iloc[0:int(len(ElOff)/2)]
                ElOff_lower_half = ElOff.iloc[int(len(ElOff)/2)::]
                
                # Replicate the electrical offsets for the lower and upper
                # half NROFBLOCKS-times
                ElOff_upper_half = pd.concat([ElOff_upper_half]*\
                                             self._DevConst['NROFBLOCKS'],axis=0)
                ElOff_lower_half = pd.concat([ElOff_lower_half]*\
                                             self._DevConst['NROFBLOCKS'],axis=0)
                # Concatenate
                ElOff = pd.concat([ElOff_upper_half,
                                   ElOff_lower_half])
                
            elif self._DevConst['NROFPTAT']==1:
                raise NotImplementedError('Yet to be implemented! Ask Bodo or Christoph!')
                pass
            
            V_el_comp = Pixel.values - ElOff.values
            
            df_row.loc[self._pix] = V_el_comp
            
            return df_row
        
        if isinstance(df_meas,pd.Series):
            df_comp = comp_electrical_offset_row(df_meas)
            
        elif isinstance(df_meas,pd.DataFrame):
            comp_list = []
            for i in df_meas.index:
                comp_list.append(comp_electrical_offset_row(df_meas.loc[i]))
            
            df_comp = pd.concat(comp_list, axis = 1).T
            
        return df_comp
            
    
    def _comp_electrical_offset_4(self,df_meas:pd.Series | pd.DataFrame):
        
        if not (isinstance(df_meas,pd.Series) or isinstance(df_meas,pd.DataFrame)):
            raise TypeError('df_meas must be pd.Series or pd.DataFrame')
            return None
        
        ''' Eletrical offset compensation '''
        
        def comp_electrical_offset_row(df_row : pd.Series):
        
            # Obtain pixels and electrical offsets from df_meas
            pix = df_row[self.DataCols.pix].values
            e_off = df_row[self.DataCols.e_off].values
            
            # Reshape pixels
            pix = pix.reshape(self._npsize)
            
            # Repeat electrical offsets along vertical axis
            e_off = np.tile(e_off,(self.height,1))
            
            # Subtract
            pix_comp = pix - e_off
            
            # Flatten and reassign to DataFrame
            df_row[self.DataCols.pix] = pix_comp.flatten()
            # raise NotImplementedError('Eletrical offset compensation not implemented '
            #                           'for this htpa device')
            
            return df_row
        
        if isinstance(df_meas,pd.Series):
            df_comp = comp_electrical_offset_row(df_meas)
            
        elif isinstance(df_meas,pd.DataFrame):
            comp_list = []
            for i in df_meas.index:
                comp_list.append(comp_electrical_offset_row(df_meas.loc[i]))
            
            df_comp = pd.concat(comp_list, axis = 1).T
            
        return df_comp
        
        
        return df_meas

    def _comp_vdd(self,df_meas:pd.Series):
        warnings.warn(
        "TPArray._comp_vdd is deprecated and will be removed. "
        "Use TPArray.comp_vdd instead.",
        DeprecationWarning,
        stacklevel=2)
        return self.comp_vdd(df_meas)

    def comp_vdd(self,df_meas:pd.Series):
        
        # Check type
        if not isinstance(df_meas,pd.Series):
            raise TypeError('df_meas must be pd.Series type')
        
        ''' Vdd compensation '''
        if self.DesignGen <= 3:
            return self._comp_vdd_3(df_meas)
        elif self.DesignGen == 4:
            return self._comp_vdd_4(df_meas)
        else:
            raise ValueError('DesignGen is not set or value not known.')
        
    def _comp_vdd_3(self,df_meas:pd.Series):
        
        ''' Vdd compensation '''
        
        Pixel = df_meas[self._pix] 
        pixel_dtype = df_meas[self._pix].dtypes
        
        # Get stuff for calculation
        vddCompGrad = self.BCC['vddCompGrad']
        vddCompOff = self.BCC['vddCompOff']
        vddScOff = self.BCC['vddScOff'].item()
        vddScGrad = self.BCC['vddScGrad'].item()


        vdd_av = df_meas[self._vdd].values.item()
        vdd_th1 = self.BCC['avgVDD_ThCalib_@(Ta1,Vdd2)']
        vdd_th2 = self.BCC['avgVDD_ThCalib_@(Ta2,Vdd2)']
        ptat_th1 = self.BCC['avgPTAT_ThCalib_@(Ta1,Vdd2)']
        ptat_th2 = self.BCC['avgPTAT_ThCalib_@(Ta2,Vdd2)']
        ptat_av = df_meas[self._PTAT].mean()

        # Replicate vddCompGrad and vddCompOff according to their
        # corresponding pixels
        
        # Replicate electrical offsets corresponding to their pixels
        if self._DevConst['NROFPTAT']==2:
            
            vdd_shape =  vddCompGrad.shape
            
            vddCompGrad_uh = vddCompGrad[0:int(vdd_shape[0]/2),:].flatten()
            vddCompGrad_lh = vddCompGrad[int(vdd_shape[0]/2):,:].flatten()
            
            vddCompOff_uh = vddCompOff[0:int(vdd_shape[0]/2),:].flatten()
            vddCompOff_lh = vddCompOff[int(vdd_shape[0]/2):,:].flatten()   
            
            # Replicate them all NROFBLOCKS-times
            vddCompGrad_uh = np.hstack([vddCompGrad_uh]*self._DevConst['NROFBLOCKS'])
            vddCompGrad_lh = np.hstack([vddCompGrad_lh]*self._DevConst['NROFBLOCKS'])
            vddCompOff_uh = np.hstack([vddCompOff_uh]*self._DevConst['NROFBLOCKS'])
            vddCompOff_lh = np.hstack([vddCompOff_lh]*self._DevConst['NROFBLOCKS'])
            
            # Concatenate
            vddCompGrad = np.hstack([vddCompGrad_uh,vddCompGrad_lh])
            vddCompOff = np.hstack([vddCompOff_uh,vddCompOff_lh])
                        
        elif self._DevConst['NROFPTAT']==1:
            print('Yet to be implemented! Ask Bodo or Christoph!')
            return None
        
        # Apply compensation 
        vdd = ((vddCompGrad*ptat_av)/(2**vddScGrad)+vddCompOff) / (2**vddScOff)
        vdd = vdd * (vdd_av - vdd_th1 - \
                     ((vdd_th2-vdd_th1)/(ptat_th2-ptat_th1))*(ptat_av-ptat_th1))
        
        V_vdd_comp = Pixel.values - vdd
        
        df_meas.loc[self._pix] = V_vdd_comp.flatten().astype(pixel_dtype)
        
        return df_meas

    def _comp_vdd_4(self,df_meas:pd.Series):
        
        ''' Vdd compensation '''
        
        raise NotImplementedError('Vdd compensation not implemented '
                                  'for this htpa device')
        

    def _comp_sens(self,df_meas:pd.Series):
        warnings.warn(
        "TPArray._comp_sens is deprecated and will be removed. "
        "Use TPArray.comp_sens instead.",
        DeprecationWarning,
        stacklevel=2)
        return self.comp_sens(df_meas)

    def comp_sens(self,df_meas:pd.Series):
        
        # Check type
        if not isinstance(df_meas,pd.Series):
            raise TypeError('df_meas must be pd.Series type')
            
        ''' Sensitivity compensation '''
        if self.DesignGen <= 3:
            return self._comp_sens_3(df_meas)
        elif self.DesignGen == 4:
            return self._comp_sens_4(df_meas)
        else:
            raise ValueError('DesignGen is not set or value not known.')
    
    def _comp_sens_3(self,df_meas:pd.Series):
        
        ''' Sensitivity compensation '''
        Pixel = df_meas[self._pix] 
        pixel_dtype = df_meas[self._pix].dtypes

        
        # Get stuff for calculation
        Pij = self.BCC['pij']
        PixCmin = self.BCC['pixcmin']
        PixCmax = self.BCC['pixcmax']
        GlobGain = self.BCC['globalGain']
        eps = self.BCC['epsilon']
        
        
        # Calculate Sensitivity coefficients
        PixC = (( Pij.reshape((-1,1)) * (PixCmax-PixCmin)  / 65535)  + PixCmin) \
            * eps/100 * GlobGain/10000
            
        # Apply pixel constants
        VijPixC =  Pixel * self._PCSCALEVAL / PixC.flatten()
        
        # Write to dataframe
        df_meas.loc[self._pix] = VijPixC.astype(pixel_dtype)
        
        return df_meas

    def _comp_sens_4(self,df_meas:pd.Series):
        
        ''' Sensitivity compensation '''
        
        raise NotImplementedError('Vdd compensation not implemented '
                                  'for this htpa device')

    def _calc_Tamb0(self,df_meas:pd.Series):
        warnings.warn(
        "TPArray._calc_Tamb0 is deprecated and will be removed. "
        "Use TPArray.calc_Tamb0 instead.",
        DeprecationWarning,
        stacklevel=2)
        return self.calc_Tamb0(df_meas)

    def calc_Tamb0(self,df_meas:pd.Series):
        
        # Check type
        if not isinstance(df_meas,pd.Series):
            raise TypeError('df_meas must be pd.Series type')
        
        ''' Sensitivity compensation '''
        if self.DesignGen <= 3:
            return self._calc_Tamb0_3(df_meas)
        elif self.DesignGen == 4:
            return self._calc_Tamb0_4(df_meas)
        else:
            raise ValueError('DesignGen is not set or value not known.')

    def _calc_Tamb0_3(self,df_meas:pd.Series):
        
        ptat_av = df_meas[self._PTAT].mean()
        
        ptat_grad = self.BCC['ptatGrad']
        ptat_off = self.BCC['ptatOffset']
        
        Tamb0 = ptat_av*ptat_grad+ptat_off
        
        dtype = df_meas.loc[self._T_amb].dtypes
        df_meas.loc[self._T_amb] = Tamb0.astype(dtype)
        
        return df_meas
    
    def _calc_Tamb0_4(self,df_meas:pd.Series):
        
        ptat_av = df_meas[self._PTAT].mean()
        
        ptat_grad = self.BCC['ptatGrad']
        ptat_off = self.BCC['ptatOffset']
        
        Tamb0 = ptat_av*ptat_grad+ptat_off
        
        dtype = df_meas.loc[self._T_amb].dtypes
        df_meas.loc[self._T_amb] = Tamb0.astype(dtype)
        
        return df_meas
        
    
    def frame_to_blocks(self,frame:np.ndarray,**kwargs)->dict:
        """
        Divides the frame into its blocks. Content of the frame is rearranged 
        into a dictionary where each entry corresponds to a block.
        Parameters
        ----------
        frame : np.ndarray
            Frame to be divided into its blocks.
        **kwargs : dict
            Optional keyword arguments.

        Returns
        -------
        dict
            Dictionary containing the content of each block. in the following 
            manner:
              block_dict[0] = (rows_upper_half_block0, rows_lower_half_block0),
              ...
              block_dict[N] = (rows_upper_half_block0, rows_lower_half_blockN)

        """
        
        # Calculate the number of rows
        rows = self._rowsPerBlock
        
        # Dictionary for storing blocks in
        block_dict = {}
        
        # Divide the frame into an upper and a lower half. Flip the lower half.
        frame_upper_half = frame[0:int(self.height/2)]
        frame_lower_half = frame[int(self.height/2)::]
        
        frame_lower_half = np.flipud(frame_lower_half)
        
        # Loop through the upper and lower half simultaneously
        for block in range(self._DevConst['NROFBLOCKS']):
            
            # Extract the content of the block from the lower and upper half
            top_rows = frame_upper_half[block*rows:(block+1)*rows,::]
            bottom_rows = frame_lower_half[block*rows:(block+1)*rows,::]
        
            # Write block of upper and lower half to dict
            block_dict[block] = (top_rows,bottom_rows)


        return block_dict
    
    
    def Ucomp2Uscaled(self,df_meas:pd.DataFrame):
        """
        Apply sensitivity coefficients to provided data. Data should contain 
        measurements, to which electrical offsets as well as thermal and vdd 
        compensation have already been applied.

        Parameters
        ----------
        df_meas : pd.DataFrame
            DESCRIPTION.

        Returns
        -------
        None.

        """
    
        # Convert pixel values to signed interger 64bit
        df_meas = df_meas.astype(np.int64)
        
        df_calib = []
        
        for i in df_meas.index:
            
            df_frame = df_meas.loc[i]
            
            df_frame = self._comp_sens(df_frame)
                
            df_frame = self._calc_Tamb0(df_frame)
        
            # Convert back to DataFrame
            df_frame = pd.DataFrame(df_frame).transpose()
            df_calib.append(df_frame)
        
        df_calib = pd.concat(df_calib)
        
        return df_calib
    
    def rawmeas_comp(self,df_meas:pd.DataFrame,**kwargs):
        """
        Copy from Calc_CompTemp.py, only application of calibration, 
        no conversion to dK
        """
        
        # Apply pixel constants for sensitivity compensation?
        comp_sense = kwargs.pop('comp_sense',True)
        
        # Convert pixel values to signed interger 64bit
        try:
            df_meas = df_meas.astype(np.int64)
        except:
            pass
        
        df_calib = []
        
        for i in df_meas.index:
            
            df_frame = df_meas.loc[i]
            
            df_frame = self._comp_electrical_offset(df_frame)
            
            df_frame = self._comp_thermal_offset(df_frame.copy())
            
            # Vdd compensation for all sensors but 8x8
            if not self.ArrayType == ArrayTypes['HTPA8x8']:
                df_frame = self._comp_vdd(df_frame)
            
            # Compensate pixel constants only on demand
            if comp_sense == True:
                df_frame = self._comp_sens(df_frame)
                
            df_frame = self._calc_Tamb0(df_frame)
        
            # Convert back to DataFrame
            df_frame = pd.DataFrame(df_frame).transpose()
            df_calib.append(df_frame)
            
            print(f'Applied calibration to frame {i}')
        
        df_calib = pd.concat(df_calib)
        
        return df_calib
    
    def rawmeas_to_dK(self,df_meas:pd.DataFrame):
        """
        Copy from Calc_CompTemp.py, no compensation of pixel sensitivity and
        no conversion in dK
        """
        
        if isinstance(df_meas,pd.Series):
            df_meas = df_meas.to_frame().T
        
        # Perform all compensation operations on data
        df_meas = self.rawmeas_comp(df_meas)
              
        df_dK = []
        
        # Map every single pixel to the LuT
        for i in df_meas.index:
            
            df_frame = df_meas.loc[i]
            
            for p in self._pix:
                
                Ud = df_frame[p]
                Tamb0 = df_frame[self._T_amb[0]] / 10   # Convert from dK to K
                
                pnt = pd.DataFrame(data = [[Ud,Tamb0]],
                                   columns = ['Ud','Tamb0'])
                
                # try:
                pnt = self._LuT.eval_LuT(pnt)
                # except:
                    # print(pnt)
                    # raise Exception('Error converting the printed measurement')
                df_frame[p] = int(pnt['To_LuT'].item()*10)
                
            print(f'Converted frame {i} to dK')
                
            # Convert back to DataFrame
            df_frame = pd.DataFrame(df_frame).transpose()
            df_dK.append(df_frame)
                
        df_dK = pd.concat(df_dK)
        
        return df_dK
    
    def _binary_mask(self,r_lim:float)->np.ndarray:
        """
        Creates a binary mask that is 1 if distance from image center
        is less of equal to r and 0 otherwise

        Parameters
        ----------
        r_lim : float
            Maximal distance in pixels from the center, where mask is supposed 
            to be 1.

        Returns
        -------
        None.

        """
        
        # Calculate center
        x_center = (self.width-1) / 2
        y_center = (self.height-1) / 2
        
        # Initialize mask
        mask = np.zeros(( self.height, self.width))
        
        # Create meshgrid of coordinates
        y, x = np.meshgrid(np.arange(self.height), np.arange(self.width),
                           indexing = 'ij')

        
        # Calculate the distance of each point from the center
        r = np.sqrt((x - x_center)**2 + (y - y_center)**2)
        
        # Use numpy.where to set values to 0 if d is larger than r_lim
        mask = np.where(r > r_lim, 0, 1)
        
        return mask
    
    def df_to_np(self,df,**kwargs):
        
        row_idx = kwargs.pop('idx',df.index[0])

        # Reshape dataframe to numpy array, if dataframe has multiple rows,
        # take the first one by default
        img = df.loc[row_idx,self._pix]
        img = img.values.reshape(self._npsize)
        
        return img
    
    def save(self):
        '''
        Returns all non-private an non-builtin attributes of this class
        as a dictionary with the purpose of reloading this instance from the
        attribute dictionary. 

        Returns
        -------
        None.

        '''
        
        
        
        # Get all names of properties of the instance by doing
        properties = []
        for d in dir(self):
            if isinstance(getattr(type(self), d, None), property):
                properties.append(d)
                
        
        # Save all properties to an attr_dict
        # Some properties have their own save-method. Use that, where available
        attr_dict = {}
        
        # Loop over property keys
        for p in properties:
            
            # Get property value
            prop = getattr(self,p) 
            
            # Check if prop has a save method
            save_method = getattr(prop, "save", None)
            
            # Check if its a callable method
            if callable(save_method):
                # If its a callable save method, call it
                attr_dict[p] = {}
                attr_dict[p] = save_method()
            else:
                attr_dict[p] = prop
                
        # Get all class attributes as well
        class_dict = {}
        for attribute in TPArray.__dict__.keys():
            # Check if it's a built-in type
            if (attribute[:2] != '__') and attribute not in attr_dict :
                # Check if its a method
                value = getattr(TPArray, attribute)
                if not callable(value):
                    # If not append to dict
                    class_dict[attribute] = value
        
        # Concatenate both
        attr_dict.update(class_dict)
            
        return attr_dict 


@dataclass
class DataCols(Sequence[str]):
    """
    - Attribute access: cols.pix -> list of headers in group "pix"
    - Sequence behavior: list(cols) -> all headers concatenated
    """
    pix: List[str]
    e_off: List[str]
    vdd: List[str]
    T_amb: List[str]
    PTAT: List[str]
    ATC: List[str]
    
    @property
    def _groups(self) -> Dict[str, List[str]]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def __iter__(self) -> Iterator[str]:
        for group in self._groups.values():
            yield from group

    def __len__(self) -> int:
        return sum(len(g) for g in self._groups.values())

    def __getitem__(self, idx: int) -> str:
        # Index into the flattened/concatenated view
        if idx < 0:
            idx = len(self) + idx
        for group in self._groups.values():
            if idx < len(group):
                return group[idx]
            idx -= len(group)
        raise IndexError("DataCols index out of range")

    def all(self) -> List[str]:
        """Explicit 'give me the concatenated list'."""
        return list(self)

    def as_dict(self) -> Mapping[str, List[str]]:
        """Optional: expose groups safely."""
        return dict(self._groups)