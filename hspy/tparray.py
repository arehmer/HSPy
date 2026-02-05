# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 11:16:48 2023

@author: Rehmer
"""
import numpy as np
import pandas as pd
import json
from pathlib import Path
import struct
import time

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
              'HTPA80x64d'	: 11,
              'HTPA120x84d' : 12,
              'HTPA84x60d' :	13,
              'HTPA60x40d'	: 14,
              'HTPA160x120d' :	15,
              'HTPA120x84dR2' : 16,
              'HTPA16x16dR3' : 17,
              'HTPA160x120dR1' : 18,
              'HTPA80x60d' : 19,
              'HTPA60x40dr2' : 20}

class TPArray():
    """
    Class contains hard-coded properties of Thermopile-Arrays relevant
    for reading from Bytestream
    """
    
    def __init__(self,**attr_dict):
        
        self._SensorType = attr_dict.pop('SensorType',None)
        self._ArrayType = attr_dict.pop('ArrayType',None)
        
        if self._SensorType is not None:
            self._init_by_SensorType()
        elif self._ArrayType is not None:
            self._init_by_ArrayType()
        else:
            raise Exception('Provide either SensorType or ArrayType!')
        
        
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

    def _init_by_SensorType(self):
        
        if (self.SensorType == SensorTypes['HTPA60x40D_L1K9_0K8']):
            self._width = 60
            self._height = 40
        elif (self.SensorType == SensorTypes['HTPA120x84DR2_L3K95_0K8']):
            self._width = 120
            self._height = 84
        elif (self.SensorType == SensorTypes['HTPA160x120DR1_L3K95_0K8']):
            self._width = 160
            self._height = 120
        elif (self.SensorType == SensorTypes['HTPA8x8DR1_L0K8_0K8']):
            self._width = 8
            self._height = 8
        elif (self.SensorType == SensorTypes['HTPA32x32dR2_L1k9_0k8']):
            self._width = 32
            self._height = 32
            self._NETD = 60 # Tuned on Archesens-Data
        elif (self.SensorType == SensorTypes['HTPA32x32dR2_L1k7_0k8']):
            self._width = 32
            self._height = 32
            self._NETD = 152 # from datasheet   
        elif self.SensorType is None:
            self._width = None
            self._height = None
        else:
            raise Exception('SensorType not known!')
            
    def _init_by_ArrayType(self):
        
        if (self.ArrayType == ArrayTypes['HTPA8x8']):
            self._width = 8
            self._height = 8
        elif (self.ArrayType == ArrayTypes['HTPA16x16']):
            self._width = 16
            self._height = 16
        elif (self.ArrayType == ArrayTypes['HTPA32x32d']):
            self._width = 32
            self._height = 32
        elif (self.ArrayType == ArrayTypes['HTPA80x64d']):
            self._width = 80
            self._height = 64
        elif (self.ArrayType == ArrayTypes['HTPA120x84d']):
            self._width = 120
            self._height = 84
        elif (self.ArrayType == ArrayTypes['HTPA84x60d']):
            self._width = 84
            self._height = 60
        elif (self.ArrayType == ArrayTypes['HTPA60x40d']):
            self._width = 60
            self._height = 40
        elif (self.ArrayType == ArrayTypes['HTPA160x120d']):
            self._width = 160
            self._height = 120
        elif (self.ArrayType == ArrayTypes['HTPA120x84dR2']):
            self._width = 120
            self._height = 84
        elif (self.ArrayType == ArrayTypes['HTPA16x16dR3']):
            self._width = 16
            self._height = 16
        elif (self.ArrayType == ArrayTypes['HTPA160x120dR1']):
            self._width = 160
            self._height = 120
        elif (self.ArrayType == ArrayTypes['HTPA80x60d']):
            self._width = 80
            self._height = 60
        elif (self.ArrayType == ArrayTypes['HTPA60x40dr2']):
            self._width = 60
            self._height = 40  
        elif self.ArrayType is None:
            self._width = None
            self._height = None
        else:
            raise Exception('ArrayType not known!')

    def _init_DevConst(self):
        
        self._size = (self.width,self.height)
        self._npsize = (self.height,self.width)
        
        # Calculate order of data in bds file
        DevConst = {}
        
        if (self.width,self.height) == (8,8):
            DevConst['ATCaddr']=0
            DevConst['NROFBLOCKS']=1
            DevConst['NROFPTAT']=1
            
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
            
        elif (self.width,self.height) == (16,16):
            DevConst['ATCaddr']=0
            DevConst['NROFBLOCKS']=2
            DevConst['NROFPTAT']=2
            
            self._package_num = 1
            self._package_size = 780
            self._fs = 70
            self._NETD = 130
            self.Pitch = 90.0e-6    #equal for r3
            self.Ampl = 40          #equal for r3  
            
            self._mask = np.ones(self._npsize)
            
            print('16x16.json has not yet been tested!')
            # path to array data
            path = Path(__file__).parent / 'arraytypes' / '16x16.json'
            # Load calibration data from file
            self._load_calib_json(path)  
        
        elif (self.width,self.height) == (32,32):
            DevConst['ATCaddr']=0
            DevConst['NROFBLOCKS']=4
            DevConst['NROFPTAT']=2
            
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
            
        elif (self.width,self.height) == (80,64):
            DevConst['NROFBLOCKS']=4
            DevConst['NROFPTAT']=2
            DevConst['ATCaddr']=0
            
            self._package_num = 10
            self._package_size = 1283
            self._fs = 41
            self._NETD = 70
            
            self._mask = np.ones(self._npsize)
            
            # path to array data
            path = Path(__file__).parent / 'arraytypes' / '80x64.json'
            # Load calibration data from file
            self._load_calib_json(path)  
            
        elif (self.width,self.height) == (60,84):
            DevConst['NROFBLOCKS']=7
            DevConst['NROFPTAT']=2
            DevConst['ATCaddr']= 1
                        
            
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
            
        elif (self.width,self.height) == (120,84):
            DevConst['ATCaddr']=0
            DevConst['NROFBLOCKS']=6
            DevConst['NROFPTAT']=2
            
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
            
        elif (self.width,self.height) == (60,40):
            DevConst['ATCaddr']=1
            DevConst['NROFBLOCKS']=5
            DevConst['NROFPTAT']=2
            
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

        elif (self.width,self.height) == (160,120):
            DevConst['ATCaddr'] = 1
            DevConst['NROFBLOCKS'] = 12
            DevConst['NROFPTAT'] = 2
            
            self._package_num = 30
            self._package_size = 1401
            self._fs = 25
            self._NETD = 110
            self.Pitch = 45.0e-6        #equal for r1
            self.Ampl = 60              #40 for r1         
            
            self._mask = np.ones(self._npsize)
            
            # path to array data
            path = Path(__file__).parent / 'arraytypes' / '160x120.json'
            # Load calibration data from file
            self._load_calib_json(path)
            
        elif (self.width,self.height) == (80,60):
            DevConst['ATCaddr'] = 1
            DevConst['NROFBLOCKS'] = 6
            DevConst['NROFPTAT'] = 2
            
            warnings.warn('ArrayType not fully implemented!')
            
            # self._package_num = 30
            # self._package_size = 1401
            # self._fs = 25
            # self._NETD = 110
            # self.Pitch = 45.0e-6        #equal for r1
            # self.Ampl = 60              #40 for r1         
            
            # self._mask = np.ones(self._npsize)
            
            # path to array data
            # path = Path(__file__).parent / 'arraytypes' / '160x120.json'
            # Load calibration data from file
            # self._load_calib_json(path)
        
        else:
            raise Exception('This Thermopile Array is not known.') 
         
        # Remaining DevConst can be derived
        DevConst['VDDaddr'] = \
            int(self.width*self.height+self.height/DevConst['NROFBLOCKS']*self.width)   
            
        DevConst['TAaddr']=DevConst['VDDaddr'] + 1
        DevConst['PTaddr']=DevConst['TAaddr'] + 1
            
        self._DevConst = DevConst        
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
        if DevConst['ATCaddr'] == 0:
            no_atc = 0
        else:
            no_atc = 2
            
        ATC = ['ATC'+str(a) for a in range(0,no_atc)]
        
        
        self._pix = pix
        self._e_off = e_off
        self._vdd = vdd
        self._T_amb = T_amb
        self._PTAT = PTAT
        self._ATC = ATC
        
        self._serial_data_order = pix + e_off + vdd + T_amb + PTAT + ATC
        
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

        
        # Convert all arrays to appropriate shape and flip them
        # properly
        bcc['pij'] = np.array(bcc['pij']).reshape(self._npsize)
        bcc['thGrad'] = np.array(bcc['thGrad']).reshape(self._npsize)
        bcc['thOff'] = np.array(bcc['thOff']).reshape(self._npsize)
        
        # Only 8x8 Arrays don't have vdd calibration data and pij, thGrad and 
        # thOff are not flipped
        if not (self.width,self.height) == (8,8):
            
            NROFBLOCKS = self.get_DevConst()['NROFBLOCKS']
            vdd_size = (int(self._height/NROFBLOCKS),self._width)
                        
            # The lower half needs to be flipped vertically
            bcc['pij'][int(self._height/2):,::] = \
                np.flipud(bcc['pij'][int(self._height/2):,::])
            
            bcc['thGrad'][int(self._height/2):,::] = \
                np.flipud(bcc['thGrad'][int(self._height/2):,::])
                
            bcc['thOff'][int(self._height/2):,::] = \
                np.flipud(bcc['thOff'][int(self._height/2):,::])
            
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
    
    def _comp_electrical_offset(self,df_meas:pd.Series):
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
        
        if not isinstance(df_meas,pd.Series):
            raise Exception('Excpected a single measurement frame as pd.Series')
            return None
        
        
        ''' Electrical offset compensation '''
        ElOff = df_meas[self._e_off]
        
      
        Pixel = df_meas[self._pix] 
        
        
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
        
        df_meas.loc[self._pix] = V_el_comp
        
        return df_meas
        
    def _comp_vdd(self,df_meas:pd.Series):
        
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
    
    def _comp_sens(self,df_meas:pd.Series):
        
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
        
    def _calc_Tamb0(self,df_meas:pd.Series):
        
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
            if not (self.width,self.height) == (8,8) :
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
        x_center = (self._width-1) / 2
        y_center = (self._height-1) / 2
        
        # Initialize mask
        mask = np.zeros(( self._height, self._width))
        
        # Create meshgrid of coordinates
        y, x = np.meshgrid(np.arange(self._height), np.arange(self._width),
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
        