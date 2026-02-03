# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 11:11:32 2023

@author: Rehmer
"""
import pandas as pd
from pathlib import Path
import h5py
import numpy as np
import warnings
import os
import inspect

from hspytools.readers import HTPAdGUI_FileReader
from hspytools.tparray import TPArray

class hdf5_mgr():

    def __init__(self,hdf5_path,**kwargs):
        """
        Initializes the container and links it to the hdf5 file specified via 
        the given path.
        
        Parameters
        ----------
        path : str
            path to the hdf5-file that already exists or will be created by
            calling this method.
        mode : char
            A character specifying what should be done if the file already
            exists. 
            'a': Read/write if exists, create otherwise
            'w': overwrite
            
        Returns
        -------
        None.
        """
        
        width = kwargs.pop('width',None)
        height = kwargs.pop('height',None)
        
                
        # Check if file exists
        if os.path.isfile(hdf5_path) == True:
            file_exists = True
            print('Specified file already exists.' )
        
        # open hdf5_file or create if it doesn't exist
        hdf5_file = h5py.File(hdf5_path, "a")
                       
        # Check if hdf5_file is empty
        if len(hdf5_file.keys()) == 0:
            is_empty = True
        else:
            is_empty = False
        
        # Save path of hdf5 file for future reference
        self._hdf5_path = hdf5_path
        
        hdf5_file.close()

        # If hdf5 file is completely empty, initialize attributes of the class
        # and write them to the hdf5 file
        if is_empty:
            self.width = kwargs.pop('width',0)
            self.height = kwargs.pop('height',0)
            self.hdf5Index_dtypes = kwargs.pop('hdf5Index_dtypes',{})
            
            # Also create an empty DataFrame which will serve as a ledger 
            # of the files content
            df_index = pd.DataFrame(data = [])
            df_index.to_hdf(self._hdf5_path, key = 'index')
              
        else:
            # Else read dict with attributes from the file
            with h5py.File(self._hdf5_path, 'r')  as h5file:
                
                attr_dict = {}
                
                # Loop over all items in attributes
                for attr_name in h5file['attributes'].keys():
                    
                    # Check if item itself has subitems
                    if not getattr(h5file['attributes/' + attr_name], "keys", False):
                        attr_dict[attr_name] = \
                            h5file['attributes/' + attr_name][()]
                    else:
                        
                        attr_dict[attr_name] = {}
                        
                        for key in h5file['attributes/' + attr_name].keys():
                            attr_dict[attr_name][key] = \
                                h5file['attributes/' + attr_name+ '/' + key][()]
            
            # And assign them
            for attr in attr_dict.keys():
                setattr(self,attr,attr_dict[attr])
    
    @property
    def hdf5_path(self):
        return self._hdf5_path
    @hdf5_path.setter
    def hdf5_path(self,path:Path):
        if not isinstance(path,Path):
            raise TypeError('Path must be povided as pathlib.Path!')
        self._hdf5_path = path
    
    
    
    @property
    def hdf5Index_dtypes(self):
        return self._hdf5Index_dtypes
    @hdf5Index_dtypes.setter
    def hdf5Index_dtypes(self,dtypes:dict):
        
        # Check type
        if not isinstance(dtypes,dict):
            raise TypeError('dtypes must be provided as a dictionary!')
        
        # When read from the hdf5 file, strings are bytestrings. Convert
        for key in dtypes.keys():
            if isinstance(dtypes[key],str):
                pass
            elif isinstance(dtypes[key],bytes):
                dtypes[key] = dtypes[key].decode('utf-8')
            else:
                raise TypeError(f'{key} datatype must be provided as str or  '+\
                                'bytes but is provided as f{type(dtypes[key])}')
                
        self._hdf5Index_dtypes = dtypes
        self._save_attributes_to_hdf5()        
    
    def _save_attributes_to_hdf5(self):
        """
        Writes all attributes of the class to the "attributes" group in the
        associated hdf5 file

        Returns
        -------
        None.

        """
        
        props = {}
        for name, val in inspect.getmembers(self.__class__):
            if isinstance(val, property):
                try:
                    props[name] = getattr(self, name)
                except Exception as e:
                    props[name] = f"<error: {e}>"
        
        # Path does not need to be saved
        props.pop('hdf5_path')
        
        # Write attributes of class to file
        with h5py.File(self._hdf5_path, 'a')  as h5file:
            for attr_name, attr_val in props.items():
                
                # Check if attributes dataset already exists
                if 'attributes' in h5file:
                    # If key already exists in file, delete it
                    if attr_name in h5file['attributes']:
                        del h5file['attributes/' + attr_name]
                
                # Write the updated attribute to file
                if isinstance(attr_val,dict):
                    for key, value in attr_val.items():
                        h5file['attributes/' + attr_name + '/' + key] = \
                            value       
                else:
                    h5file['attributes/' + attr_name] = attr_val
    
    def _initialize_index(self):

        df_index = pd.DataFrame(data = [])
        df_index.to_hdf(self._hdf5_path, key = 'index')

    def _initialize_types(self):
        """
        Initializes an empty group in the hdf5 file

        Returns
        -------
        None.

        """
        self._write_dict_to_hdf5('types',{})
    
    def _write_fields(self,field_dict):
        """
        Writes pandas dictionary arranged in field_dict to hdf5. Key of 
        field_dict is address in hdf5 to write to. Value is Data Frame to write
        """
        
        # Since a string is written to a dataframe and then saved in a hdf5-file
        # this will raise a PerformanceWarning, which is lengthy and should not
        # be displayed in the console
        
        with warnings.catch_warnings():
            
            warnings.simplefilter("ignore")
            
            for address in field_dict.keys():
                field_dict[address].to_hdf(self._hdf5_path,address)
                
    def load_df(self,address,**kwargs):
        
        return pd.read_hdf(self._hdf5_path, address)
    
    def load_index(self):
        
        address = 'index'
        
        df_index = pd.read_hdf(self._hdf5_path, address)
        
        # Check if hdf5-index contains any entries at this point
        if len(self.hdf5Index_dtypes)!=0 and len(df_index.columns)!=0:
            # Create the intersection between columns in the hdf5-index
            # and the keys in the hdf5Index_dtypes
            index_columns = list(df_index.columns)
            dtypes_keys =  list(self.hdf5Index_dtypes)
            intersect_keys = set(index_columns).intersection(dtypes_keys)
            
            # Cast types of hdf5-Index as specified
            df_index = df_index.astype({key:self.hdf5Index_dtypes[key] for key in intersect_keys} )
       
        return df_index
    
    def load_meas(self,idx,**kwargs):
    
        # Get adresses
        df_index = self.load_index()
        address = df_index.loc[idx,'address']
        
        # appendix = kwargs.pop('appendix','')
        
        # address = 'meas/' + meas_name
    
        # Load and return specified video sequence
        return pd.read_hdf(self._hdf5_path, address + '/data')
    
    def load_LuT(self,lut_name):
        
        lut_address = '/LuT/' + lut_name
        
        with h5py.File(self._hdf5_path,'a') as hdf5_file:
            keys = list(hdf5_file[lut_address].keys())
        
        # Load dfs for every key into dictionary 
        LuT =  self.load_df(lut_address)
        # {key:self.load_df(lut_address + '/' + key) for key in keys}
        
        
        return LuT
    
    def load_BCC(self,bcc_name):
        
        bcc_address = '/BCC/' + bcc_name
        
        with h5py.File(self._hdf5_path,'a') as hdf5_file:
            keys = list(hdf5_file[bcc_address].keys())
        
        # Load dfs for every key into dictionary 
        BCC = {key:self.load_df(bcc_address + '/' + key) for key in keys}
        
        # Convert BCC into numpy arrays
        BCC = {key:BCC[key].values for key in BCC.keys()}
        
        return BCC
    
    def delete_BCC(self,bcc_name):
        
        # Complete name to full hdf5_address
        bcc_address = '/BCC/' + bcc_name
        
        # load bds index
        bds_index = self.load_index()
        
        # Delete the BCC in the hdf5 file and references to it in bds_index 
        
        with h5py.File(self._hdf5_path,'a') as hdf5_file:
            try:
                del hdf5_file[bcc_address]
                bds_index.loc[bds_index['BCC'] == bcc_name,'BCC'] = None
            except:
                print(bcc_name + " couldn't be deleted or doesn't exist anymore")
                bds_index.loc[bds_index['BCC'] == bcc_name,'BCC'] = None
        
        # write the new bds index to the file
        self._write_fields({'index':bds_index})
        
        return None
    
    def _create_group_by_copy(self,source,target,groups,**kwargs):
                
        
        with h5py.File(self._hdf5_path,  'a') as hdf5_file:
                        
            # copy specified groups in source to target
            for group in groups:
                hdf5_file.copy(source+'/'+group,target+'/'+group)
            
        return None

    def _delete_group(self,group,**kwargs):
               
       
       with h5py.File(self._hdf5_path,  'a') as hdf5_file:
           
           if group in hdf5_file:
               del hdf5_file[group]
           else:
               warnings.warn('Group does not exist.')

       return None
    
        
    def _group_exists(self,hdf5_address:str)->bool:
        """
        Checks if a group already exists at the given address in the hdf5-file
    
        Parameters
        ----------
        hdf5_address : str
            DESCRIPTION.
    
        Returns
        -------
        bool
            DESCRIPTION.
    
        """
        
        group_exists = False
        
        with h5py.File(self._hdf5_path,'a') as hdf5_file:
            if hdf5_address in hdf5_file:
                group_exists = True
        
        return group_exists
    
    def video_to_avi(self,unique_id,**kwargs):
        """
        Writes a video sequence already stored in the hdf5-file under the name
        video_name to an mp4-file. Video will be saved in the folder specified 
        in folder_path under the file name <video_name>.mp4
        
        Parameters
        ----------
        video_name : str
            name under which the video is stored in the hdf5-file. If a 
            postprocessed version of the video is to be addressed, then pass
            'video_name/filtered' for example, or 'video_name/gradient'
        folder_path : str
            Path to the folder where the video should be saved. 
    
            
        Returns
        -------
        None.
        """
        
        print('''For annotation purposes it is highly recommended to export the 
              video sequence as png-images via write_to_png(). mp4 performs
              interpolations between frames that to not reflect the original
              sensor image and can deaviate from it.''')
        
        fs = kwargs.pop('fs',8)
        appendix = kwargs.pop('appendix','')
        
        
        # appendix = kwargs.pop('appendix','')
        
        # Convert device to list
        if not isinstance(unique_id,list):
            unique_id = [unique_id]
        
        # Load index of hdf5 file
        index = self.load_index()
        
        for u_id in unique_id:
            
            # Get address
            a = index.loc[u_id,'address']
            
            # Load specified video sequence
            df_video = self.load_df(a+'/data')
            
            # Get size information
            (w,h) = (self.tparray.width,self.tparray.height)
            
            # Initialize a HTPAdGUI_FileReader, which has a function for 
            # writing DataFrames of video to .png
            reader = HTPAdGUI_FileReader(w,h)
            
            # Pass the DataFrame to the method for writing to .png
            path = Path.cwd() /  (a+'/avi')
            video_name = a.split('/')[-1]
            reader.export_avi(df_video,video_name,path)
            
        return None
        
    
    
    def video_to_png(self,unique_id,**kwargs):
        """
        Writes a video sequence already stored in the hdf5-file under the name
        video_name to png-images frame by frame. The files will be saved 
        in a folder with the name <video_name> in the folder specified in
        folder_path. The png-files will be named according to their frame 
        number.
        
        Parameters
        ----------
        video_name : str
            name under which the video is stored in the hdf5-file. If a 
            postprocessed version of the video is to be addressed, then pass
            'video_name/filtered' for example, or 'video_name/gradient'
        folder_path : str
            Path to the folder where the video should be saved. 
    
            
        Returns
        -------
        None.
        """
        
       
        # Load index of hdf5 file
        index = self.load_index()
        
        for u_id in unique_id:
            
               
            # Get address
            a = index.loc[u_id,'address']
            
            # Load specified video sequence
            df_video = self.load_df(a+'/data')
            
            # Get size information
            (w,h) = (index.loc[u_id,'Width'],index.loc[u_id,'Height'])
            
            # Initialize a HTPAdGUI_FileReader, which has a function for 
            # writing DataFrames of video to .png
            reader = HTPAdGUI_FileReader(TPArray(width = w,
                                                 height = h))
            
            # Pass the DataFrame to the method for writing to .png
            path = Path.cwd() /  (a+'/png')
            reader.export_png(df_video,
                              path,
                              **kwargs)
            
        return path