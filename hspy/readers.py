# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 16:08:22 2024

@author: rehmer
"""

import numpy as np
from pathlib import Path
import pandas as pd
import os
import matplotlib
import re
import subprocess
import shutil

import warnings

from .tparray import TPArray



class HTPAdGUI_FileReader():
    
    def __init__(self,tparray = None):
        
        self.tparray = tparray
       
                
    def read_htpa_video(self,path):
        warnings.warn('This function is deprecated and will be deleted in the future. Use import_bdstxt instead.',DeprecationWarning)
        return self.import_bdstxt(path)

    def import_bdstxt(self,path:Path) -> (pd.DataFrame, bytes):
        
        """
        

        Parameters
        ----------
        path : TYPE
            DESCRIPTION.

        Returns
        -------
        df_video : TYPE
            DESCRIPTION.
        header : TYPE
            DESCRIPTION.

        """
        
        # Check if path is provided as pathlib
        if not isinstance(path,Path):
            path = Path(path)
        
        # get suffix
        suffix = path.suffix

        if suffix.casefold() == ('.txt').casefold():
            df_video, header = self._import_txt(path) 
        elif suffix.casefold() == ('.bds').casefold():
            df_video, header = self._import_bds(path)
        else:
            print('File extension not recognized.')
            return None
        
        return df_video, header
        
    def _import_txt(self,path,**kwargs):
        
        # Read the very first line, which contains the header
        with open(path) as file:
            header = file.readline()
        
        # Get ArrayType from header
        arraytype = self._get_header_info(header)['ArrayType']
            
        # Set tparray attribute according to header info 
        self.tparray = TPArray(ArrayType = arraytype)
        
        # Use pandas' read_csv() method to read in the rest of the file
        sep = kwargs.pop('sep',' ')
        skiprows = kwargs.pop('skiprows',1)
        
        # Get columns names
        columns = self.tparray.get_serial_data_order()
       
        # Read txt with pandas to DataFrame
        txt_content = pd.read_csv(path,
                                  sep=sep,
                                  skiprows=skiprows,
                                  header=None)
        
        # txt file contains a time stamp at the end, that's not in the bds-file
        # use only the first columns that contain actual data
        txt_content = txt_content[np.arange(0,len(columns),1)]
        
        # rename columns appropriately
        txt_content.columns = columns

        # rename index appropriately
        txt_content.index = range(0,len(txt_content))
        txt_content.index.name = 'image_id'
                
        return txt_content, header
    
    def _import_bds(self,bds_path,**kwargs):
        
        # open file and save content byte by byte in list
        bds_content = []
        
        with open(bds_path, "rb") as f:
            
            # Read the header byte by byte until '\n'
            header = []
            header_end = False
            
            while header_end == False:
                header.append(f.read(1))

                if header[-1].decode()=='\n':
                    header_end = True
            
            # Join bytes of header together
            header = bytes().join(header)
            
            # Get ArrayType
            arraytype = self._get_header_info(header)['ArrayType']
                
            # Set tparray attribute
            self.tparray = TPArray(ArrayType = arraytype)
            
            # Read two bytes at a time 
            while (LSb := f.read(2)):
                
                # and combine in LSb fashion
                bds_content.append(int.from_bytes(LSb, 
                                                  byteorder='little'))
               
        # Cast the data to a DataFrame of appropriate size
        columns = self.tparray.get_serial_data_order()
        
        # If the last frame has not been fully transmitted, reshaping fails
        # Therefore throw out the last incomplete frame
        num_full_frames = int(len(bds_content)/len(columns))
        bds_content = bds_content[0:num_full_frames*len(columns)]
        bds_content = (np.array(bds_content)).reshape(-1,len(columns))
        bds_content = pd.DataFrame(data=bds_content,
                                   columns = columns)
        
        bds_content.index = range(len(bds_content))
        bds_content.index.name = 'image_id'
        
        return bds_content, header

    def _get_header_info(self,header:bytes):
        
        header_str = str(header)
        
        # Get ArrayType
        match = re.search(r"(?<=ARRAYTYPE=)\d+", header_str)
        arraytype = int(match.group())
        
        return {'ArrayType':arraytype}

    def reverse(self,df_video):
        """
        Function for rotating a video by 180°. Intended for postprocessing 
        a video sequence from a sensor that was mounted upside-down

        Parameters
        ----------
        df_video : pd.DataFrame
            DataFrame containing the content of a text or bds file exported 
            from the HTPAdGUI

        Returns
        -------
        df_video: pd.DataFrame
            Upside-down version of the original video.

        """
        
        # Rotation must be applied to the pixel values and the electrical offsets
        pix_cols = self.tparray._pix
        e_off_cols = self.tparray._e_off
        
        NROFBLOCKS = self.tparray._DevConst['NROFBLOCKS']
        width = self.tparray._width
        height = self.tparray._height
        
        # go through the DataFrame image by image
        for i in df_video.index:
            
            # get pixel and electrical offsets values
            pix_val = df_video.loc[i,pix_cols]
            e_off_val = df_video.loc[i,e_off_cols]
            
            # reshape values according to sensor
            pix_val = pix_val.values.reshape(self.tparray._npsize)
            e_off_val = e_off_val.values.reshape((int(height/NROFBLOCKS),
                                                  width))
            
            # rotate both arrays by 180°
            pix_val = np.rot90(pix_val,k=2)
            e_off_val = np.rot90(e_off_val,k=2)
            
            # reshape to a row vector and write to dataframe
            df_video.loc[i,pix_cols] = pix_val.flatten()
            df_video.loc[i,e_off_cols] = e_off_val.flatten()
            
        # Return the rotated dataframe
        return df_video
    
    def export_bds(self,df_video:pd.DataFrame,header:str,bds_path:Path,**kwargs):
        """
        Export a video sequence in dataframe format to a .bds file that is
        compatible with HTPAdGUI
        

        Parameters
        ----------
        df_video : pd.DataFrame
            DESCRIPTION.
        header : byte or str
            DESCRIPTION.
        path : pathlib.Path
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        mode = kwargs.pop('mode','x')
        
        
        # Check if file already exists
        if bds_path.exists():
            if mode=='x':
                print('File exists. No data will be written. Pass mode="x" to overwrite.')
                return None
            elif mode=='w':
                print('File exists and will be overwritten.')
                os.remove(bds_path)
        
        # first write the header to the list byte by byte
        bds_content = []
        
        if isinstance(header, str):
            header = header.encode()
        
        bds_content.append(header)
        # Go over the video sequence image by image, convert all integers to 
        # bytes and append to the list
        
        for i in df_video.index:
            
            # get the whole row as a row vector
            row = df_video.loc[i].values
            
            # cast every integer to a byte in little endian byteorder
            for val in row:
                bds_content.append(int(val).to_bytes(length=2,
                                                     byteorder='little'))
            
        # Write bytes to file
        with open(bds_path, "wb") as bds_file:
            [bds_file.write(b) for b in bds_content]
            
        return None

    def export_txt(self,df_video:pd.DataFrame,header:str,txt_path:Path,**kwargs):
        """
        Export a video sequence in dataframe format to a .txt file that is
        compatible with HTPAdGUI
        

        Parameters
        ----------
        df_video : pd.DataFrame
            DESCRIPTION.
        header : byte
            DESCRIPTION.
        txt_path : pathlib.Path
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        mode = kwargs.pop('mode','x')
        
        
        # Check if file already exists
        if txt_path.exists():
            if mode=='x':
                print('File exists. No data will be written. Pass mode="x" to overwrite.')
                return None
            elif mode=='w':
                print('File exists and will be overwritten.')
                os.remove(txt_path)
        else:
            # if it doesn't, create it
            with open(txt_path, 'w') as file:
                pass
        
        # Add a column for the timestamp at the very end for compatbility with
        # HTPA GUI
        # df_video['t_string'] = '' 
        
        # Convert header to string if provided as bytes
        if isinstance(header, bytes):
            header = header.decode('utf-8')
        
        # first write the header to the file
        with open(txt_path, 'w') as file:
            header = file.writelines([header])
            

            
        # Then use pandas' to_csv() method to write the rest of the data to the
        # file
        df_video.to_csv(txt_path,
                        sep = ' ',
                        mode='a',
                        header = False,
                        index = False)

            
        return None    
    
    def export_png(self,df_video,path,**kwargs):
        """
        A function for writing a video sequence given as a DataFrame to .png
        frame by frame in a sepcified folder (path)
        

        Parameters
        ----------
        df_video : pd.DataFrame
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        # Get shape of sensor array
        npsize = self.tparray._npsize
        
        # Get columns with pixel values
        pix_cols = self.tparray._pix
        
        # Get rid of everything else
        df_video = df_video[pix_cols]
        
        # Limits for linear scaling
        vmin_perc = kwargs.pop('vmin_perc',2)
        vmax_perc = kwargs.pop('vmax_perc',98)
       
        if not path.exists():
            path.mkdir(parents=True,exist_ok=False)
        
        for i in df_video.index:
            img = df_video.loc[[i]].values.reshape(npsize)

            
            file_name = str(i) + '.png'
            
            # Calculate percentiles for scaling the png
            vmin_i = np.percentile(img, vmin_perc)
            vmax_i = np.percentile(img, vmax_perc)
            
            matplotlib.image.imsave(path / file_name, img,
                                    vmin = vmin_i,
                                    vmax = vmax_i)
        
        return None

    def export_mp4(self,df_video:pd.DataFrame,mp4_path:Path,**kwargs):
        """
        A function for writing a video sequence given as a DataFrame to .mp4
        in a sepcified folder (path)
        

        Parameters
        ----------
        df_video : pd.DataFrame
            DESCRIPTION.

        Returns
        -------
        None.

        """        
        
        # Framerate
        fps = kwargs.pop('fps',8)
        crf = kwargs.pop('crf',5)
        
        # First, write the whole sequence as .png in a temporary folder
        png_folder = mp4_path.parents[0] / 'png_temp'
        
        self.export_png(df_video,png_folder)
        
        
        # Make a video from the .png files using subprocess with ffmpeg
        os.chdir(png_folder)
        subprocess.call([
            'ffmpeg', '-framerate', str(fps),
            '-i', '%d.png',
            '-r', str(fps),
            '-crf',str(crf),
            '-pix_fmt', 'yuv420p',
            mp4_path.as_posix()])
        
        # Remove temporary folder
        shutil.rmtree(png_folder)

        return None   
    
    def _scale_img(self,img):
        
        # Get width an height of image
        w = self.tparray._width
        h = self.tparray._height
        
        # Crop image by 10 % circumferential
        # crop_w = int(np.ceil(0.1*w))
        # crop_h = int(np.ceil(0.1*h))
        
        # Crop the image by 3 pixels to get rid of corners.
        # img_crop = img[crop_h:h-crop_h,crop_w:w-crop_w]
        
        # Use the pixel values in the cropped frame to scale the image
        dK_max  = img.max()
        dK_min  = img.min()
        
        if (dK_max - dK_min) != 0:
            img = ( img - dK_min ) / (dK_max - dK_min)
        
        img[img<=0] = 0


        return img
    
    def _flip(self,df_video):

        w = self.tparray.width
        h = self.tparray.height
        
        pix_cols = self.tparray._pix
        
        for i in df_video.index:
            img = df_video.loc[i,pix_cols].values.reshape((h,w))
            img = np.flip(img,axis=1).flatten()
            df_video.loc[i,pix_cols] = img

            
        return df_video
               
 


# ── Moved to hspy.drivers.udp ────────────────────────────────────────────────
#
# HTPA_UDPReader, HTPA_ByteStream_Converter and HTPA_UDPTimeout used to live in
# this module. They are drivers, not file readers, so they now live next to the
# I2C, SPI and USB drivers in hspy.drivers.udp -- together with UDP_HTPA, the
# single-device driver that UDP_Client / UDP_ClientProcess drive.
#
# Importing them from here keeps working for now and only warns. Resolved
# lazily (PEP 562), so this module does not pull the UDP driver in unless
# something actually asks for it.

_MOVED_TO_DRIVERS_UDP = ('HTPA_UDPReader',
                         'HTPA_ByteStream_Converter',
                         'HTPA_UDPTimeout')


def __getattr__(name):

    if name in _MOVED_TO_DRIVERS_UDP:

        warnings.warn(
            f"hspy.readers.{name} has moved to hspy.drivers.udp. Import it "
            f"from there instead -- 'from hspy.drivers.udp import {name}'. "
            f"This fallback will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2)

        from hspy.drivers import udp

        return getattr(udp, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + list(_MOVED_TO_DRIVERS_UDP))
