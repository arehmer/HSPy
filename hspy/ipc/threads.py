# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 16:13:39 2024

@author: rehmer
"""
import os

import time
import cv2

from queue import Queue
from threading import Condition

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import pickle as pkl
import csv

from hspytools.ipc.threads_base import WThread,RThread, RWThread
from hspytools.readers import HTPA_UDPReader
from hspytools.tparray import TPArray

from collections import deque



class UDP(WThread):
    """
    Class for running HTPA_UDP_Reader in a thread. Can only bind on device
    at this point
    """
    
    def __init__(self,udp_reader:HTPA_UDPReader,
                 write_buffer:Queue,
                 write_condition:Condition,
                 IP:str = '',
                 DevID:int = -1,
                 Bcast_Addr:str = '',
                 **kwargs):
        """
        Parameters
        ----------
        udp_reader : HTPA_UDP_Reader
            DESCRIPTION.
        IP : str
            DESCRIPTION.
        start_trigger : Event
            DESCRIPTION.
        finish_trigger : Event
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        # self.output_type = kwargs.pop('output_type','np')
        
        # Set UDP Reader object as attribute
        self.udp_reader = udp_reader
        
        # Depending on whether the devices IP or the Device ID along with the 
        # broadcast address are provided, the process of finding and 
        # binding the corresponding htpa device differs
        if (DevID!=-1) and (len(Bcast_Addr)!=0):
            self.udp_reader.broadcast(Bcast_Addr)
            self.udp_reader.bind_tparray(DevID = DevID)
            
            # Save DevID of bound device to attribute
            self.DevID = DevID
        elif len(IP)!=0:
            self.udp_reader.bind_tparray(IP = IP)
            
            # Save DevID of bound device to attribute
            devices = self.udp_reader.devices
            self.DevID = devices[devices['IP']==IP].index.item()
        else:
            print('Either IP or DevID and Bcast_Addr of the device to be ' +\
                  'bound have to be specified!')
        
        # Start continuous bytestream 
        self.udp_reader.start_continuous_bytestream(self.DevID)
        
        # Set image_id counter
        self.image_id = 0
        
        # Set time
        self.t0 = time.time()
        
        super().__init__(target = self._target_function,
                         write_buffer = write_buffer,
                         write_condition = write_condition,
                         **kwargs)
            
    def _target_function(self):
        
        # print('Executed upd thread: ' + str(time.time()-self.t0) )
        
        # Dictionary for storing results in
        result = {}
        
        try:
            # Try to assemble a frame from the udp packages
            frame = self.udp_reader.read_continuous_bytestream(self.DevID)

            # Set success flag and store frame
            result['success'] = True
            result['frame'] = frame    
        except:
            # Set success flag to False in case of failure
            result['success'] = False
        
        # Store image_id
        result['image_id'] = self.image_id
        
        # Store device ID
        result['DevID'] = self.DevID
        
        # Increment image_id
        self.image_id = self.image_id + 1
               
        return result
        
    def stop(self):
        
        # Set attribute exit to stop run method of thread
        self._exit = True
        
        # Stop the stream
        self.udp_reader.stop_continuous_bytestream(DevID = self.DevID)
        
        # Release the array
        self.udp_reader.release_tparray(self.DevID)
        
        
class Imshow(RThread):
    """
    Class for plotting frames and possibly bounding boxes in a thread.
    """
    
    def __init__(self,
                 read_buffer:Queue,
                 read_condition:Condition,
                 ArrayType:int = None,
                 SensorType:int = None,
                 **kwargs):
        
        if ArrayType is None and SensorType is not None:
            self.tparray = TPArray(SensorType = SensorType)
        elif ArrayType is not None and SensorType is None:
            self.tparray = TPArray(ArrayType = ArrayType)
        else:
            raise Exception('Either ArrayType or SensorType must be provided.')
            
        self.num_pix = len(self.tparray._pix)
        
        self.window_name = kwargs.pop('window_name','Sensor stream')
        
        # Set time
        self.t0 = time.time()
        
        # Call parent class
        super().__init__(target = self._target_function,
                         read_buffer = read_buffer,
                         read_condition = read_condition,
                         **kwargs)
    
    def _target_function(self):
        
        # print('Executed imshow thread: ' + str(time.time()-self.t0) )
        
        # Get result from upstream thread
        result = self.read_buffer.get()

        # Check success flag of upstream thread
        if result['success'] == True:

            if 'frame_proc' in list(result.keys()):
                frame = result['frame_proc']
            else:
                frame = result['frame']
    
            # Reshape if not the proper size
            if frame.ndim == 1:
                frame = frame[0:self.num_pix]
                frame = frame.reshape(self.tparray._npsize)
                 
            # convert to opencv type
            frame = cv2.normalize(frame,frame,0,255,cv2.NORM_MINMAX)
            frame = frame.astype(np.uint8)
            
            # Save to dict
            result['frame_plot'] = frame 
            
        else:
            # If upstream thread failed, set success flag to False
            result['success'] = False
            
            print('here')
            
        return result
    
    def run(self):
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        # Check if thread has been stopped
        while self._exit == False:
            
            # Acquire the read condition
            with self.read_condition:
            
                # Wait until the upstream thread notifies this thread
                while self.read_buffer.empty():
                    self.read_condition.wait()
            
                # Execute target function
                result = self._target()
                
                # Notify the upstream thread, that the item has been retrieved
                # from the buffer and processed
                self.read_condition.notify()
                
                # Check success flag of upstream thread
                if result['success'] is True:
                    
                    # Get frame (processed)
                    frame = result['frame_plot']
                    
                    # Convert frame to RGB to be able to plot colored 
                    # boxes
                    frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB)
                    
                    # Get bboxes if available
                    if 'bboxes' in result.keys():
                        bboxes = result['bboxes']
                    
                        # Draw bounding boxes
                        for b in bboxes.index:
                            
                            box = bboxes.loc[[b]]
                        
                            x,y = box['xtl'].item(),box['ytl'].item(),
                            w = box['xbr'].item() - box['xtl'].item()
                            h = box['ybr'].item() - box['ytl'].item()
            
                            frame = cv2.rectangle(frame, (x,y), (x+w,y+h), (255,255,0),1)
                    
                    cv2.imshow(self.window_name,frame)
                    cv2.waitKey(1)
                
                else:
                    pass
    
                # Signal that processing on this item in the read_buffer is done
                # self.read_buffer.task_done()

            # The opencv window needs to be closed inside the run function,
            # otherwise a window with the same name can never be opened until
            # the console is restarted
            if self._exit == True:
                cv2.destroyWindow(self.window_name)
            
    def stop(self):
        self._exit = True


class Record_Thread(RWThread):
    """
    Class for writing a stream of frames along with possible bounding
    boxes in a thread.
    """
    
    def __init__(self,
                 width:int,
                 height:int,
                 read_buffer:Queue,
                 read_condition:Condition,
                 write_buffer:Queue,
                 write_condition:Condition,
                 n_pre_record:int,
                 imshow:bool,
                 **kwargs):
        
        self.tparray = TPArray(width = width, height = height)                  # Array type
        
        self.n_pre_record = n_pre_record                                        # Number of pre-record items
        self.pre_record_buffer = deque(maxlen=n_pre_record)                     # Buffer for pre-record data
        
        self.imshow = imshow                                                    # Show the sensor stream
        self.window_name = kwargs.pop('window_name','Sensor stream')            # Name of the window the stream is shown in
        
        self.recording = False                                                  # Flag to check if recording is active
        self.recorded_data = []                                                 # Store recorded data
        self.recorded_sets = []
        
        self.save_dir = kwargs.pop('save_dir',Path.cwd())                       # Directory to write results and recorded data to
        self.save_keys = ['bboxes','frame']                                     # Keys of values in the received data that are to be written to files
        self.rec_dir = None                                                     # Directory within save_dir, where recorded data is saved. Created automatically.
        self.file_path = {}                                                     # Dictionary of file paths
        
        # Set time
        self.t0 = time.time() 
        
        # Call parent class
        super().__init__(target = self._target_function,
                         read_buffer = read_buffer,
                         read_condition = read_condition,
                         write_buffer = write_buffer,
                         write_condition = write_condition,
                         **kwargs)
    
    def _target_function(self):
        """
        Gets result from upstream thread. Optionally converts the frame to a
        plotable format.

        Returns
        -------
        result : TYPE
            DESCRIPTION.

        """
        
        # Get result from upstream thread
        result = self.read_buffer.get()
        
        # Check success flag of upstream thread
        if result['success'] is True:
            
            if self.imshow == True:
                
                frame = result['frame_proc'].copy()
        
                # Reshape if not the proper size
                if frame.ndim == 1:
                    frame = frame[0:self.num_pix]
                    frame = frame.reshape(self.tparray._npsize)
                     
                # convert to opencv type
                frame = cv2.normalize(frame,frame,0,255,cv2.NORM_MINMAX)
                frame = frame.astype(np.uint8)
                
                # Save to dict
                result['frame_plot'] = frame 
        
        else:
            # If upstream thread failed, set success flag to False
            result['success'] = False
        
        return result
    
    def _start_condition(self,data:dict):
        '''
        Checks if tracks or bounding boxes containing persons exist

        Parameters
        ----------
        data : dict
            Dictionary containing data from upstream threads. It needs to 
            contain a key 'bboxes' with a DataFrame containing bounding
            boxes as value.

        Returns
        -------
        None.

        '''
        
        # Set start flag to False as default
        start = False
        
        # Check if tracks with detected persons exist
        if 'bboxes' in data.keys():
            if len(data['bboxes'])!=0:
                start = True
        
        return start
    
    def _stop_condition(self,data:dict):
        """
        
        Checks if tracks or bounding boxes containing persons exist

        Parameters
        ----------
        data : dict
            Dictionary containing data from upstream threads. It needs to 
            contain a key 'bboxes' with a DataFrame containing bounding
            boxes as value.

        Returns
        -------
        None.

        """
        
        # Set start flag to False as default
        stop = False
        
        # Check if tracks with detected persons exist
        if 'bboxes' in data.keys():
            if len(data['bboxes'])==0:
                stop = True
        
        return stop
        
    def run(self):
        """
        Function that it executed in the thread.

        Returns
        -------
        None.

        """
        
        # Open a cv namedWindow if specified
        if self.imshow == True:
            cv2.namedWindow(self.window_name,
                            cv2.WINDOW_NORMAL)
        
        # Check if thread has been stopped
        while self._exit == False:
            
            # Acquire the read condition
            with self.read_condition:

                # Wait until the upstream thread notifies this thread
                while self.read_buffer.empty():
                    self.read_condition.wait()    

                # Execute target function to get data from upstream thread
                data = self._target()
                               
                ############ Plotting part  ###################################
                if (data['success'] == True) and (self.imshow == True):
                    
                    # Get frame (processed)
                    frame = data['frame_plot']
                    
                    # Get bboxes if available
                    if 'bboxes' in data.keys():
                        bboxes = data['bboxes']
                    
                        # Draw bounding boxes
                        for b in bboxes.index:
                            
                            box = bboxes.loc[[b]]
                        
                            x,y = box['xtl'].item(),box['ytl'].item(),
                            w = box['xbr'].item() - box['xtl'].item()
                            h = box['ybr'].item() - box['ytl'].item()
            
                            frame = cv2.rectangle(frame, (x,y), (x+w,y+h), 1 ,1)
                    
                    cv2.imshow(self.window_name,frame)
                    cv2.waitKey(1)
        
                ############ Recording part #######################################
                # Check if we're currently recording
                if not self.recording:
    
                    # Before recording, keep buffering the incoming data
                    self.pre_record_buffer.append(data)
    
                    # Check the start condition
                    if self._start_condition(data):
                        
                        # Once the starting condition is met, initialize
                        # a new folder and files in it to write to
                        self._initialize_recording_directory()
                        
                        # Write the pre-buffered data to the created files
                        self._write_data_to_files(self.pre_record_buffer)
                        
                        # Clear the pre-recorded buffer
                        self.pre_record_buffer.clear()  
                        
                        # Set recording flag
                        self.recording = True
                        
                        print("Recording started at frame " + str(data['image_id']) + \
                              ". Pre-recorded data included.")
    
                        
                else:
    
                    # During recording, write new data to file immediately
                    self._write_data_to_files([data])
                    
                    # self.recorded_data.append(data)
    
                    # Check if the stop condition is met
                    if self._stop_condition(data):
                        
                        # If so, unset recording flag
                        print("Recording stopped.")
                        self.recording = False
                        
                        # Inform the downstream thread that recorded data is available
                        # Acquire the write condition
                        with self.write_condition:
                            
                            # Write result to buffer
                            self.write_buffer.put({'rec_dir':self.rec_dir})
                            
                            # Notify the downstream thread, that item has been 
                            # placed in the write buffer
                            self.write_condition.notify()
                        
                        # Reset some class attributes
                        self.rec_dir = None
                        self.file_path = {}
                        
                # Notify the upstream thread, that the item has been retrieved
                # from the buffer and processed
                self.read_condition.notify()
        
        # If thread was stopped during recording, put the current data in the
        # write buffer and destroy the cv window if it exists
        if self._exit == True:
            
            if self.recording == True:
                self.write_buffer.put(self.recorded_data) 
                self.recorded_data = []
            
            if self.imshow == True:
                cv2.destroyWindow(self.window_name)
    
    def _initialize_recording_directory(self):
        '''
        Creates a folder in the specified save directory based on the current
        datetime and the ID of the sensor the data is coming from.

        Returns
        -------
        None.

        '''
        # Create a folder to write recorded data to
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%d_%m_%y_%H%M%S")
        DevID = self.pre_record_buffer[-1]['DevID']
        self.rec_dir = self.save_dir / (formatted_datetime + '_' + str(DevID))
        
        self.rec_dir.mkdir(parents=True, exist_ok=True)
        
        # Within that folder, create a file for each entry in the data-dict
        # that is to be written to file
        self.file_path = {}
        for key in self.save_keys:
            # Save data path as attribute
            self.file_path[key] = self.rec_dir / (key+".txt")
            
            # Use touch to create the file, if it doesn't exist
            self.file_path[key].touch()
    
    def _write_data_to_files(self,data:list):
        """
        Writes the values of specified keys (self.save_keys) to corresponding
        files. 

        Parameters
        ----------
        data : list or iterable
            A list containing dictionaries.

        Returns
        -------
        None.

        """
        
        # Loop through the iterable containing the data packages as dicts
        for data_dict in data:
            
            # Loop over keys to write to file
            for key in self.save_keys:
                
                # Check if key is in data dict
                if key in data_dict:
                    
                    # Check if corresponding file is empty
                    file_empty = os.path.getsize(self.file_path[key]) == 0
                            
                    # Parse values behind keys to a header and a numpy array
                    if key == 'bboxes':
                        
                        values = data_dict[key].values.flatten()
                        header = list(data_dict[key].columns)
                        
                    if key == 'frame':
                        
                        # Frame is in the format of a numpy array and needs to 
                        # be parsed to a pandas DataFrame
                        values = data_dict[key]
                        header = self.tparray.get_serial_data_order()
                    
                    # If file was empty, write a header first
                    if file_empty:
                        
                        with open(self.file_path[key],'w',newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow(header)

                    
                    # In any case write values to file
                    if len(values)!=0:
                        with open(self.file_path[key],'a',newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow(values)
               
    
    def stop(self):
        self._exit = True
            


class FileWriter_Thread(RThread):
    """
    Thread for writing data into a file.
    """
    def __init__(self,
                 width:int,
                 height:int,
                 read_buffer:Queue,
                 read_condition:Condition,
                 **kwargs:dict):

        self.tparray = TPArray(width = width, height = height)                  # Array type

        self.save_dir = kwargs.pop('save_dir',Path.cwd())
        
        # Set time
        self.t0 = time.time()
        
        # Call parent class
        super().__init__(target = self._target_function,
                         read_buffer = read_buffer,
                         read_condition = read_condition,
                         **kwargs)
        
        # For debugging, write data to this attribute instead of to file
        self.debug_list = []
        
    def run(self):
        
        # Check if thread has been stopped
        while self._exit == False:
            
            # Acquire the read condition
            with self.read_condition:
                
                # Wait until the upstream thread notifies this thread
                while self.read_buffer.empty():
                    self.read_condition.wait()   
                    
                
                # Get dictionary with data from upstream thread by calling target 
                # function
                data = self._target_function()

                
                # Data is a list of dictionaries, which need to be organized 
                # properly in order to be stored as a file
                organized_data = self._organize_data(data)
                
                try:
                    
                    # Create a folder with an expressive name
                    current_datetime = datetime.now()
                    formatted_datetime = current_datetime.strftime("%d_%m_%y_%H%M")
                    DevID = organized_data.pop('DevID')
                    folder = self.save_dir / (formatted_datetime + '_' + str(DevID))
                    
                    folder.mkdir(parents=True, exist_ok=True)
                    
                    # Save the remaining values in dict (DataFrames) to files
                    for key,df in organized_data.items():
                        file = folder / (key + '.df')
                        pkl.dump(df,open(file,'wb'))
                    
                    # self.read_buffer.task_done()
                    
                except self.read_buffer.empty:
                    continue
                
                # Notify the upstream thread, that the item has been retrieved
                # from the buffer
                self.read_condition.notify()
            
    def _organize_data(self,data:list):
        """
        organize dictionary such that content can be easily written to files

        Parameters
        ----------
        data : dict
            DESCRIPTION.

        Returns
        -------
        None.

        """
                
        # Initialize dictionaries/lists for storing re-organized data in
        frames = {}
        bboxes = []
        frames_proc = {}
        
        # Go through all dictionaries in the list and organize the data 
        for frame_dict in data:
            
            frames[frame_dict['image_id']] = frame_dict['frame']
            
            if 'bboxes' in frame_dict.keys():
                bboxes.append(frame_dict['bboxes'])


            if 'frame_proc' in frame_dict.keys():
                # Processed frame is a 2d array of pixel values
                temp = frame_dict['frame_proc'].reshape((-1,))
                
                # Append PTAT, elOff, etc, ...
                temp = np.hstack((temp,
                                  frame_dict['frame'][len(self.tparray._pix)::] ))
                # frames_proc_temp 
                frames_proc[frame_dict['image_id']] = temp
            
        # Make pandas DataFrames out of all of the dictionaries/lists
        df_frames = pd.DataFrame.from_dict(frames,
                                           orient='index',
                                           columns = self.tparray.get_serial_data_order())
    
        df_frames_proc = pd.DataFrame.from_dict(frames_proc,
                                                orient='index',
                                                columns = self.tparray.get_serial_data_order())
        
        df_bboxes = pd.concat(bboxes)
        
        
        # Put them in dictionary and return
        organized_data = {}
        
        organized_data['DevID'] = frame_dict['DevID']
        organized_data['frames'] = df_frames
        organized_data['frames_proc'] = df_frames_proc
        organized_data['bboxes'] = df_bboxes        
        
        return organized_data 
    
    def _target_function(self):
        
        # print('Executed writer thread: ' + str(time.time()-self.t0) )
        
        # Get result from upstream thread
        upstream_dict = self.read_buffer.get()
    
        return upstream_dict
