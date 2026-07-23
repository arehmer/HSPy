# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 16:13:39 2024

@author: rehmer
"""
import time
import cv2

from queue import Queue
from threading import Condition

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import pickle as pkl


from hspytools.ipc.threads_base import WThread,RThread, RWThread
from hspytools.readers import HTPA_UDPReader
from hspytools.tparray import TPArray

from collections import deque

from hspy.ipc.threads_base import WThread_R1, RThread, RThread_R1, WThread
from hspytools.readers import HTPA_UDPReader
from hspytools.tparray import TPArray



class UDP(WThread_R1):
    """
    Class for running HTPA_UDP_Reader in a thread. Can only bind one device
    at this point
    """
    
    def __init__(self,
                 udp_reader:HTPA_UDPReader,
                 write_buffer:Queue,
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
        
        super().__init__(name = 'udp_thread',
                         write_buffer = write_buffer,
                         **kwargs)
            
    def _target(self):
        
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
        self._exit.set()
        
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

            # if 'frame_proc' in list(result.keys()):
            #     frame = result['frame_proc']
            # else:
            #     frame = result['frame']
                
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
                    
                    # Scale it up by a factor of 5
                    sf = 10
                    # frame = cv2.resize(frame, (0,0), fx=sf, fy=sf) 
                    frame = np.repeat(np.repeat(frame, sf, axis=0), sf, axis=1)
                    
                    # Convert frame to RGB to be able to plot colored 
                    # boxes
                    frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB)
                    
                    # Get bboxes if available
                    if 'bboxes' in result.keys():
                        bboxes = result['bboxes']
                        
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        fontScale              = 0.4
                        fontColor              = (255,255,0)
                        thickness              = 1
                        lineType               = 2
                                       
                        # Draw bounding boxes
                        for b in bboxes.index:
                                                        
                            box = bboxes.loc[[b]]
                        
                            if box['confirmed'].item() != True:
                                continue
                            
                            x,y = int(box['xtl'].item()),int(box['ytl'].item()),
                            w = int(box['xbr'].item()) - int(box['xtl'].item())
                            h = int(box['ybr'].item()) - int(box['ytl'].item())
                            
                            # Scale coordinates
                            x = int(sf*x); y = int(sf*y); w = int(sf*w); h = int(sf*h)
                            
                            frame = cv2.rectangle(frame,
                                                  (x,y),
                                                  (x+w,y+h),
                                                  (255,255,0),1)
                            
                            # Write score if it exists
                            if box['score'].item() != -99:
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                blxy = (x+1,y+10)
                                fontScale              = 0.4
                                fontColor              = (255,255,0)
                                thickness              = 1
                                lineType               = 2
                                
                                score = f'{box['score'].item():.2}'
                                
                                cv2.putText(frame,
                                            score, 
                                            blxy, 
                                            font, 
                                            fontScale,
                                            fontColor,
                                            thickness,
                                            lineType)
                                
                        # Write the number of persons in the upper left corner
                        num_persons = sum(bboxes['confirmed'] == True)
                        
                        cv2.putText(frame,
                                    f'Number of persons: {num_persons}', 
                                    (0,20), 
                                    font, 
                                    fontScale,
                                    fontColor,
                                    thickness,
                                    lineType)

                            
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


class DummyConsumer(RThread_R1):
    
    def __init__(self,
                 name:str,
                 read_buffer:Queue,
                 **kwargs):
        
        super().__init__(name=name,
                         read_buffer = read_buffer,
                         **kwargs)
    
    def _target(self):
        _ = self.read_buffer.get()
        print('Frame obtained!')
        return None

class DataRecord(RThread_R1):
    """
    Class for writing a stream of frames along with possible bounding
    boxes in a thread.
    """
    
    def __init__(self,
                 name:str,
                 read_buffer:Queue,
                 save_dir:Path,
                 **kwargs):
        
        
        # Check input arguments
        if not isinstance(save_dir,Path):
            raise TypeError(f'save_dir is type {type(save_dir)} instead of {Path}.')
              
        # Call __init__ of parent class
        super().__init__(name = name,
                         read_buffer = read_buffer,
                         **kwargs)
        
        # Assign user specified attributes
        self.save_dir = save_dir                                                # Directory to write results and recorded data to
        self.counter = 0                                                        # File counter, incremented after every write operation
        
        
        # Create a directory within save_dir to store recorded data
        self._init_rec_dir()
       
        
    def _init_rec_dir(self):
        '''
        Create a folder within save_dir, in which the recorded data is stored.
        The create directory is stored as class attribute rec_dir.
        Returns
        -------
        None.

        '''

        # rec_dir is named after the date and time of its creation
        formatted_datetime = datetime.now().strftime("%d_%m_%y_%H%M%S")
        self.rec_dir = self.save_dir / formatted_datetime
        
        # Create rec_dir and all its parents if necessary
        self.rec_dir.mkdir(parents=True,
                           exist_ok=True)
        
   
    def _target(self):
        """
        Gets result from upstream thread. Optionally converts the frame to a
        plotable format.

        Returns
        -------
        result : TYPE
            DESCRIPTION.

        """
        
        # Get result from upstream thread
        data = self.read_buffer.get()
        
        # Acquire timestamp from data, if existing
        if 't' in data.keys():
            timestamp = data['t']
        else:
            timestamp = datetime.now()
            data['t'] = timestamp
        
        # Make a filename
        filename = self.make_filename(self.counter,
                                      timestamp)
        
        # Pickle data in specified directory
        with open(self.rec_dir / filename, "wb") as f:
            pkl.dump(data, f)
            
        # Increment counter
        self.counter += 1
        
        return None

    def make_filename(self,
                      counter: int,
                      timestamp: datetime) -> str:
        
        if not isinstance(timestamp, datetime):
            raise TypeError('timestamp is type {type(timestamp)} instead of ' +\
                            'datetime.datetime.')
        if not isinstance(counter,int):
            raise TypeError('counter is type {type(counter)} instead of int.')
        
        # Convert datetime.datetime to string
        ts = timestamp.strftime("%Y%m%d_%H%M%S_%f")  # %f = microseconds
        
        return f"{counter:06d}_{ts}.pkl"
            
    # def run(self):
    #     """
    #     Function that it executed in the thread.

    #     Returns
    #     -------
    #     None.

    #     """
        
    #     # Open a cv namedWindow if specified
    #     if self.imshow == True:
    #         cv2.namedWindow(self.window_name,
    #                         cv2.WINDOW_NORMAL)
        
    #     # Check if thread has been stopped
    #     while self._exit == False:
            
    #         # Acquire the read condition
    #         with self.read_condition:

    #             # Wait until the upstream thread notifies this thread
    #             while self.read_buffer.empty():
    #                 self.read_condition.wait()    

    #             # Execute target function to get data from upstream thread
    #             data = self._target()
                               
    #             ############ Plotting part  ###################################
    #             if (data['success'] == True) and (self.imshow == True):
                    
    #                 # Get frame (processed)
    #                 frame = data['frame_plot']
                    
    #                 # Get bboxes if available
    #                 if 'bboxes' in data.keys():
                        
    #                     import pickle as pkl
                        
    #                     bboxes = data['bboxes']
                        
    #                     # Draw bounding boxes
    #                     for b in bboxes.index:
                            
    #                         box = bboxes.loc[[b]]
                            
    #                         x,y = int(box['xtl']),int(box['ytl']),
    #                         w = int(box['xbr'] - box['xtl'])
    #                         h = int(box['ybr'] - box['ytl'])
                            

    #                         frame = cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0) ,1)
                    
    #                 cv2.imshow(self.window_name,frame)
    #                 cv2.waitKey(1)
        
    #             ############ Recording part #######################################
    #             # Check if we're currently recording
    #             if not self.recording:
    
    #                 # Before recording, keep buffering the incoming data
    #                 self.pre_record_buffer.append(data)
    
    #                 # Check the start condition
    #                 if self._start_condition(data):
                        
    #                     # Once the starting condition is met, initialize
    #                     # a new folder and files in it to write to
    #                     self._initialize_recording_directory()
                        
    #                     # Write the pre-buffered data to the created files
    #                     self._write_data_to_files(self.pre_record_buffer)
                        
    #                     # Clear the pre-recorded buffer
    #                     self.pre_record_buffer.clear()  
                        
    #                     # Set recording flag
    #                     self.recording = True
                        
    #                     print("Recording started at frame " + str(data['image_id']) + \
    #                           ". Pre-recorded data included.")
    
                        
    #             else:
    
    #                 # During recording, write new data to file immediately
    #                 self._write_data_to_files([data])
                    
    #                 # self.recorded_data.append(data)
    
    #                 # Check if the stop condition is met
    #                 if self._stop_condition(data):
                        
    #                     # If so, unset recording flag
    #                     print("Recording stopped.")
    #                     self.recording = False
                        
    #                     # Inform the downstream thread that recorded data is available
    #                     # Acquire the write condition
    #                     with self.write_condition:
                            
    #                         # Write result to buffer
    #                         self.write_buffer.put({'rec_dir':self.rec_dir})
                            
    #                         # Notify the downstream thread, that item has been 
    #                         # placed in the write buffer
    #                         self.write_condition.notify()
                        
    #                     # Reset some class attributes
    #                     self.rec_dir = None
    #                     self.file_path = {}
                        
    #             # Notify the upstream thread, that the item has been retrieved
    #             # from the buffer and processed
    #             self.read_condition.notify()
        
    #     # If thread was stopped during recording, put the current data in the
    #     # write buffer and destroy the cv window if it exists
    #     if self._exit == True:
            
    #         if self.recording == True:
    #             self.write_buffer.put(self.recorded_data) 
    #             self.recorded_data = []
            
    #         if self.imshow == True:
    #             cv2.destroyWindow(self.window_name)
    

    
    # def _write_data_to_files(self,data:list):
    #     """
    #     Writes the values of specified keys (self.save_keys) to corresponding
    #     files. 

    #     Parameters
    #     ----------
    #     data : list or iterable
    #         A list containing dictionaries.

    #     Returns
    #     -------
    #     None.

    #     """
        
    #     # Loop through the iterable containing the data packages as dicts
    #     for data_dict in data:
            
    #         # Loop over keys to write to file
    #         for key in self.save_keys:
                
    #             # Check if key is in data dict
    #             if key in data_dict:
                    
    #                 # Check if corresponding file is empty
    #                 file_empty = os.path.getsize(self.file_path[key]) == 0
                            
    #                 # Parse values behind keys to a header and a numpy array
    #                 if key == 'bboxes':
                        
    #                     values = data_dict[key].values.flatten()
    #                     header = list(data_dict[key].columns)
                        
    #                 if key == 'frame':
                        
    #                     # Frame is in the format of a numpy array and needs to 
    #                     # be parsed to a pandas DataFrame
    #                     values = data_dict[key]
    #                     header = self.tparray.get_serial_data_order()
                    
    #                 # If file was empty, write a header first
    #                 if file_empty:
                        
    #                     with open(self.file_path[key],'w',newline='') as file:
    #                         writer = csv.writer(file)
    #                         writer.writerow(header)

                    
    #                 # In any case write values to file
    #                 if len(values)!=0:
    #                     with open(self.file_path[key],'a',newline='') as file:
    #                         writer = csv.writer(file)
    #                         writer.writerow(values)

class EMACounting_Thread(RThread_R1):
    """
    Thread that counts the number of confirmed tracks per frame, and returns
    an exponential moving average (EMA) of that number via
        
        s_t = a*x_t + (1-a) * s_t-1
        
        s_t:    Estimated average at time t
        a:      Filter coefficient
        x_t:    Detections at time t
        s_t-1:  Estimated average at time t-1
        
    Useful for counting the number of detected objects, e.g. persons, over time.
    """
    def __init__(self,
                 name : str,
                 read_buffer:Queue,
                 **kwargs):
        
        # self.T0 = T0                      # Desired time constant of EMA filter in seconds
        # self.a = None                     # Filter coefficient
        # self.filt_estimated  = False      # Boolean that indicates whether the filter coefficients have been estimated or not
        
        
        self.N_win = 10                # Number of initial frames from which dT is estimated, which is required to calculate the filter coefficient
        self.tau = 2
        
        self.Win = [np.nan]*self.N_win
        # self.k = 0                        # Counter that is incremented until N_window is reached
        # self.t0 = None                    # Time at receiving first detection result
        # self.dT_samples = []              # List of samples to estimate dT from
        
        # Call constructor of parent class
        super(EMACounting_Thread,self).__init__(name = name,
                                                read_buffer = read_buffer,
                                                **kwargs)
        
    @property
    def a(self):
        return self._a
    @a.setter
    def a(self, a:float):
        self._a = a
        
    @property
    def T0(self):
        return self._T0
    @T0.setter
    def T0(self, T0:int):
        self._T0 = T0

    @property
    def filt_estimated(self):
        return self._filt_estimated
    @filt_estimated.setter
    def filt_estimated(self, filt_estimated:bool):
        self._filt_estimated = filt_estimated
    
    @property
    def N_win(self):
        return self._N_win
    @N_win.setter
    def N_win(self, N_win:int):
        self._N_win = N_win
        
    @property
    def k(self):
        return self._k
    @k.setter
    def k(self, k:int):
        self._k = k
        
        
    def _target(self):
        
        
        # # ---------------------------------------------------------------------
        # # Estimate filter coefficients
        # # ---------------------------------------------------------------------
        
        # # At each call of this target function, sample the time since the last
        # # call to obtain an estimate for the sampling time interval dT
        # if not self.filt_estimated:
        #     if self.k == 0:
        #         self.t0 = time.time()
        #         self.k+=1
        #     elif self.k == 1:
        #         t_now = time.time()
        #         self.dT_samples.append(t_now-self.t0)
        #     elif self.k < self.N_init+1:
        #         t_now = time.time()
        #         self.dT_samples.append(t_now-self.dT_samples[-1])
        
        #     # If desired number of samples is acquired, estimate dT and a
        #     if len(self.dT_samples) == self.N_init:
        #         dT = np.mean(self.dT_samples)
        #         self.a = 1 - np.exp(-dT/self.T0)
                
        
        # ---------------------------------------------------------------------
        # Acquire data from read buffer
        # ---------------------------------------------------------------------
        result = self.read_buffer.get()
        
        # Try to obtain number of confirmed detections
        if result['success'] == True:
            if 'bboxes' in result.keys():
                bboxes = result['bboxes']
                
                n_conf = len(bboxes.loc[bboxes['confirmed']==True])
                
            else:
                n_conf = np.nan
                
        else:
            n_conf = np.nan
            
        # Add obtained confirmed detections to window
        self.Win.append(n_conf)
        self.Win = self.Win[1::]
        
        # Calculate the mean over the whole window
        n_avg = np.nanmean(np.array(self.Win))
        
        # Return the result as a dictionary Dictionary for storing results in
        result['n_avg'] = n_avg
        
        
        print(np.round(n_avg))

               
        return result
    
    

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

