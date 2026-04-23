# -*- coding: utf-8 -*-
"""
HTPA Device Discovery and Binding Script
----------------------------------------


This script discovers HTPA thermal imaging devices on a network using a user-
provided broadcast address. After detecting devices via a UDP broadcast, the
script interactively prompts the user to select one of the discovered devices.
It then initializes the appropriate ArrayType for that device, starts threads
responsible for receiving UDP packets and visualizing the frames using OpenCV,
keeps them running for approximately 20 seconds, and finally shuts them down
cleanly.


Usage Examples
==============


Windows PowerShell:
python ./StreamI2C.py --bus 1


Linux shell (bash):
python ./StreamI2C.py --bus 1


Command-line Arguments
======================
--bus : int (required)
Broadcast address used to detect HTPA devices on the network.


This script follows standard Python CLI practices, using argparse for argument
parsing and conventional object/thread lifecycle management.
"""


import pickle as pkl
from queue import Queue
import time
from pathlib import Path
import argparse  

from hspytools.readers import HTPA_UDPReader
from hspytools.tparray import TPArray
from hspytools.ipc.threads import UDP,Imshow
from hspytools.ipc.threads_base import RThread_R1
from hspy.drivers.i2c import I2C_HTPA32x32d
from hspy.LuT import LuT

# from hspytools.ipc.threads import UDP, Record_Thread, FileWriter_Thread

from threading import Condition

# %% Create an argument parser to enable passing argument from the
# command line to this script
arg_parser = argparse.ArgumentParser(prog = 'StreamI2C.py',
                                     description="Starts a continuous live-stream of an HTPA device connected via I2C.")

# %% Add arguments using '--key' style
arg_parser.add_argument("--bus",
                        dest = "bus",
                        type=int,
                        required=True,
                        help="I2C bus that HTPA device is connected to.")

# arg_parser.add_argument("--no-imshow",
#                         dest = "imshow",
#                         action="store_false",
#                         required = False,
#                         help="Flag disabling cv2.imshow()")


# %% Parse arguments
args = arg_parser.parse_args()
bus = args.bus

# %% Main loop
if __name__ == '__main__':
    
    # %% Create instance of i2c driver
    i2c_driver = I2C_HTPA32x32d(bus)
    i2c_driver.init()
   
    # Load and assign the LuT
    lut154 = LuT()
    lut154.from_csv('./lut/32x32dL1k7.csv', offset=0)
    i2c_driver.LuT = lut154
    
    
    # Create and set buffer for i2c data
    i2c_buffer = Queue(maxsize=10)
    
    i2c_driver.output_queue = i2c_buffer
    
    
    # Create simple consumer thread to empty the i2c_buffer and prevent blocking
    class DebugReader(RThread_R1):
        def _target(self):
            data = self.read_buffer.get()
            print(data['pix_dK'])
            return None
        
    reader_thread = DebugReader(name='reader_thread',
                                read_buffer = i2c_buffer)
        
    # Assign i2c_buffer to i2c_driver
    
    i2c_driver.start_i2cstream()
    reader_thread.start()
    
    # Let threads run 20 seconds
    time.sleep(3)
    
    # # Stop the threads in reversed order!
    reader_thread.stop()
    i2c_driver.stop_i2cstream()

    # print('End')
        