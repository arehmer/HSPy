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

print('Loading libraries...')

from queue import Queue
import time
from pathlib import Path
import argparse  

from hspytools.ipc.threads_base import RThread_R1
from hspy.ipc.threads import DataRecord, DummyConsumer
from hspy.drivers.i2c import I2C_HTPA32x32d
from hspy.LuT import LuT

# from hspytools.ipc.threads import UDP, Record_Thread, FileWriter_Thread

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

arg_parser.add_argument("--save_dir",
                        dest = "save_dir",
                        type=Path,
                        required = False,
                        default = None,
                        help="Directory for storing acquired i2c data")


# %% Parse arguments
args = arg_parser.parse_args()
bus = args.bus
save_dir = args.save_dir.expanduser()
print(f'0: {save_dir}')
# %% Main loop
if __name__ == '__main__':
    
    print('Initializing objects...')
    # %% Create instance of i2c driver
    i2c_driver = I2C_HTPA32x32d(bus)
   
    # Load and assign the LuT
    lut154 = LuT()
    lut154.from_csv(Path.cwd() / 'lut/32x32dL1k7.csv', offset=0)
    i2c_driver.LuT = lut154
    
    
    # Create and set buffer for i2c data
    i2c_buffer = Queue(maxsize=1)
    i2c_driver.output_queue = i2c_buffer
    
    
    # If save_dir is specified, create a DataRecord thread, which writes 
    # the aqcuired data to file
    if save_dir is not None:
        print(f'1: {save_dir}')
        consumer_thread = DataRecord(name = 'record_thread',
                                     read_buffer = i2c_buffer,
                                     save_dir = save_dir)
        
    # If no save_dir is specified, create a dummy consumer thread that only
    # clears the i2c buffer to keep it from blocking
    else:
        consumer_thread = DummyConsumer(name = 'record_thread',
                                        read_buffer = i2c_buffer)
    

        
    # Start threads
    print('Starting threads...')
    i2c_driver.start_i2cstream()
    consumer_thread.start()
    
    # Let threads run 20 seconds
    time.sleep(5)
    
    # # Stop the threads in reversed order!
    print('Shutting down threads...')
    consumer_thread.stop()
    i2c_driver.stop_i2cstream()

        