# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 09:14:10 2024

@author: rehmer
"""

import pickle as pkl
from queue import Queue
import time
from pathlib import Path
import argparse  

from hspy.ipc.threads import Imshow, I2C_Client
from hspy.drivers.i2c import I2C_HTPA32x32d
from hsod.ipc.threads import ProcessingThread
from hspy.LuT import LuT

from hsod.cv.detectors import build_detector


# %% Create an argument parser to enable passing argument from the
# command line to this script
arg_parser = argparse.ArgumentParser(prog = 'i2c_imshows.py',
                                     description="Connects to a HTPA device via I2C, applies a processing engine and writes the result to a directory.")

# %% Add arguments using '--key' style
arg_parser.add_argument("--bus",
                        dest = "bus",
                        type=int,
                        required=True,
                        help="I2C bus that HTPA device is connected to.")



# %% Parse arguments
args = arg_parser.parse_args()
bus = args.bus
target_ip = args.target_ip
target_port = args.target_port


# %% Main loop
if __name__ == '__main__':
    
    
    #%% Create buffers for communication between threads
    i2c_buffer = Queue(maxsize=1)           # Buffer that i2c thread puts data in
    
    #%% Create instance of i2c driver
    # __init__ already opens the bus and calls init(), so the driver is
    # ready to acquire frames as soon as it is constructed
    i2c_driver = I2C_HTPA32x32d(bus)
    
    # Load and assign the LuT
    lut154 = LuT()
    lut154.from_csv(Path.cwd() / 'lut/32x32dL1k7.csv', offset=0)
    i2c_driver.LuT = lut154
    
    # %% Create I2C_Client thread, which repeatedly calls
    i2c_thread = I2C_Client(i2c_driver = i2c_driver,
                            write_buffer = i2c_buffer)
    
    #%% Create instance of Imshow thread for plotting using cv2.imshow
    plot_thread = Imshow(name = 'imshow_thread',
                         read_buffer = i2c_buffer,
                         ArrayType = 10)
    
if __name__ == '__main__':
     
    print('Starting threads...')  
    i2c_thread.start()
    plot_thread.start()
    print('Threads started.')
    
    try:
        # Keep the main thread alive and responsive without wastin
        # too many ressources, waiting for Ctrl+C
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping threads...")
    finally:
        # Stop in reverse order regardless of how we exit
        plot_thread.stop()
        i2c_thread.stop()
        
        i2c_thread.join()
        plot_thread.join()
        
        print("All threads stopped.")