# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 13:19:51 2026

@author: rehmer

Multiprocessing counterpart of udp-imshow.py.



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
python ./StreamHTPADevice.py --bcast 192.168.178.255


Linux shell (bash):
python ./StreamHTPADevice.py --bcast 192.168.178.255


Command-line Arguments
======================
--bcast : str (required)
Broadcast address used to detect HTPA devices on the network.


This script follows standard Python CLI practices, using argparse for argument
parsing and conventional object/thread lifecycle management.
"""

import multiprocessing as mp
import time
import argparse

from hspy.ipc.processes_base import SharedMemoryQueue
from hspy.ipc.processes import USB_ClientProcess, ImshowProcess
from hspy.drivers.usb import USB_HTPA


import pickle as pkl
from queue import Queue
import time
from pathlib import Path
import argparse  

from hspy.readers import HTPA_UDPReader
from hspy.tparray import TPArray
from hspy.ipc.threads import UDP_Client,Imshow
from hspy.ipc.threads_base import RThread

# from hspytools.ipc.threads import UDP, Record_Thread, FileWriter_Thread

from threading import Condition

# %% Create an argument parser to enable passing argument from the
# command line to this script
arg_parser = argparse.ArgumentParser(prog = 'udp-imshow.py.py',
                                     description="Searches for HTPA devices in a specified subnet and starts a continuous live-stream of a specified device.")

# %% Add arguments using '--key' style
arg_parser.add_argument("--bcast",
                        dest = "bcast",
                        type=str,
                        required=True,
                        help="Broadcast Address of the subnet to search for HTPA devices. Format: xxx.xxx.x.255")

arg_parser.add_argument("--no-imshow",
                        dest = "imshow",
                        action="store_false",
                        required = False,
                        help="Flag disabling cv2.imshow()")


# %% Parse arguments
args = arg_parser.parse_args()
bcast_addr = args.bcast
imshow = args.imshow

# %% Main loop
if __name__ == '__main__':
    
    # %% Create instance of UDP reader
    udp_reader = HTPA_UDPReader()
    
    # %% Broadcast looking for devices
    devices = udp_reader.broadcast(bcast_addr)
    
    # If no devices found, end script
    if len( udp_reader.devices.index) == 0:
        raise ValueError('No devices found.')
    
    # Get user input on which device to bind
    while True:
        try:
            DevID = int(input('Enter the device number of the HTPA device to bind: ' ))
            if DevID in udp_reader.devices.index:
                break
            else:
                print(f'HTPA device with ID {DevID} not discovered.')
        except ValueError:
            print("Please enter a valid integer.")
    
    
    # Get the ArrayType of the selected device
    ArrayType = devices.loc[DevID,'Arraytype']
    
    # Set the ArrayType of the HTPA_UDPReader class
    udp_reader.ArrayType = ArrayType
    
    # Create threads which bind the device, receive the udp packages and 
    # plot the frame
    
    # Create buffers for communication between threads
    udp_buffer = Queue(maxsize=1)

    # Create instance of UDP thread 
    udp_thread = UDP_Client(udp_reader = udp_reader,
                            name = 'udp_thread',
                            DevID = DevID,
                            Bcast_Addr = bcast_addr,
                            write_buffer = udp_buffer)
        
    # Distinguish between the cases of plotting the sensor stream or not
    if imshow: 
        # Create instance of Imshow thread for plotting using cv2.imshow
        plot_thread = Imshow(ArrayType = ArrayType,
                             name = 'imshow_thread',
                             read_buffer = udp_buffer)
            
    elif not imshow:
        # Create a dummy read thread, that only empties the queue and releases 
        # the lock on the writing condition
        def dummy_target():
            udp_buffer.get()  # drain one item from the queue each cycle
        
        plot_thread = RThread(target = dummy_target,
                              read_buffer = udp_buffer)
        
        
        
    # Start the threads
    print('Starting threads...')
    udp_thread.start()
    plot_thread.start()
    print('Threads started.')

    try:
        # Keep the main process alive and responsive without wasting
        # too many resources, waiting for Enter
        input('Press Enter to stop...\n')
        print("Stopping processes...")
        
    except (KeyboardInterrupt, EOFError):
        # Ctrl+C still works as a fallback; EOFError if stdin is closed
        print("\nStopping processes...")
    finally:
        # Stop the threads in reversed order!
        plot_thread.stop()
        udp_thread.stop()
        
        udp_thread.join()          # stops the device stream on its way out
        plot_thread.join()
        
        # udp_reader.close()         # release the USB device
        
        print("All threads stopped.")
    
    print('End')