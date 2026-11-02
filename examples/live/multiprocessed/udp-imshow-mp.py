# -*- coding: utf-8 -*-
"""
Multiprocessing counterpart of usb-imshow.py.

Acquires frames from an HTPA device over USB and displays them with
cv2.imshow -- each stage running in its own OS process instead of a
thread, connected via SharedMemoryQueue instead of queue.Queue.

Usage:
    python usb-imshow-mp.py                                 # first device found
    python usb-imshow-mp.py --serial ABC123                 # specific device
    python usb-imshow-mp.py --serial ABC123 --arraytype 10  # skip ArrayType probing
"""

import multiprocessing as mp
import time
import argparse

from hspy.readers import HTPA_UDPReader

from hspy.ipc.processes_base import SharedMemoryQueue
from hspy.ipc.processes import UDP_ClientProcess, ImshowProcess
from hspy.drivers.usb import USB_HTPA
from hspy.ipc.threads_base import RThread

# %% Create an argument parser to enable passing argument from the
# command line to this script
arg_parser = argparse.ArgumentParser(prog = 'udp-imshow-mp.py',
                                     description="Searches for HTPA devices in a specified subnet and starts a continuous live-stream of a specified device.")

# %% Add arguments using '--key' style
arg_parser.add_argument("--bcast",
                        dest = "bcast",
                        type=str,
                        required=True,
                        help="Broadcast Address of the subnet to search for HTPA devices. Format: xxx.xxx.x.255")

arg_parser.add_argument("--no-imshow",
                        dest = "imshow",
                        action = "store_false",
                        required = False,
                        help = "Flag disabling cv2.imshow()")


# %% Parse arguments
args = arg_parser.parse_args()
bcast_addr = args.bcast
imshow = args.imshow


# %% Main
#
# NOTE: EVERYTHING that touches the USB device or constructs Process /
# SharedMemoryQueue objects has to live inside this single
# `if __name__ == '__main__':` guard. With the 'spawn' start method every
# child re-imports this module; without the guard each child would search
# for devices and build the whole pipeline again.
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
    
    #%% Create shared-memory-backed buffers for communication between
    # processes (replaces the plain queue.Queue buffers used between
    # threads)
    udp_buffer = SharedMemoryQueue(maxsize=4, slot_bytes=4 * 1024 * 1024)
    
    # Create instance of UDP thread 
    udp_process = UDP_ClientProcess(udp_reader_cls = HTPA_UDPReader,
                                    udp_reader_kwargs = {'ArrayType': ArrayType},
                                    write_buffer = udp_buffer,
                                    name = 'imshow_process',
                                    DevID = DevID,
                                    Bcast_Addr = bcast_addr)


    # Distinguish between the cases of plotting the sensor stream or not
    if imshow: 
        # Create instance of Imshow thread for plotting using cv2.imshow
        plot_process = ImshowProcess(ArrayType = ArrayType,
                                     name = 'imshow_process',
                                     read_buffer = udp_buffer)
            
    elif not imshow:
        # Create a dummy read thread, that only empties the queue and releases 
        # the lock on the writing condition
        def dummy_target():
            udp_buffer.get()  # drain one item from the queue each cycle
        
        plot_process = RThread(target = dummy_target,
                              read_buffer = udp_buffer)

    print('Starting processes...')
    udp_process.start()
    plot_process.start()
    print('Processes started.')

    try:
        # Keep the main process alive and responsive without wasting
        # too many resources, waiting for Enter
        input('Press Enter to stop...\n')
        print("Stopping processes...")

    except (KeyboardInterrupt, EOFError):
        # Ctrl+C still works as a fallback; EOFError if stdin is closed
        print("\nStopping processes...")
        
    finally:
        # Stop in reverse order regardless of how we exit
        plot_process.stop()
        udp_process.stop()

        udp_process.join()
        plot_process.join()

        # Release the shared memory backing the buffers
        udp_buffer.close()

        print("All processes stopped.")