# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 13:19:51 2026

@author: rehmer

HTPA Device Discovery and Binding Script
----------------------------------------


This script discovers HTPA thermal imaging devices on a network using a user-
provided broadcast address. After detecting devices via a UDP broadcast, the
script interactively prompts the user to select one of the discovered devices.
It then opens a UDP_HTPA driver for that device, starts threads responsible for
receiving UDP packets and visualizing the frames using OpenCV, keeps them
running until Enter is pressed, and finally shuts them down cleanly.

The device's ArrayType does not have to be looked up: UDP_HTPA reads it off the
answer to the bind call and builds its own TPArray from it.


Usage Examples
==============


Windows PowerShell:
python ./udp-imshow.py --bcast 192.168.178.255


Linux shell (bash):
python ./udp-imshow.py --bcast 192.168.178.255


Command-line Arguments
======================
--bcast : str (required)
Broadcast address used to detect HTPA devices on the network.

--no-imshow : flag
Disables cv2.imshow().


This script follows standard Python CLI practices, using argparse for argument
parsing and conventional object/thread lifecycle management.
"""

from queue import Queue
import argparse

from hspy.drivers.udp import UDP_HTPA, HTPA_UDPReader
from hspy.ipc.threads import UDP_Client, Imshow
from hspy.ipc.threads_base import RThread

# %% Create an argument parser to enable passing argument from the
# command line to this script
arg_parser = argparse.ArgumentParser(prog = 'udp-imshow.py',
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

    # %% Discovery is the one job that is not about a single device, so it
    # is still HTPA_UDPReader's: broadcast, then let the user pick.
    udp_reader = HTPA_UDPReader()

    devices = udp_reader.broadcast(bcast_addr)

    # If no devices found, end script
    if len(udp_reader.devices.index) == 0:
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

    # %% Open the driver for the selected device. It binds the device here,
    # in the main thread, and reads its ArrayType off the bind answer -- so
    # nothing below has to be told which sensor this is.
    IP = udp_reader.devices.loc[DevID, 'IP']

    udp_driver = UDP_HTPA(IP = IP)

    print(f'Bound {udp_driver!r}')

    # Create buffers for communication between threads
    udp_buffer = Queue(maxsize=1)

    # Create instance of UDP thread. It starts and stops the stream; the
    # driver itself belongs to this script (see the close() below).
    udp_thread = UDP_Client(udp_driver = udp_driver,
                            name = 'udp_thread',
                            write_buffer = udp_buffer)

    # Distinguish between the cases of plotting the sensor stream or not
    if imshow:
        # Create instance of Imshow thread for plotting using cv2.imshow
        plot_thread = Imshow(ArrayType = udp_driver.ArrayType,
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

        udp_driver.close()         # release the device and close its socket

        print("All threads stopped.")

    print('End')
