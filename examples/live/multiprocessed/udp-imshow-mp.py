# -*- coding: utf-8 -*-
"""
Multiprocessing counterpart of udp-imshow.py.

Discovers HTPA devices on a subnet, binds the one the user picks and
displays its frames with cv2.imshow -- each stage running in its own OS
process instead of a thread, connected via SharedMemoryQueue instead of
queue.Queue.

Discovery happens here, in the parent, because it is a question about the
network rather than about one device. The device itself is bound inside the
child process: UDP_ClientProcess builds the UDP_HTPA driver in its _setup(),
since a bound socket must not be handed across a process boundary.

Usage:
    python udp-imshow-mp.py --bcast 192.168.178.255
    python udp-imshow-mp.py --bcast 192.168.178.255 --no-imshow
"""

import argparse

from hspy.ipc.processes_base import SharedMemoryQueue
from hspy.ipc.processes import UDP_ClientProcess, ImshowProcess
from hspy.drivers.udp import UDP_HTPA, HTPA_UDPReader
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
# NOTE: EVERYTHING that constructs Process / SharedMemoryQueue objects has
# to live inside this single `if __name__ == '__main__':` guard. With the
# 'spawn' start method every child re-imports this module; without the
# guard each child would search for devices and build the whole pipeline
# again.
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

    # The IP is what the child binds by, so that it does not have to repeat
    # the 5 s broadcast. The ArrayType is only needed here for the display
    # process -- the driver reads its own off the bind answer.
    IP = udp_reader.devices.loc[DevID, 'IP']
    ArrayType = devices.loc[DevID, 'Arraytype']

    #%% Create shared-memory-backed buffers for communication between
    # processes (replaces the plain queue.Queue buffers used between
    # threads)
    udp_buffer = SharedMemoryQueue(maxsize=4, slot_bytes=4 * 1024 * 1024)

    # Create instance of the UDP process. Unlike the threading version, the
    # UDP_HTPA driver is NOT constructed here in the parent -- only the
    # class and its constructor kwargs are, and the child binds the device
    # (and therefore opens its socket) inside its own _setup().
    udp_process = UDP_ClientProcess(udp_driver_cls = UDP_HTPA,
                                    udp_driver_kwargs = {'IP': IP},
                                    write_buffer = udp_buffer,
                                    name = 'udp_process')


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
