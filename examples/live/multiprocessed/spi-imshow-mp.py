# -*- coding: utf-8 -*-
"""
Multiprocessing counterpart of spi-imshow.py.

Acquires frames from an HTPA160x120dR1 sensor over SPI and displays them
with cv2.imshow -- each stage running in its own OS process instead of a
thread, connected via SharedMemoryQueue instead of queue.Queue.

Note the slot size of the buffer below: a 160x120 array is 19200 pixels, so
one frame dict with the raw digits, the compensated voltages and the dK
image is a few hundred kB once pickled. The 8 MB slots here leave room for
that even with keep_raw=True.

@author: rehmer
"""

import time
from pathlib import Path
import argparse

from hspy.ipc.processes_base import SharedMemoryQueue
from hspy.ipc.processes import SPI_ClientProcess, ImshowProcess
from hspy.drivers.spi import SPI_HTPA160x120dR1


# %% Create an argument parser to enable passing argument from the
# command line to this script
arg_parser = argparse.ArgumentParser(prog = 'spi-imshow-mp.py',
                                     description="Connects to a HTPA device via SPI and shows its frames with cv2.imshow -- multiprocessing version.")

# %% Add arguments using '--key' style
arg_parser.add_argument("--bus",
                        dest = "bus",
                        type=int,
                        default=0,
                        help="SPI bus the HTPA device is connected to (e.g. 0 -> /dev/spidev0.x).")

arg_parser.add_argument("--device",
                        dest = "device",
                        type=int,
                        default=0,
                        help="SPI chip select number (e.g. 0 -> /dev/spidev0.0).")

arg_parser.add_argument("--lut",
                        dest = "lut",
                        type=str,
                        default=None,
                        help="Path to the LuT csv. Without it the frames stay compensated pixel voltages instead of dK.")


# %% Parse arguments
args = arg_parser.parse_args()
bus = args.bus
device = args.device
lut_path = None if args.lut is None else Path(args.lut)


# %% Main
#
# NOTE: unlike the threading version, EVERYTHING that constructs
# Process / SharedMemoryQueue objects has to live inside a single
# `if __name__ == '__main__':` guard. Without it, on platforms/start
# methods that re-import this module in the child (e.g. the 'spawn'
# start method, which is the default on Windows/macOS), every child
# process would itself try to build the whole pipeline again from
# scratch -> runaway recursive process creation.
if __name__ == '__main__':

    #%% Create shared-memory-backed buffers for communication between
    # processes (replaces the plain queue.Queue buffers used between
    # threads)
    spi_buffer = SharedMemoryQueue(maxsize=4, slot_bytes=8 * 1024 * 1024)

    #%% Create SPI_ClientProcess
    # Unlike the threading version, the SPI_HTPA160x120dR1 driver is NOT
    # constructed here in the parent process -- only the class and its
    # constructor kwargs are. SPI_ClientProcess builds the actual driver
    # (opening the spidev handle) inside the child process, since an open
    # file descriptor cannot be safely shared/pickled across a process
    # boundary. LuT loading is likewise deferred: pass the CSV path and
    # let SPI_ClientProcess load and assign it inside the child.
    #
    # How a frame is acquired goes into spi_driver_kwargs, since that is
    # what the driver is built from. eloff_every=8 follows Section 15.3 of
    # the datasheet: the electrical offsets only need a fresh BLIND
    # conversion every 8th to 10th frame, and the driver keeps count
    # itself.
    spi_process = SPI_ClientProcess(spi_driver_cls = SPI_HTPA160x120dR1,
                                    spi_driver_kwargs = {'bus': bus,
                                                         'device': device,
                                                         'eloff_every': 8,
                                                         'applyCalib': True},
                                    write_buffer = spi_buffer,
                                    lut_csv_path = lut_path,
                                    profile = True,
                                    profile_every = 50)


    #%% Create instance of ImshowProcess for plotting using cv2.imshow
    plot_process = ImshowProcess(name='imshow_process',
                                 read_buffer=spi_buffer,
                                 ArrayType=18)        # HTPA160x120dR1

    print('Starting processes...')
    spi_process.start()
    plot_process.start()
    print('Processes started.')

    try:
        # Keep the main process alive and responsive without wasting
        # too many resources, waiting for Ctrl+C
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping processes...")
    finally:
        # Stop in reverse order regardless of how we exit
        plot_process.stop()
        spi_process.stop()

        spi_process.join()
        plot_process.join()

        # Release the shared memory backing the buffers
        spi_buffer.close()

        print("All processes stopped.")
