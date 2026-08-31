# -*- coding: utf-8 -*-
"""
Multiprocessing counterpart of i2c-process-udp.py.

Acquires frames from an HTPA sensor over I2C, runs an object-detection
engine on them, and sends the frame + detection result out via UDP to a
peer running udp-imshow-mp.py (or the original udp-imshow.py, since the
wire format hasn't changed) -- each stage running in its own OS process
instead of a thread, connected via SharedMemoryQueue instead of
queue.Queue.

@author: rehmer
"""

import pickle as pkl
import time
from pathlib import Path
import argparse

from hspy.ipc.processes_base import SharedMemoryQueue
from hspy.ipc.processes import I2C_ClientProcess, ImshowProcess
from hspy.drivers.i2c import I2C_HTPA32x32d


# %% Create an argument parser to enable passing argument from the
# command line to this script
arg_parser = argparse.ArgumentParser(prog = 'i2c-imshow-mp.py',
                                     description="Connects to a HTPA device via I2C, sapplies a processing engine and sends the result via UDP -- multiprocessing version.")

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


# %% Main
#
# NOTE: unlike the threading version, EVERYTHING that constructs
# Process / SharedMemoryQueue objects has to live inside a single
# `if __name__ == '__main__':` guard, and there can only be one such
# block (not the two separate ones the threading script used, which
# only happened to work there because thread construction has no
# equivalent re-import hazard). Without this guard, on platforms/start
# methods that re-import this module in the child (e.g. the 'spawn'
# start method, which is the default on Windows/macOS), every child
# process would itself try to build the whole pipeline again from
# scratch -> runaway recursive process creation.
if __name__ == '__main__':

    #%% Create shared-memory-backed buffers for communication between
    # processes (replaces the plain queue.Queue buffers used between
    # threads)
    i2c_buffer = SharedMemoryQueue(maxsize=4, slot_bytes=4 * 1024 * 1024)

    #%% Create I2C_ClientProcess
    # Unlike the threading version, the I2C_HTPA32x32d driver is NOT
    # constructed here in the parent process -- only the class and its
    # constructor kwargs are. I2C_ClientProcess builds the actual driver
    # (opening the SMBus handle) inside the child process, since an open
    # file descriptor cannot be safely shared/pickled across a process
    # boundary. LuT loading is likewise deferred: pass the CSV path and
    # let I2C_ClientProcess load and assign it inside the child.
    i2c_process = I2C_ClientProcess(i2c_driver_cls = I2C_HTPA32x32d,
                                    i2c_driver_kwargs = {'bus': bus},
                                    write_buffer = i2c_buffer,
                                    lut_csv_path = Path.cwd() / 'lut/32x32dL1k7.csv',
                                    lut_offset = 0,
                                    profile = True,
                                    profile_every= 50)


    #%% Create instance of ImshowProcess for plotting using cv2.imshow
    plot_process = ImshowProcess(name='imshow_process',
                                 read_buffer=i2c_buffer,
                                 ArrayType=10)

    print('Starting processes...')
    i2c_process.start()
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
        i2c_process.stop()

        i2c_process.join()
        plot_process.join()

        # Release the shared memory backing the buffers
        i2c_buffer.close()

        print("All processes stopped.")
