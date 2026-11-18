# -*- coding: utf-8 -*-
"""
Multiprocessing counterpart of spi-process-udp.py.

Acquires frames from an HTPA sensor over SPI, runs an object-detection
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
from hspy.ipc.processes import SPI_ClientProcess, ImshowProcess
from hspy.drivers.spi import SPI_HTPA160x120dR1
from hspy.ipc.processes_base import RProcess_R1


# %% Create an argument parser to enable passing argument from the
# command line to this script
arg_parser = argparse.ArgumentParser(prog = 'spi-imshow-mp.py',
                                     description="Connects to a HTPA device via SPI, sapplies a processing engine and sends the result via UDP -- multiprocessing version.")

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
    spi_process = SPI_ClientProcess(i2c_driver_cls = I2C_HTPA32x32d,
                                    i2c_driver_kwargs = {'bus': bus},
                                    write_buffer = i2c_buffer,
                                    lut_csv_path = Path.cwd() / 'lut/32x32dL1k7.csv',
                                    lut_offset = 0,
                                    profile = True,
                                    profile_every= 50)


    #%% Create a dummy consumer process
    class dummy_process(RProcess_R1):
        def __init__(self, name, read_buffer, ArrayType):
            super().__init__(name=name,
                             read_buffer=read_buffer,
                             ArrayType=ArrayType)

        def _target(self):
            while not self._stop_event.is_set():
                try:
                    data = self.read_buffer.get(timeout=0.5)
                    print(f"Dummy process received data of length {len(data)}")
                except Exception as e:
                    print(f"Dummy process encountered an error: {e}")
                    break

    dummy_consumer = RProcess_R1(name='dummy_process',
                                read_buffer=i2c_buffer,
                                ArrayType=10)

    print('Starting processes...')
    spi_process.start()
    dummy_consumer.start()
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
        dummy_consumer.stop()
        spi_process.stop()

        spi_process.join()
        dummy_consumer.join()

        # Release the shared memory backing the buffers
        i2c_buffer.close()

        print("All processes stopped.")
