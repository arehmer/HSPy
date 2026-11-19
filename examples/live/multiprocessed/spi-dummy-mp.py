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
                        help="SPI bus that HTPA device is connected to.")


# %% Parse arguments
args = arg_parser.parse_args()
bus = args.bus

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
    print('Reached main')
    #%% Create shared-memory-backed buffers for communication between
    # processes (replaces the plain queue.Queue buffers used between
    # threads)
    spi_buffer = SharedMemoryQueue(maxsize=4, slot_bytes=4 * 1024 * 1024)

    #%% Create SPI_ClientProcess
    spi_process = SPI_ClientProcess(spi_driver_cls = SPI_HTPA160x120dR1,
                                    spi_driver_kwargs = {},
                                    write_buffer = spi_buffer,
                                    lut_csv_path = None,
                                    lut_offset = 0,
                                    profile = True,
                                    profile_every= 50)


    #%% Create a dummy consumer process
    class dummy_process(RProcess_R1):

        def __init__(self, name, read_buffer):
            super().__init__(name=name,
                             read_buffer=read_buffer)

        def _target(self):
            print('target1')
            data = self._get()
            print(data)
            print('Received data from SPI process: ', data.keys())

    dummy_consumer = dummy_process(name='dummy_process',
                                read_buffer=spi_buffer)

    print('Starting processes...')
    spi_process.start()
    dummy_consumer.start()
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
        dummy_consumer.stop()
        spi_process.stop()

        spi_process.join()
        dummy_consumer.join()

        # Release the shared memory backing the buffers
        spi_buffer.close()

        print("All processes stopped.")
