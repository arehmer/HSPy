# -*- coding: utf-8 -*-
"""
Multiprocessing counterpart of spi-udp.py.

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

# gpiozero is only defined fo Raspberry Pis
try:
    from gpiozero import DigitalOutputDevice
    from gpiozero.exc import BadPinFactory
    _GPIOZERO_AVAILABLE = True
except ImportError:
    _GPIOZERO_AVAILABLE = False

from hspy.ipc.processes_base import SharedMemoryQueue
from hspy.ipc.processes import SPI_ClientProcess, ImshowProcess, UDP_PickleServerProcess
from hspy.drivers.spi import SPI_HTPA160x120dR1


# %% Create an argument parser to enable passing argument from the
# command line to this script
arg_parser = argparse.ArgumentParser(prog = 'spi-udp-mp.py',
                                     description="Connects to a HTPA device via SPI, sapplies a processing engine and sends the result via UDP -- multiprocessing version.")

# %% Add arguments using '--key' style
arg_parser.add_argument("--bus",
                        dest = "bus",
                        type=int,
                        required=True,
                        help="SPI bus that HTPA device is connected to.")

arg_parser.add_argument("--target_ip",
                        dest="target_ip",
                        type=str,
                        required=True,
                        help="IP address of the peer running UDP_PickleClient(Process).")

arg_parser.add_argument("--target_port",
                        dest="target_port",
                        type=int,
                        required=True,
                        help="Port the peer's UDP_PickleClient(Process) is listening on.")

# Returns a callable that drives the GPIO pin to enable/disable the EEPROM on the HTPA board
def make_ee_enable(pin: int = 8):
    """
    Return an ee_enable(level) callable for SPI_HTPA160x120dR1, driving
    BCM GPIO `pin`. Raises RuntimeError with a clear message on machines
    without gpiozero or without GPIO hardware.
    """
    if not _GPIOZERO_AVAILABLE:
        raise RuntimeError("gpiozero is not installed. Run: pip install gpiozero")
    try:
        ee = DigitalOutputDevice(pin, initial_value=True)   # idle HIGH
    except BadPinFactory as e:
        raise RuntimeError(
            "No GPIO hardware found: gpiozero only drives pins on a "
            "Raspberry Pi (or with a mock pin factory for testing)."
        ) from e

    def ee_enable(level: bool) -> None:
        ee.value = level

    ee_enable.device = ee        # keep a handle, e.g. for ee_enable.device.close()
    return ee_enable

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
    spi_buffer = SharedMemoryQueue(maxsize=4, slot_bytes=4 * 1024 * 1024)

    #%% Create SPI_ClientProcess
    # Driver needs to know which GPIO pin is used to enable the EEPROM on the HTPA board
    spi_driver_kwargs = {'bus': bus,
                         'device' : 0,
                         'ee_enable': make_ee_enable(pin=8)}
    
    spi_process = SPI_ClientProcess(spi_driver_cls = SPI_HTPA160x120dR1,
                                    spi_driver_kwargs = spi_driver_kwargs,
                                    write_buffer = spi_buffer,
                                    lut_csv_path = None,
                                    profile = True,
                                    profile_every= 50)


    #%% Create instance of UDP_PickleServerProcess, which sends every frame
    # from the SPI process to the peer
    server_process = UDP_PickleServerProcess(name='udp_pickle_server_process',
                                             read_buffer=spi_buffer,
                                             target_ip=target_ip,
                                             target_port=target_port,
                                             profile = True,
                                             profile_every= 50)

    print('Starting processes...')
    spi_process.start()
    server_process.start()
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
        server_process.stop()
        spi_process.stop()

        spi_process.join()
        server_process.join()

        # Release the shared memory backing the buffers
        spi_buffer.close()

        print("All processes stopped.")
