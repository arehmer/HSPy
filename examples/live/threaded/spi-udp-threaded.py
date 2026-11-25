# -*- coding: utf-8 -*-
"""
Threading counterpart of spi-udp-mp.py.

Acquires frames from an HTPA160x120dR1 over SPI and sends them out via UDP
to a peer running udp-pickle-imshow-mp.py. Same pipeline as spi-udp-mp.py
(acquisition -> processing -> UDP server), but each stage runs in a thread
of one process instead of its own OS process, connected via queue.Queue
instead of SharedMemoryQueue. Useful on single-core boards (e.g. Raspberry
Pi Zero W), where extra processes only add copying and context switches.


Usage Examples
==============

Linux shell (bash):
python ./spi-udp-threaded.py --bus 0 --target_ip 192.168.178.20 --target_port 5005


Command-line Arguments
======================
--bus : int (required)
    SPI bus that the HTPA device is connected to.
--target_ip : str (required)
    IP address of the peer running UDP_PickleClient(Process).
--target_port : int (optional, default 5005)
    Port the peer's UDP_PickleClient(Process) is listening on.

@author: rehmer
"""

from queue import Queue
import argparse

# gpiozero is only defined fo Raspberry Pis
try:
    from gpiozero import DigitalOutputDevice
    from gpiozero.exc import BadPinFactory
    _GPIOZERO_AVAILABLE = True
except ImportError:
    _GPIOZERO_AVAILABLE = False

from hspy.ipc.threads import (SPI_AcquisitionThread, SPI_ProcessingThread,
                              UDP_PickleServer, RThread_R1)
from hspy.drivers.spi import SPI_HTPA160x120dR1

# db
import sys
sys.setswitchinterval(0.0005)
# /db

# %% Create an argument parser to enable passing argument from the
# command line to this script
arg_parser = argparse.ArgumentParser(prog = 'spi-udp-threaded.py',
                                     description="Connects to a HTPA device via SPI, processes the frames and sends the result via UDP -- threading version.")

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
                        default = 5005,
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
if __name__ == '__main__':

    #%% Open the driver here, in the main thread. Its __init__ opens the
    # SPI device, initialises the sensor and reads the calibration.
    spi_driver = SPI_HTPA160x120dR1(bus = bus,
                                    device = 0,
                                    ee_enable = make_ee_enable(pin=8))

    #%% Create buffers for communication between threads
    # Raw frames from acquisition to processing
    raw_buffer = Queue(maxsize=4)

    # Processed frames from processing to the UDP server
    spi_buffer = Queue(maxsize=4)

    #%% Create the SPI pipeline: one thread only acquires raw frames, a
    # second one converts them and applies calibration and LuT, so that the
    # next frame is acquired while the previous one is processed
    spi_thread = SPI_AcquisitionThread(spi_driver = spi_driver,
                                       write_buffer = raw_buffer,
                                       profile = True,
                                       profile_every= 50)

    class ConsumerThread(RThread_R1):
        def __init__(self, name, read_buffer):
            super().__init__(name=name,
                             read_buffer=read_buffer)
            

        def _target(self):
            data = self._get()

    consumer_thread = ConsumerThread(name='consumer_thread',
                                     read_buffer=raw_buffer)   

    # proc_thread = SPI_ProcessingThread(spi_driver = spi_driver,
    #                                    read_buffer = raw_buffer,
    #                                    write_buffer = spi_buffer,
    #                                    lut_csv_path = None,
    #                                    profile = True,
    #                                    profile_every= 50)

    #%% Create instance of UDP_PickleServer, which sends every frame from
    # the processing thread to the peer
    # server_thread = UDP_PickleServer(name='udp_pickle_server_thread',
    #                                  read_buffer=spi_buffer,
    #                                  target_ip=target_ip,
    #                                  target_port=target_port,
    #                                  profile = True,
    #                                  profile_every= 50)

    print('Starting threads...')
    spi_thread.start()
    consumer_thread.start()
    #proc_thread.start()
    #server_thread.start()
    print('Threads started.')


    try:
        # Keep the main thread alive and responsive without wasting
        # too many resources, waiting for Enter
        input('Press Enter to stop...\n')
        print("Stopping threads...")

    except (KeyboardInterrupt, EOFError):
        # Ctrl+C still works as a fallback; EOFError if stdin is closed
        print("\nStopping threads...")

    finally:
        # Stop in reverse order regardless of how we exit
        
        
        #server_thread.stop()
        #proc_thread.stop()
        consumer_thread.stop()
        spi_thread.stop()

        spi_thread.join()
        consumer_thread.join()
        #proc_thread.join()
        #server_thread.join()

        # The driver belongs to this script: put the sensor to sleep and
        # release the SPI device
        spi_driver.close()

        print("All threads stopped.")
