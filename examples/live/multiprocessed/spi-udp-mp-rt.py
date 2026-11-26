# -*- coding: utf-8 -*-
"""
Acquires frames from an HTPA160x120dR1 over SPI and sends them out via UDP
to a peer running udp-pickle-imshow-mp.py. Same pipeline as spi-udp-mp.py
(acquisition -> processing -> UDP server, each stage in its own process,
connected via SharedMemoryQueue), set up against the flicker of the top
and bottom rows (block 0):

- warmup: the driver starts every frame with a dummy conversion of block
  0, whose data is discarded. Block 0 settles badly after a conversion of
  block 11 or a BLIND conversion.
- rt_priority: the acquisition process runs at real-time priority
  (SCHED_FIFO). Being a process of its own, it has its own GIL, so neither
  the scheduler nor the processing and UDP stages can delay a conversion
  START: the time between the conversions of a frame stays constant.
- eloff_every=8: fresh electrical offsets every 8th frame only, as
  recommended in Section 15.3 of the datasheet.


Usage Examples
==============

Linux shell (bash):
python ./spi-udp-mp-rt.py --bus 0 --target_ip 192.168.178.20 --target_port 5005


Command-line Arguments
======================
--bus : int (required)
    SPI bus that the HTPA device is connected to.
--target_ip : str (required)
    IP address of the peer running UDP_PickleClient(Process).
--target_port : int (optional, default 5005)
    Port the peer's UDP_PickleClient(Process) is listening on.


Real-time priority (one-time setup)
===================================
The acquisition process asks for the real-time priority SCHED_FIFO. Allow
it for your user once, then log out and in again:

echo "$USER - rtprio 50" | sudo tee /etc/security/limits.d/99-rtprio.conf

Check with 'ulimit -r' (must print 50). Without it the script still runs,
prints a warning and the top and bottom rows may flicker.


Notes
=====
The EE_Enable GPIO is opened here and inherited by the acquisition
process. This needs the 'fork' start method of multiprocessing, the Linux
default up to Python 3.13.

@author: rehmer
"""

import argparse

import lgpio

from hspy.ipc.processes_base import SharedMemoryQueue
from hspy.ipc.processes import (SPI_AcquisitionProcess, SPI_ProcessingProcess,
                                UDP_PickleServerProcess)
from hspy.drivers.spi import SPI_HTPA160x120dR1


# %% Create an argument parser to enable passing argument from the
# command line to this script
arg_parser = argparse.ArgumentParser(prog = 'spi-udp-mp-rt.py',
                                     description="Connects to a HTPA device via SPI, processes the frames and sends the result via UDP -- multiprocessing version with warm-up and a real-time acquisition process.")

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
def make_ee_enable(pin: int = 8, chip: int = 0):
    h = lgpio.gpiochip_open(chip)
    lgpio.gpio_claim_output(h, pin, 1)           # idle HIGH
    write = lgpio.gpio_write
    def ee_enable(level: bool) -> None:
        write(h, pin, level)
    ee_enable.handle = h
    return ee_enable


# %% Main
#
# Everything that constructs Process / SharedMemoryQueue objects lives
# inside this single guard (see spi-udp-mp.py).
if __name__ == '__main__':

    # %% Parse arguments
    args = arg_parser.parse_args()

    #%% Create shared-memory-backed buffers for communication between
    # processes
    # Raw frames (about 42 kB each) from acquisition to processing
    raw_buffer = SharedMemoryQueue(maxsize=4, slot_bytes=256 * 1024)

    # Processed frames from processing to the UDP server
    spi_buffer = SharedMemoryQueue(maxsize=4, slot_bytes=4 * 1024 * 1024)

    #%% The driver is built inside the acquisition process from these
    # keyword arguments
    spi_driver_kwargs = {'bus': args.bus,
                         'device': 0,
                         'ee_enable': make_ee_enable(pin=8),
                         # fresh electrical offsets every 8th frame only
                         'eloff_every': 8,
                         # dummy conversion of block 0 before every frame
                         'warmup': True,
                         # no early start: a late return of the
                         # acquisition loop would then fall between the
                         # warm-up and block 0
                         'early_start': False}

    #%% Create the SPI pipeline: one process only acquires raw frames, at
    # real-time priority (keep it below 50, see the docstring of
    # SPI_ClientProcess), a second one converts them and applies
    # calibration and LuT
    spi_process = SPI_AcquisitionProcess(spi_driver_cls = SPI_HTPA160x120dR1,
                                         spi_driver_kwargs = spi_driver_kwargs,
                                         write_buffer = raw_buffer,
                                         rt_priority = 20,
                                         profile = True,
                                         profile_every= 50)

    proc_process = SPI_ProcessingProcess(processor_queue = spi_process.processor_queue,
                                         read_buffer = raw_buffer,
                                         write_buffer = spi_buffer,
                                         lut_csv_path = None,
                                         profile = True,
                                         profile_every= 50)

    #%% Create instance of UDP_PickleServerProcess, which sends every frame
    # from the processing process to the peer
    server_process = UDP_PickleServerProcess(name='udp_pickle_server_process',
                                             read_buffer=spi_buffer,
                                             target_ip=args.target_ip,
                                             target_port=args.target_port,
                                             profile = True,
                                             profile_every= 50)

    print('Starting processes...')
    spi_process.start()
    proc_process.start()
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
        proc_process.stop()
        spi_process.stop()

        spi_process.join()
        proc_process.join()
        server_process.join()

        # Release the shared memory backing the buffers
        raw_buffer.close()
        spi_buffer.close()

        print("All processes stopped.")
