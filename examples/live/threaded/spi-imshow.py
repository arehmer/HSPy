# -*- coding: utf-8 -*-
"""
Acquires frames from an HTPA160x120dR1 sensor over SPI and displays them
with cv2.imshow, driver and display each in their own thread.

Threading counterpart of spi-imshow-mp.py, and the SPI counterpart of
i2c-imshow.py: the driver is constructed here, in the main thread, and
handed to SPI_Client, which calls its acquire_frame() once per loop
iteration.

Usage:
    python ./spi-imshow.py --bus 0 --device 0
    python ./spi-imshow.py --bus 0 --device 0 --lut lut/160x120dL3k95.csv

@author: rehmer
"""

from queue import Queue
import time
from pathlib import Path
import argparse

from hspy.ipc.threads import Imshow, SPI_Client
from hspy.drivers.spi import SPI_HTPA160x120dR1
from hspy.LuT import LuT


# %% Create an argument parser to enable passing argument from the
# command line to this script
arg_parser = argparse.ArgumentParser(prog = 'spi-imshow.py',
                                     description="Connects to a HTPA device via SPI and shows its frames with cv2.imshow.")

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


# %% Main loop
if __name__ == '__main__':

    #%% Create buffers for communication between threads
    spi_buffer = Queue(maxsize=1)        # Buffer that the spi thread puts data in

    #%% Create instance of the spi driver
    # __init__ already opens the bus and calls init(), so the driver is
    # ready to acquire frames as soon as it is constructed. How a frame is
    # acquired is set here, on the driver:
    #   eloff_every=8  the electrical offsets get a fresh BLIND conversion
    #                  every 8th frame only (Section 15.3), which the driver
    #                  keeps count of itself
    #   applyCalib     every frame carries 'pix_comp'
    # A LuT matching the table number in the flash is loaded from hspy/luts
    # automatically; frames carry 'pix_dK' if there is one.
    spi_driver = SPI_HTPA160x120dR1(bus = bus,
                                    device = device,
                                    eloff_every = 8,
                                    applyCalib = True)

    # A LuT given on the command line replaces the automatic one
    if lut_path is not None:
        lut = LuT()
        lut.from_csv(lut_path)
        spi_driver.LuT = lut

    #%% Create SPI_Client thread, which calls spi_driver.acquire_frame()
    # once per loop iteration
    spi_thread = SPI_Client(spi_driver = spi_driver,
                            write_buffer = spi_buffer)

    #%% Create instance of Imshow thread for plotting using cv2.imshow
    plot_thread = Imshow(name = 'imshow_thread',
                         read_buffer = spi_buffer,
                         ArrayType = spi_driver.ArrayType)

    print('Starting threads...')
    spi_thread.start()
    plot_thread.start()
    print('Threads started.')

    try:
        # Keep the main thread alive and responsive without wasting
        # too many resources, waiting for Ctrl+C
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping threads...")
    finally:
        # Stop in reverse order regardless of how we exit
        plot_thread.stop()
        spi_thread.stop()

        spi_thread.join()
        plot_thread.join()

        spi_driver.close()     # sleep the sensor and release the bus

        print("All threads stopped.")
