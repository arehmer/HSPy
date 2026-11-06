# -*- coding: utf-8 -*-
"""
Connects to a HTPA device via USB and displays the dK stream with cv2.imshow,
using threads (USB_Client + Imshow).

Threading counterpart of usb-imshow-mp.py; USB counterpart of i2c-imshow.py.

Usage:
    python usb-imshow.py                 # first device found
    python usb-imshow.py --serial ABC123 # specific device
"""

from queue import Queue
import time
import argparse

from hspy.ipc.threads import Imshow, USB_Client
from hspy.drivers.usb import USB_HTPA


# %% Main loop
if __name__ == '__main__':

    # %% Create an argument parser to enable passing argument from the
    # command line to this script
    arg_parser = argparse.ArgumentParser(prog='usb-imshow.py',
                                         description="Connects to a HTPA device via USB and displays the stream.")

    # %% Add arguments using '--key' style
    arg_parser.add_argument("--serial",
                            dest="serial",
                            type=str,
                            default=None,
                            help="USB serial number of the HTPA device. Default: first device found.")

    # %% Parse arguments
    args = arg_parser.parse_args()
    serial = args.serial

    # %% Find device if no serial number was given
    if serial is None:
        serials = USB_HTPA.find_all_devices()
        if not serials:
            raise SystemExit("No HTPA USB device found!")
        serial = serials[0]
        print(f"Using device {serial}")
        time.sleep(USB_HTPA.RELEASE_DELAY_S)   # Windows needs time after dispose_resources()

    #%% Create buffers for communication between threads
    usb_buffer = Queue(maxsize=1)           # Buffer that usb thread puts data in

    #%% Create instance of usb driver
    # __init__ already opens the device and calls init() (version, config,
    # ArrayType, frame length), so the driver is ready to stream as soon as
    # it is constructed
    
    usb_driver = USB_HTPA(serial_number=serial,
                          auto_init = True)
    print(f"Firmware: {usb_driver.version}, ArrayType: {usb_driver.ArrayType}")

    # %% Create USB_Client thread, which starts the stream and repeatedly
    # calls usb_driver.acquire_frame()
    usb_thread = USB_Client(usb_driver=usb_driver,
                            write_buffer=usb_buffer)

    #%% Create instance of Imshow thread for plotting using cv2.imshow
    plot_thread = Imshow(name='imshow_thread',
                           read_buffer=usb_buffer,
                           ArrayType=usb_driver.ArrayType)

    print('Starting threads...')
    usb_thread.start()
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
        # Stop in reverse order regardless of how we exit
        plot_thread.stop()
        usb_thread.stop()

        usb_thread.join()          # stops the device stream on its way out
        plot_thread.join()

        usb_driver.close()         # release the USB device

        print("All threads stopped.")