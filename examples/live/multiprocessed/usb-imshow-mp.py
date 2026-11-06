# -*- coding: utf-8 -*-
"""
Multiprocessing counterpart of usb-imshow.py.

Acquires frames from an HTPA device over USB and displays them with
cv2.imshow -- each stage running in its own OS process instead of a
thread, connected via SharedMemoryQueue instead of queue.Queue.

Usage:
    python usb-imshow-mp.py                                 # first device found
    python usb-imshow-mp.py --serial ABC123                 # specific device
    python usb-imshow-mp.py --serial ABC123 --arraytype 10  # skip ArrayType probing
"""

import multiprocessing as mp
import time
import argparse

from hspy.ipc.processes_base import SharedMemoryQueue
from hspy.ipc.processes import USB_ClientProcess, ImshowProcess
from hspy.drivers.usb import USB_HTPA


# %% Create an argument parser to enable passing argument from the
# command line to this script
arg_parser = argparse.ArgumentParser(prog = 'usb-imshow-mp.py',
                                     description="Connects to a HTPA device via USB and displays the stream -- multiprocessing version.")

# %% Add arguments using '--key' style
arg_parser.add_argument("--serial",
                        dest = "serial",
                        type=str,
                        default=None,
                        help="USB serial number of the HTPA device. Default: first device found.")

arg_parser.add_argument("--arraytype",
                        dest = "arraytype",
                        type=int,
                        default=None,
                        help="ArrayType of the device. Default: read from the device.")


# %% Parse arguments
args = arg_parser.parse_args()
serial = args.serial
array_type = args.arraytype


# %% Main
#
# NOTE: EVERYTHING that touches the USB device or constructs Process /
# SharedMemoryQueue objects has to live inside this single
# `if __name__ == '__main__':` guard. With the 'spawn' start method every
# child re-imports this module; without the guard each child would search
# for devices and build the whole pipeline again.
if __name__ == '__main__':

    # libusb is not fork-safe: a child forked from a parent that has already
    # used libusb (device search / ArrayType probing below) inherits a broken
    # libusb context. 'spawn' starts every child with a fresh interpreter --
    # it is the default on Windows/macOS anyway, this makes Linux behave the
    # same.
    mp.set_start_method('spawn', force=True)

    #%% Find the device if no serial number was given
    if serial is None:
        serials = USB_HTPA.find_all_devices()
        if not serials:
            raise SystemExit("No HTPA USB device found!")
        serial = serials[0]
        print(f"Using device {serial}")
        time.sleep(USB_HTPA.RELEASE_DELAY_S)    # Windows needs time after dispose_resources()

    #%% Determine the ArrayType
    # ImshowProcess needs the ArrayType at construction time, but the device
    # is only opened inside the child process. Unlike the I2C example, where
    # ArrayType=10 is fixed by the sensor, a USB device can be any array, so
    # read it here (opens, queries 'G', releases the device again) unless it
    # was given on the command line.
    if array_type is None:
        array_type = USB_HTPA.probe_array_type(serial_number = serial)
    print(f"ArrayType: {array_type}")

    #%% Create shared-memory-backed buffers for communication between
    # processes (replaces the plain queue.Queue buffers used between
    # threads)
    usb_buffer = SharedMemoryQueue(maxsize=4, slot_bytes=4 * 1024 * 1024)

    #%% Create USB_ClientProcess
    # Unlike the threading version, the USB_HTPA driver is NOT constructed
    # here in the parent process -- only the class and its constructor
    # kwargs are. USB_ClientProcess builds the actual driver (opening the
    # USB device, reading version + config) inside the child process, since
    # an open libusb handle cannot be safely shared/pickled across a process
    # boundary. The child also starts the stream ('K' -> pixel values in dK)
    # and stops it + releases the device again in _teardown().
    usb_process = USB_ClientProcess(usb_driver_cls = USB_HTPA,
                                    usb_driver_kwargs = {'serial_number': serial},
                                    write_buffer = usb_buffer,
                                    stream_cmd = b't',
                                    profile = True,
                                    profile_every = 50)


    #%% Create instance of ImshowProcess for plotting using cv2.imshow
    plot_process = ImshowProcess(name='imshow_process',
                                 read_buffer=usb_buffer,
                                 ArrayType=array_type)

    print('Starting processes...')
    usb_process.start()
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
        usb_process.stop()

        usb_process.join()
        plot_process.join()

        # Release the shared memory backing the buffers
        usb_buffer.close()

        print("All processes stopped.")