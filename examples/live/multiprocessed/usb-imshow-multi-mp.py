# -*- coding: utf-8 -*-
"""
Finds all HTPA devices connected via USB and displays each device's dK
stream in its own cv2 window, using processes: one USB_ClientProcess + one
ImshowProcess per device, connected via one SharedMemoryQueue per device.

Threading counterpart: usb-imshow-multi.py

Usage:
    python usb-imshow-multi-mp.py
"""

import multiprocessing as mp
import time

from hspy.ipc.processes_base import SharedMemoryQueue
from hspy.ipc.processes import USB_ClientProcess, ImshowProcess
from hspy.drivers.usb import USB_HTPA


# %% Main
#
# NOTE: everything that touches the USB devices or constructs Process /
# SharedMemoryQueue objects has to live inside this single
# `if __name__ == '__main__':` guard. With the 'spawn' start method every
# child re-imports this module; without the guard each child would search
# for devices and build the whole pipeline again.
if __name__ == '__main__':

    # libusb is not fork-safe: a child forked from a parent that has already
    # used libusb (device search / ArrayType probing below) inherits a broken
    # libusb context. 'spawn' is the default on Windows/macOS anyway, this
    # makes Linux behave the same.
    mp.set_start_method('spawn', force=True)

    #%% Find all devices
    serials = USB_HTPA.find_all_devices()
    if not serials:
        raise SystemExit("No HTPA USB device found!")
    print(f"{len(serials)} device(s) found: {serials}")

    # Windows needs time after dispose_resources() before re-opening
    time.sleep(USB_HTPA.RELEASE_DELAY_S)

    #%% Determine the ArrayType of every device
    # Each ImshowProcess needs its device's ArrayType at construction, but
    # the devices are only opened inside the USB processes. So open each
    # device briefly here (opens, queries 'G', releases it again). A device
    # that fails is skipped, the others still run.
    array_types = {}
    for sn in serials:
        try:
            array_types[sn] = USB_HTPA.probe_array_type(serial_number=sn)
            print(f"[{sn}] ArrayType: {array_types[sn]}")
        except Exception as e:
            print(f"[{sn}] Could not read ArrayType, skipped: {e}")

    if not array_types:
        raise SystemExit("None of the devices could be opened!")

    #%% Create one buffer, USB_ClientProcess and ImshowProcess per device
    usb_processes = []
    plot_processes = []
    buffers = []

    for sn, array_type in array_types.items():

        # Shared-memory buffer between this device's USB process and its
        # Imshow process
        usb_buffer = SharedMemoryQueue(maxsize=4, slot_bytes=4 * 1024 * 1024)
        buffers.append(usb_buffer)

        # USB_ClientProcess: only the driver class and its kwargs cross the
        # process boundary; the device is opened inside the child
        usb_processes.append(USB_ClientProcess(usb_driver_cls=USB_HTPA,
                                               usb_driver_kwargs={'serial_number': sn},
                                               write_buffer=usb_buffer,
                                               stream_cmd=b't',
                                               name=f'usb_process_{sn}'))

        # ImshowProcess: one window per device, named after its serial
        # number
        plot_processes.append(ImshowProcess(name=f'imshow_process_{sn}',
                                            read_buffer=usb_buffer,
                                            ArrayType=array_type,
                                            window_name=f'HTPA {sn}'))

    #%% Start processes
    print('Starting processes...')
    for usb_process, plot_process in zip(usb_processes, plot_processes):
        usb_process.start()
        plot_process.start()
        time.sleep(0.1)           # stagger the device opens / stream starts
    print(f'{len(usb_processes)} device(s) running.')

    try:
        # Block the main process until Enter is pressed
        input('Press Enter to stop...\n')
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        print("Stopping processes...")

        # Stop the PRODUCERS first, while the Imshow processes still read
        # their buffers. Otherwise a USB process could block forever in
        # write_buffer.put() and never reach _teardown(), which stops the
        # device's stream.
        for usb_process in usb_processes:
            usb_process.stop()
        for usb_process in usb_processes:
            usb_process.join(timeout=10)
            if usb_process.is_alive():
                print(f"WARNING: {usb_process.name} did not finish within 10 s - "
                      f"the device may still be streaming.")

        # Then the consumers. Each polls its buffer with a timeout, so
        # stop() is all they need -- no dummy item to wake them up with,
        # and so no background thread to keep that put from blocking on a
        # buffer whose consumer has already gone.
        for plot_process in plot_processes:
            plot_process.stop()
        for plot_process in plot_processes:
            plot_process.join(timeout=5)
            if plot_process.is_alive():
                print(f"WARNING: {plot_process.name} did not finish within 5 s.")

        # Release the shared memory backing the buffers
        for usb_buffer in buffers:
            usb_buffer.close()

        print("All processes stopped.")
