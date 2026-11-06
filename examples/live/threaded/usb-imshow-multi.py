# -*- coding: utf-8 -*-
"""
Finds all HTPA devices connected via USB and displays each device's dK
stream in its own cv2 window, using threads: one USB_Client + one Imshow
thread per device.

Multiprocessing counterpart: usb-imshow-multi-mp.py

Usage:
    python usb-imshow-multi.py
"""

from queue import Queue
import time

from hspy.ipc.threads import Imshow, ImshowDisplay, USB_Client
from hspy.drivers.usb import USB_HTPA


# %% Main
if __name__ == '__main__':

    #%% Find all devices
    serials = USB_HTPA.find_all_devices()
    if not serials:
        raise SystemExit("No HTPA USB device found!")
    print(f"{len(serials)} device(s) found: {serials}")

    # Windows needs time after dispose_resources() before re-opening
    time.sleep(USB_HTPA.RELEASE_DELAY_S)

    #%% Create one driver, buffer, USB_Client and Imshow thread per device
    # ONE thread for all cv2 windows: OpenCV's window functions are not
    # thread-safe on Linux, so several Imshow threads must not call them
    # directly. New windows are tiled across the screen.
    display = ImshowDisplay(window_size=(640, 480), columns=3)

    usb_drivers = []          # opened drivers, closed again at the end
    usb_threads = []
    plot_threads = []
    buffers = []

    for sn in serials:

        # __init__ opens the device and calls init() (version, config,
        # ArrayType, frame length). A device that fails to open is skipped,
        # the others still run.
        try:
            usb_driver = USB_HTPA(serial_number=sn)
        except Exception as e:
            print(f"[{sn}] Could not open device, skipped: {e}")
            continue
        print(f"[{sn}] Firmware: {usb_driver.version}, ArrayType: {usb_driver.ArrayType}")
        usb_drivers.append(usb_driver)

        # Buffer between this device's USB_Client and its Imshow thread
        usb_buffer = Queue(maxsize=1)
        buffers.append(usb_buffer)

        # USB_Client thread: starts the stream and repeatedly calls
        # usb_driver.acquire_frame()
        usb_threads.append(USB_Client(usb_driver=usb_driver,
                                      write_buffer=usb_buffer,
                                      name=f'usb_thread_{sn}'))

        # Imshow thread: one window per device, named after its serial
        # number so the windows don't overwrite each other
        plot_threads.append(Imshow(name=f'imshow_thread_{sn}',
                                   read_buffer=usb_buffer,
                                   ArrayType=usb_driver.ArrayType,
                                   window_name=f'HTPA {sn}',
                                   display=display))

    if not usb_threads:
        raise SystemExit("None of the devices could be opened!")

    #%% Start threads
    print('Starting threads...')
    display.start()
    for usb_thread, plot_thread in zip(usb_threads, plot_threads):
        usb_thread.start()
        plot_thread.start()
        time.sleep(0.1)           # stagger the stream starts
    print(f'{len(usb_threads)} device(s) running.')

    try:
        # Block the main thread until Enter is pressed
        input('Press Enter to stop...\n')
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        print("Stopping threads...")

        # Stop the PRODUCERS first: each USB_Client stops its device's
        # stream on its way out (_teardown)
        for usb_thread in usb_threads:
            usb_thread.stop()
        for usb_thread in usb_threads:
            usb_thread.join()

        # Then the consumers. Each polls its buffer with a timeout, so
        # stop() is all they need -- no dummy item to wake them up with.
        for plot_thread in plot_threads:
            plot_thread.stop()
        for plot_thread in plot_threads:
            plot_thread.join()

        # Release the USB devices
        for usb_driver in usb_drivers:
            usb_driver.close()

        # Display thread last (closes the windows)
        display.stop()
        display.join(timeout=5)

        print("All threads stopped.")