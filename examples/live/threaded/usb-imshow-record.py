# -*- coding: utf-8 -*-
"""
Finds all HTPA devices connected via USB, displays each device's dK stream
in its own cv2 window and records on command, using threads: one
USB_Client + one ImshowRecorder thread per device.

Commands (type into the console, then Enter):
    r      start recording (all devices)
    s      stop recording -> each device's recording is written to a .pkl
           file in --dir, while the display keeps running
    Enter  (empty line) quit

Read a recording with:
    from hspy.ipc.threads import load_recording
    meta, frames = load_recording('recordings/imshow_SN_20261016_121500.pkl')

Usage:
    python usb-imshow-record.py [--dir recordings]
"""

from queue import Queue
import argparse
import time

from hspy.ipc.threads import ImshowRecorder, ImshowDisplay, USB_Client
from hspy.drivers.usb import USB_HTPA


# %% Main
if __name__ == '__main__':

    # %% Arguments
    arg_parser = argparse.ArgumentParser(prog='usb-imshow-record.py',
                                         description="Displays all HTPA USB devices and records on command.")
    arg_parser.add_argument("--dir",
                            dest="record_dir",
                            type=str,
                            default='recordings',
                            help="Directory recordings are written to. Default: ./recordings")
    args = arg_parser.parse_args()

    #%% Find all devices
    serials = USB_HTPA.find_all_devices()
    if not serials:
        raise SystemExit("No HTPA USB device found!")
    print(f"{len(serials)} device(s) found: {serials}")

    # Windows needs time after dispose_resources() before re-opening
    time.sleep(USB_HTPA.RELEASE_DELAY_S)

    #%% Create one driver, buffer, USB_Client and ImshowRecorder per device
    # ONE thread for all cv2 windows: OpenCV's window functions are not
    # thread-safe on Linux, so several Imshow threads must not call them
    # directly. New windows are tiled across the screen.
    display = ImshowDisplay(window_size=(640, 480), columns=3)

    usb_drivers = []
    usb_threads = []
    plot_threads = []
    buffers = []

    for sn in serials:
        try:
            usb_driver = USB_HTPA(serial_number=sn)
        except Exception as e:
            print(f"[{sn}] Could not open device, skipped: {e}")
            continue
        print(f"[{sn}] Firmware: {usb_driver.version}, ArrayType: {usb_driver.ArrayType}")
        usb_drivers.append(usb_driver)

        usb_buffer = Queue(maxsize=1)
        buffers.append(usb_buffer)

        usb_threads.append(USB_Client(usb_driver=usb_driver,
                                      stream_cmd=b't',
                                      write_buffer=usb_buffer,
                                      name=f'usb_thread_{sn}'))

        # The thread name is used in the recording's file name
        plot_threads.append(ImshowRecorder(name=f'imshow_{sn}',
                                           read_buffer=usb_buffer,
                                           ArrayType=usb_driver.ArrayType,
                                           window_name=f'HTPA {sn}',
                                           display=display,
                                           record_dir=args.record_dir))

    if not usb_threads:
        raise SystemExit("None of the devices could be opened!")

    #%% Start threads
    print('Starting threads...')
    display.start()
    for usb_thread, plot_thread in zip(usb_threads, plot_threads):
        usb_thread.start()
        plot_thread.start()
        time.sleep(0.1)
    print(f'{len(usb_threads)} device(s) running.')

    #%% Command loop
    try:
        while True:
            cmd = input("Command [r = record, s = stop + save, Enter = quit]: ").strip().lower()

            if cmd == '':
                break

            elif cmd == 'r':
                for plot_thread in plot_threads:
                    if plot_thread.is_recording:
                        print(f"[{plot_thread.name}] already recording")
                    else:
                        plot_thread.start_recording()

            elif cmd == 's':
                for plot_thread in plot_threads:
                    if plot_thread.stop_recording() is None:
                        print(f"[{plot_thread.name}] not recording")

            else:
                print(f"Unknown command {cmd!r}")

    except (KeyboardInterrupt, EOFError):
        pass

    finally:
        print("Stopping threads...")

        # Producers first (each stops its device's stream)
        for usb_thread in usb_threads:
            usb_thread.stop()
        for usb_thread in usb_threads:
            usb_thread.join()

        # Then the display threads. stop() also stops a running recording
        # and starts writing it (save_on_stop=True). Each polls its buffer
        # with a timeout, so stop() is all they need.
        for plot_thread in plot_threads:
            plot_thread.stop()
        for plot_thread in plot_threads:
            plot_thread.join()

        # Wait until all recordings are completely written
        for plot_thread in plot_threads:
            plot_thread.wait_until_saved()

        for usb_driver in usb_drivers:
            usb_driver.close()

        # Display thread last (closes the windows)
        display.stop()
        display.join(timeout=5)

        print("All threads stopped.")