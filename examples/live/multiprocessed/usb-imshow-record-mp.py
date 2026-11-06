# -*- coding: utf-8 -*-
"""
Finds all HTPA devices connected via USB, displays each device's dK stream
in its own cv2 window and records on command, using processes: one
USB_ClientProcess + one ImshowRecorderProcess per device.

Threading counterpart: usb-imshow-record.py

Commands (type into the console, then Enter):
    r      start recording (all devices)
    s      stop recording -> each device's recording is written to a .pkl
           file in --dir, while the display keeps running
    Enter  (empty line) quit

Read a recording with:
    from hspy.ipc.processes import load_recording
    meta, frames = load_recording('recordings/imshow_SN_20261016_121500.pkl')

Usage:
    python usb-imshow-record-mp.py [--dir recordings]
"""

import multiprocessing as mp
import argparse
import time

from hspy.ipc.processes_base import SharedMemoryQueue
from hspy.ipc.processes import USB_ClientProcess, ImshowRecorderProcess
from hspy.drivers.usb import USB_HTPA


# %% Main
if __name__ == '__main__':

    # libusb is not fork-safe -> start every child with a fresh interpreter
    mp.set_start_method('spawn', force=True)

    # %% Arguments
    arg_parser = argparse.ArgumentParser(prog='usb-imshow-record-mp.py',
                                         description="Displays all HTPA USB devices and records on command -- multiprocessing version.")
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

    #%% ArrayType of every device (needed by the Imshow processes)
    array_types = {}
    for sn in serials:
        try:
            array_types[sn] = USB_HTPA.probe_array_type(serial_number=sn)
            print(f"[{sn}] ArrayType: {array_types[sn]}")
        except Exception as e:
            print(f"[{sn}] Could not read ArrayType, skipped: {e}")

    if not array_types:
        raise SystemExit("None of the devices could be opened!")

    #%% One buffer, USB_ClientProcess and ImshowRecorderProcess per device
    usb_processes = []
    plot_processes = []
    buffers = []

    for sn, array_type in array_types.items():

        usb_buffer = SharedMemoryQueue(maxsize=4, slot_bytes=4 * 1024 * 1024)
        buffers.append(usb_buffer)

        usb_processes.append(USB_ClientProcess(usb_driver_cls=USB_HTPA,
                                               usb_driver_kwargs={'serial_number': sn},
                                               write_buffer=usb_buffer,
                                               stream_cmd=b't',
                                               name=f'usb_process_{sn}'))

        # The process name is used in the recording's file name
        plot_processes.append(ImshowRecorderProcess(name=f'imshow_{sn}',
                                                    read_buffer=usb_buffer,
                                                    ArrayType=array_type,
                                                    window_name=f'HTPA {sn}',
                                                    record_dir=args.record_dir))

    #%% Start processes
    print('Starting processes...')
    for usb_process, plot_process in zip(usb_processes, plot_processes):
        usb_process.start()
        plot_process.start()
        time.sleep(0.1)
    print(f'{len(usb_processes)} device(s) running.')

    #%% Command loop
    try:
        while True:
            cmd = input("Command [r = record, s = stop + save, Enter = quit]: ").strip().lower()

            if cmd == '':
                break

            elif cmd == 'r':
                for plot_process in plot_processes:
                    if plot_process.is_recording:
                        print(f"[{plot_process.name}] already recording")
                    else:
                        plot_process.start_recording()

            elif cmd == 's':
                for plot_process in plot_processes:
                    if plot_process.stop_recording() is None:
                        print(f"[{plot_process.name}] not recording")

            else:
                print(f"Unknown command {cmd!r}")

    except (KeyboardInterrupt, EOFError):
        pass

    finally:
        print("Stopping processes...")

        # Producers first, while the Imshow processes still read their
        # buffers (each USB process stops its device's stream)
        for usb_process in usb_processes:
            usb_process.stop()
        for usb_process in usb_processes:
            usb_process.join(timeout=10)
            if usb_process.is_alive():
                print(f"WARNING: {usb_process.name} did not finish within 10 s - "
                      f"the device may still be streaming.")

        # Then the Imshow processes. stop() also saves a running recording.
        # Each polls its buffer with a timeout, so stop() is all they need.
        for plot_process in plot_processes:
            plot_process.stop()

        # An Imshow process only exits once its recordings are completely
        # written -- so no short timeout here
        print("Waiting for recordings to be written...")
        for plot_process in plot_processes:
            plot_process.join()

        for usb_buffer in buffers:
            usb_buffer.close()

        print("All processes stopped.")
