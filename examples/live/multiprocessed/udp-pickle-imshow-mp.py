# -*- coding: utf-8 -*-
"""
Receiving end of spi-udp-mp.py.

Listens for the frame dicts a peer sends with UDP_PickleServerProcess,
reassembles and unpickles them in a UDP_PickleClientProcess and displays
them with cv2.imshow in an ImshowProcess -- each stage running in its own
OS process, connected via SharedMemoryQueue.

Not to be confused with udp-imshow-mp.py, which talks to an HTPA device
over its own UDP protocol instead of receiving pickled frames from a peer.

Run this script and spi-udp-mp.py on the same machine (using 127.0.0.1)
or on two different machines on the same network.


Usage Examples
==============

Windows PowerShell:
python ./udp-pickle-imshow-mp.py --listen_ip 0.0.0.0 --listen_port 5005

Linux shell (bash):
python ./udp-pickle-imshow-mp.py --listen_ip 0.0.0.0 --listen_port 5005


Command-line Arguments
======================
--listen_ip : str (optional, default '0.0.0.0')
    Local IP address to listen on.
--listen_port : int (required)
    Local port to listen on. Must match --target_port passed to the
    peer's spi-udp-mp.py.
--arraytype : int (optional, default 18)
    ArrayType of the sensor whose frames are received. The default 18
    is the HTPA160x120dR1 that spi-udp-mp.py reads.

@author: rehmer
"""

import argparse

from hspy.ipc.processes_base import SharedMemoryQueue
from hspy.ipc.processes import UDP_PickleClientProcess, ImshowProcess


# %% Create an argument parser to enable passing argument from the
# command line to this script
arg_parser = argparse.ArgumentParser(prog = 'udp-pickle-imshow-mp.py',
                                     description="Receives frames sent by a UDP_PickleServer(Process) peer and shows them with cv2.imshow -- multiprocessing version.")

# %% Add arguments using '--key' style
arg_parser.add_argument("--listen_ip",
                        dest="listen_ip",
                        type=str,
                        default='0.0.0.0',
                        help="Local IP address to listen on.")

arg_parser.add_argument("--listen_port",
                        dest="listen_port",
                        type=int,
                        required=True,
                        help="Local port to listen on. Must match --target_port passed to the peer's spi-udp-mp.py.")

arg_parser.add_argument("--arraytype",
                        dest="arraytype",
                        type=int,
                        default=18,
                        help="ArrayType of the sensor whose frames are received. Default: 18 (HTPA160x120dR1).")


# %% Parse arguments
args = arg_parser.parse_args()
listen_ip = args.listen_ip
listen_port = args.listen_port
array_type = args.arraytype


# %% Main
#
# NOTE: EVERYTHING that constructs Process / SharedMemoryQueue objects has
# to live inside a single `if __name__ == '__main__':` guard. Without it,
# on platforms/start methods that re-import this module in the child (e.g.
# the 'spawn' start method, which is the default on Windows/macOS), every
# child process would itself try to build the whole pipeline again from
# scratch -> runaway recursive process creation.
if __name__ == '__main__':

    #%% Create shared-memory-backed buffer that UDP_PickleClientProcess
    # writes the received frame dicts into
    udp_buffer = SharedMemoryQueue(maxsize=4, slot_bytes=8 * 1024 * 1024)

    #%% Create instance of UDP_PickleClientProcess, which receives the
    # frames sent by the peer's UDP_PickleServerProcess
    client_process = UDP_PickleClientProcess(name='udp_pickle_client_process',
                                             write_buffer=udp_buffer,
                                             listen_ip=listen_ip,
                                             listen_port=listen_port,
                                             profile = True,
                                             profile_every= 50)

    #%% Create instance of ImshowProcess for plotting using cv2.imshow
    plot_process = ImshowProcess(name='imshow_process',
                                 read_buffer=udp_buffer,
                                 ArrayType=array_type)

    print('Starting processes...')
    client_process.start()
    plot_process.start()
    print('Processes started.')

    # The socket only exists inside the child process, so
    # get_listen_addr() waits for the child to report the bound address
    print(f'Listening on {client_process.get_listen_addr()}...')


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
        client_process.stop()

        client_process.join()
        plot_process.join()

        # Release the shared memory backing the buffer
        udp_buffer.close()

        print("All processes stopped.")
