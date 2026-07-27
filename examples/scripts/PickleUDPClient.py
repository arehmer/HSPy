# -*- coding: utf-8 -*-
"""
UDP Pickle Client Example
--------------------------

Demonstrates UDP_PickleClient: listens for messages sent by a peer running
UDP_PickleServer (see PickleUDPServer.py), reassembles and unpickles them,
and prints a short summary of the received dict (containing a numpy array
and a pandas DataFrame).

Run this script and PickleUDPServer.py on the same machine (using
127.0.0.1) or on two different machines on the same network.


Usage Examples
==============

Windows PowerShell:
python ./PickleUDPClient.py --listen_ip 127.0.0.1 --listen_port 5005

Linux shell (bash):
python ./PickleUDPClient.py --listen_ip 127.0.0.1 --listen_port 5005


Command-line Arguments
======================
--listen_ip : str (optional, default '0.0.0.0')
    Local IP address to listen on.
--listen_port : int (required)
    Local port to listen on. Must match --target_port passed to the
    peer's PickleUDPServer.py.
"""

print('Loading libraries...')

import argparse
from queue import Queue

from hspy.ipc.threads import UDP_PickleClient

# %% Create an argument parser to enable passing arguments from the
# command line to this script
arg_parser = argparse.ArgumentParser(prog='PickleUDPClient.py',
                                     description="Receives dicts containing a numpy array and a pandas DataFrame sent by a UDP_PickleServer peer.")

# %% Add arguments using '--key' style
arg_parser.add_argument("--listen_ip",
                        dest="listen_ip",
                        type=str,
                        required=False,
                        default='0.0.0.0',
                        help="Local IP address to listen on.")

arg_parser.add_argument("--listen_port",
                        dest="listen_port",
                        type=int,
                        required=True,
                        help="Local port to listen on. Must match --target_port passed to the peer's PickleUDPServer.py.")

# %% Parse arguments
args = arg_parser.parse_args()
listen_ip = args.listen_ip
listen_port = args.listen_port

# %% Main loop
if __name__ == '__main__':

    print('Initializing objects...')

    # Buffer that UDP_PickleClient writes received data into
    recv_buffer = Queue(maxsize=1)

    # Create instance of UDP_PickleClient thread
    client_thread = UDP_PickleClient(name='udp_pickle_client',
                                     write_buffer=recv_buffer,
                                     listen_ip=listen_ip,
                                     listen_port=listen_port)

    print('Starting thread...')
    client_thread.start()
    print(f'Listening on {client_thread.listen_addr}...')

    try:
        # Keep printing received messages until Ctrl+C
        while True:
            data = recv_buffer.get()

            frame = data['frame']
            table = data['table']

            print(f"Received message {data['image_id']}: "
                 f"frame shape {frame.shape}, table shape {table.shape}")
    except KeyboardInterrupt:
        print("\nStopping thread...")
    finally:
        client_thread.stop()
        print("Thread stopped.")

    print('End')
