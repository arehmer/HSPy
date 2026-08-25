# -*- coding: utf-8 -*-
"""
UDP Pickle Server Example
--------------------------

Demonstrates UDP_PickleServer: periodically generates a dict containing a
numpy array and a pandas DataFrame and sends it via UDP to a peer running
the matching UDP_PickleClient (see PickleUDPClient.py).

Run this script and PickleUDPClient.py on the same machine (using
127.0.0.1) or on two different machines on the same network.


Usage Examples
==============

Windows PowerShell:
python ./PickleUDPServer.py --target_ip 127.0.0.1 --target_port 5005

Linux shell (bash):
python ./PickleUDPServer.py --target_ip 127.0.0.1 --target_port 5005


Command-line Arguments
======================
--target_ip : str (required)
    IP address of the peer running UDP_PickleClient.
--target_port : int (required)
    Port the peer's UDP_PickleClient is listening on.
--rate : float (optional, default 1.0)
    Number of messages to send per second.
"""

print('Loading libraries...')

import time
import argparse
from queue import Queue

import numpy as np
import pandas as pd

from hspy.ipc.threads import UDP_PickleServer

# %% Create an argument parser to enable passing arguments from the
# command line to this script
arg_parser = argparse.ArgumentParser(prog='PickleUDPServer.py',
                                     description="Periodically sends a dict containing a numpy array and a pandas DataFrame to a UDP_PickleClient peer.")

# %% Add arguments using '--key' style
arg_parser.add_argument("--target_ip",
                        dest="target_ip",
                        type=str,
                        required=True,
                        help="IP address of the peer running UDP_PickleClient.")

arg_parser.add_argument("--target_port",
                        dest="target_port",
                        type=int,
                        required=True,
                        help="Port the peer's UDP_PickleClient is listening on.")

arg_parser.add_argument("--rate",
                        dest="rate",
                        type=float,
                        required=False,
                        default=1.0,
                        help="Number of messages to send per second.")

# %% Parse arguments
args = arg_parser.parse_args()
target_ip = args.target_ip
target_port = args.target_port
rate = args.rate

# %% Main loop
if __name__ == '__main__':

    print('Initializing objects...')

    # Buffer that UDP_PickleServer reads outgoing data from
    send_buffer = Queue(maxsize=1)

    # Create instance of UDP_PickleServer thread
    server_thread = UDP_PickleServer(name='udp_pickle_server',
                                     read_buffer=send_buffer,
                                     target_ip=target_ip,
                                     target_port=target_port)

    print(f'Starting thread, sending to {target_ip}:{target_port}...')
    server_thread.start()

    image_id = 0

    try:
        # Keep generating and sending data until Ctrl+C
        while True:
            # Dict containing a numpy array (e.g. a sensor frame) and a
            # pandas DataFrame (e.g. a table of derived values)
            data = {'image_id': image_id,
                   'frame': np.random.rand(32, 32),
                   'table': pd.DataFrame(np.random.rand(10, 3),
                                         columns=['a', 'b', 'c']),
                   't': time.time()}

            send_buffer.put(data)
            print(f'Sent message {image_id}')

            image_id += 1
            time.sleep(1.0 / rate)
    except KeyboardInterrupt:
        print("\nStopping thread...")
    finally:
        server_thread.stop()
        print("Thread stopped.")

    print('End')
