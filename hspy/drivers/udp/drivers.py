# -*- coding: utf-8 -*-
"""
UDP drivers for HTPA series devices.

Two classes, with different jobs:

``UDP_HTPA``
    One object = one device, the way ``USB_HTPA`` / ``I2C_HTPA32x32d`` /
    ``SPI_HTPA160x120dR1`` work. It owns a single socket, discovers and binds
    its device in ``open()``, and ``acquire_frame()`` returns the frame
    dictionary documented in ``hspy.drivers._frame``. This is what
    ``UDP_Client`` / ``UDP_ClientProcess`` drive.

``HTPA_UDPReader``
    The original multi-device manager: a DataFrame of discovered devices and
    a dict of sockets, every method taking a ``dev_id``. Still here, unchanged
    in behaviour, for discovery (``broadcast()``) and for scripts that talk to
    several devices through one object. It moved here from ``hspy.readers``,
    which keeps re-exporting it with a ``DeprecationWarning``.

``UDP_HTPA`` deliberately reuses ``HTPA_UDPReader``'s packet-level code --
binding, the call-string parsing, the package assembly loop, and
``HTPA_ByteStream_Converter`` -- instead of reimplementing it. Only the
N-device bookkeeping is what it leaves behind.
"""

import platform
import re
import socket
import time
from datetime import datetime

import numpy as np
import pandas as pd

from hspy.tparray import TPArray
from hspy.drivers._frame import AUX_FROM_DATACOLS, new_frame


class HTPA_UDPTimeout(TimeoutError):
    """
    A bound device sent nothing within the socket's timeout.

    Raised by HTPA_UDPReader.read_continuous_bytestream() so that a caller
    can tell "the device is not answering" apart from "the device sent a
    frame". Subclasses TimeoutError, so code that only catches OSError /
    socket.timeout keeps working.
    """



class HTPA_ByteStream_Converter():
    def __init__(self,ArrayType,**kwargs):

        
        self.tparray = TPArray(ArrayType = ArrayType)
        
        self.width = self.tparray.width
        self.height = self.tparray.height
        
        # Initialize an array to which to write data
        self.data_cols = self.tparray.get_serial_data_order()
        self.data = np.zeros((len(self.data_cols)))
        
        self.output = kwargs.pop('output','np')
        
        # Depending on the specified output format, the convert method is 
        # pointing to _bytes_to_np() or _bytes_to_pd()
        if self.output == 'np':
            self.convert = self._bytes_to_np
        elif self.output == 'pd':
            self.convert = self._bytes_to_pd
            
    
    def _bytes_to_np(self,byte_stream:list):
        
        if not isinstance(byte_stream, list):
            TypeError('byte_stream needs to be a list of bytes')
                
        # Zero data array
        self.data.fill(0)
        
        j=0
        
        # Loop over all elements / packages in list
        for package in byte_stream:
            
            # Read the first byte, it's the package index, and throw away
            _ = package[0]
            
            # Loop over all bytes and combine MSB and LSB
            idx = np.arange(self.tparray.UDP_PackageIndex,len(package),2)
            for i in idx:    
                self.data[j] = int.from_bytes(package[i:i+2], byteorder='little')
                j = j+1


        # Parse the flat data array into a dictionary with the same keys as
        # DataCols. self.data is ordered like DataCols.all(), i.e. the groups
        # follow each other in the order in which DataCols declares them, so
        # every group is just the next len(cols) values. Each slice is copied,
        # so that the returned arrays are no views on self.data, which is
        # overwritten by the next call
        data_dict = {}
        offset = 0

        for group,cols in self.tparray.DataCols.as_dict().items():
            group_data = self.data[offset:offset+len(cols)]
            data_dict[group] = group_data.ravel().copy()
            offset = offset + len(cols)

        return data_dict
    
    def _bytes_to_pd(self,byte_stream:list):
        
        self._bytes_to_np(byte_stream)
        
        df_data = pd.DataFrame(data=[self.data],
                               columns=self.data_cols)
        
        return df_data
        
        
    # def bytes_to_img(self,byte_stream):
        
             
    #     # Loop over all bytes and combine MSB and LSB
    #     idx = np.arange(0,len(byte_stream),2)
        
    #     img = np.zeros((1,self.width*self.height))
    #     j=0
        
    #     for i in idx:    
    #         img[0,j] = int.from_bytes(byte_stream[i:i+2], byteorder='little')
    #         j = j+1
            
    #     img = img.reshape((self.height,self.width))
    #     img = np.flip(img,axis=1)
        
    #     return img
    
    
class HTPA_UDPReader():
    
    def __init__(self,ArrayType=None,**kwargs):
        
        # Set attributes
        self.ArrayType = ArrayType
               
        ###### Set UDP options ################################################
        self._port = kwargs.pop('port',30444)
        self._call_message = bytes('Calling HTPA series devices','utf-8')
        
        # Create a DataFrame to store all bound devices in
        self.col_dict = {'IP':str,'MAC-ID':str,'Arraytype':int,
                         'status':str}
        self.index = pd.Index(data=[],dtype=int,name='DevID')
        self._devices = pd.DataFrame(data=[],
                                     columns = self.col_dict.keys(),
                                     index = self.index)
        
        # Dictionary for storing all sockets in
        self._sockets = {}
            
    @property
    def ArrayType(self):
        return self._ArrayType
    @ArrayType.setter
    def ArrayType(self,ArrayType):
            
        # If the SensorType is given, initialize all attirbutes that
        # depend on knowing the sensor type
        if ArrayType is not None:
            self._initDevSpecificAttr(ArrayType)
                
        self._ArrayType = ArrayType
    
    @property
    def output(self):
        return self._output    
    @property
    def port(self):
        return self._port
    @property
    def tparray(self):
        return self._tparray
    
    
    @property
    def devices(self):
        return self._devices
    @devices.setter
    def devices(self,df):
        
        # Get current devices
        devices_old = self._devices
        
        # Check if that device already exists 
        if df.index[0] in devices_old.index:
            # if it does, replace the existing row
            devices_old.loc[df.index[0]] = df.loc[df.index[0]]
        
        # if it doesn't append the new device
        else:
            self._devices = pd.concat([devices_old,df])
            

    @property
    def sockets(self):
        return self._sockets
    @sockets.setter
    def sockets(self,socket:dict):
        self._sockets.update(socket)

    def _initDevSpecificAttr(self,ArrayType):
        
        # Initialize a TPArray object, which contains all information regarding
        # how data is stored, organized and transmitted for this array type
        self._tparray = TPArray(ArrayType = ArrayType)
        
        self.width = self._tparray.width
        self.height = self._tparray.height
        
   
    def _read_port(self,udp_socket,server_address):
        """
        Read all packages available at the specified port and return the last
        one

        Parameters
        ----------
        server_address : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        clear = False
        
        packages = []
        
        # In order to prevent and endless loop that reads in package after 
        # package (e.g. if Appset is still continuously streaming), read in 
        # a maximum amount of packages that equals 10 whole frames
        p_max = 50 * self.tparray._package_num
        p = 0
                
        while clear == False:
            
            try:
                package = udp_socket.recv(self.tparray._package_size)
                packages.append(package)
                p = p + 1
            except:
                clear = True
                break
            
            if p > p_max:
                clear = True
                break
            
            time.sleep(10/1000)
                   
        return packages
            
    
    def broadcast(self,BROADCAST_ADDRESS:str):
        
        RESPONSE_TIMEOUT = 5  # Wait time in seconds for devices to respond
        
        devices = []

        # Set up a UDP socket for broadcast
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.settimeout(RESPONSE_TIMEOUT)
            
            # Send call message
            print("Calling HTPA devices...")
            sock.sendto(self._call_message, (BROADCAST_ADDRESS, self._port))
                        
            # Listen for responses
            start_time = time.time()
            while time.time() - start_time < RESPONSE_TIMEOUT:
                try:
                    # Try receiving data from socket
                    data, addr = sock.recvfrom(2048)                             # Buffer size of 2048 bytes
                    # print(f"Received response from {addr}: {data}")
                
                    # Check if received data is callstring of an HTPA device
                    if data.startswith(b'HTPA series responsed!'):
                        # If so, try to extract device information from the
                        # call string
                        dev_info = self._callstring_to_information(data)
                        
                        # Set status to 'discovered'
                        dev_info['status'] = 'discovered'
                        
                        # Get the device id
                        DevID = dev_info.index[0]
                        
                        # if the device has already been found or bound,
                        # don't update the device index
                        if DevID in self.devices.index:
                            pass
                        else:
                            # Else do update the index
                            self.devices = dev_info
                        
                except socket.timeout:
                    break  # Timeout reached, stop listening
        
        if len(self.devices)!=0:
            print('Devices found:\n')
            print(self.devices)
        else:
            print('No devices found.')
        
        return self.devices
    
    def bind_tparray(self,IP:str='',DevID:int=-1):
        """
        Binds an HTPA device which is specified either via its IP or its DevID.
        If a device is specified via its DevID, it must have been discovered
        via broadcasting beforehand and therefore exist in the device index
        self.devices

        Parameters
        ----------
        **kwargs : dict
            DESCRIPTION.

        Returns
        -------
        None.

        """

        if DevID!=-1:
            
            # if the DevID is provided, look if a corresponding device has
            # already been discovered
            
            # Check if the specified device exists
            if DevID not in self.devices.index:
                raise Exception('Device ' + str(DevID) + ' seems to have not been ' +\
                                'discovered yet. Discover it via broadcasting first.')
            
            
            # Get the corresponding IP
            IP = self.devices.loc[DevID,'IP']
                        
            # And bind it
            self._bind_tparray(IP)
            
        elif len(IP)!=0:
            self._bind_tparray(IP)
        else:
            print('Either IP or DevID of the device to be bound have to ' +\
                  'be specified!')

        
    def _bind_discovered_htpa(self,DevID:int):
        """
        Binds an already discored htpa device which is specified via its
        DevID.

        Parameters
        ----------
        DevID : int
            DESCRIPTION.

        Returns
        -------
        None.

        """
        

        
        # Get a copy of the row corresponding to this device (as a pd.Series)
        dev_info = self.devices.loc[[DevID]].copy()
        
        # Get the devices IP
        IP = dev_info.loc[DevID,'IP']
        
        
        # Create the udp socket
        udp_socket = socket.socket(socket.AF_INET,          # Internet
                                   socket.SOCK_DGRAM)       # UDP
        
        # Allow re-using socket addresses in case of crashes
        udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # Bind socket
        udp_socket.bind(('0.0.0.0', self._port))
        
        # Create server address
        server_address = (IP,self._port)
        
        # Connect
        udp_socket.connect(server_address)   
               
        # Set timeout to 1 second
        udp_socket.settimeout(2)
        
        # Stop any stream that might still continue, e.g. if program 
        # crashed 
        _ = self.stop_continuous_bytestream(IP = IP)
        
        # Next try to bind the device that answered the call        
        try:
            _ = udp_socket.sendto(bytes('Bind HTPA series device',
                                        'utf-8'),
                                  server_address)
            
            # Read the answer to the bind command from socket
            answer = self._read_port(udp_socket,server_address)

        except:
            raise Exception('Calling HTPA series device failed')
        
        dev_info.loc[DevID,'status'] = 'bound'
        
        # Add socket to dictionary
        self.sockets = {DevID:udp_socket}
        
        # Append new device to device list and store
        self.devices = dev_info
        
        print('Bound HTPA device with DevID: ' + str(DevID) )

    
    
    def _bind_undiscovered_htpa(self,IP:str):
        """
        Tries to call and bind an htpa device under the specified IP address.

        Parameters
        ----------
        IP : str
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        self._bind_tparray(IP)
    
    def _bind_tparray(self,ip:str):
        """
        Creates a socket for the device with the given ip

        Parameters
        ----------
        ip : int
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # Create the udp socket
        udp_socket = socket.socket(socket.AF_INET,          # Internet
                                   socket.SOCK_DGRAM)       # UDP
        
        # Allow re-using socket addresses in case of crashes
        udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # Bind socket (sometimes needed, sometimes breaks the code)
        if platform.system() == "Linux":
            udp_socket.bind(('0.0.0.0', self._port))
                 
        # Set timeout to 1 second
        udp_socket.settimeout(1)
        
        # Create server address
        server_address = (ip,self._port)
        
        # Connect
        udp_socket.connect(server_address)   
                
        # Try calling the device and check if it's a HTPA device

        # Stop any stream that might still continue, e.g. if program 
        # crashed 
        _ = self.stop_continuous_bytestream(IP = ip)
        
        # Try to call device
        test = udp_socket.sendto(bytes('Calling HTPA series devices',
                                    'utf-8'),
                                    server_address)

        # The package following the call should contain device information
        call = self._read_port(udp_socket,server_address)
        
        # If calling was successfull, extract basic information from the 
        # answer string
        
        call_fail = False
        
        if len(call) == 1:
            try:
                dev_info = self._callstring_to_information(call[0])
            except:
                call_fail = True 
        else:
            call_fail = True
            
        if call_fail == True:
            raise Exception('Calling HTPA series device failed')
            return None
        
        # Next try to bind the device that answered the call        
        try:
            _ = udp_socket.sendto(bytes('Bind HTPA series device',
                                        'utf-8'),
                                       server_address)
            
            # Read the answer to the bind command from socket
            _ = self._read_port(udp_socket,server_address)
        except:
            raise Exception('Calling HTPA series device failed')
            return None
        
        dev_info['status'] = 'bound'
        
        # Add socket to dictionary
        dev_id = dev_info.index.item()

        self.sockets = {dev_id:udp_socket}
        
        # Append new device to device list and store
        self._devices = dev_info
        
        print('Bound HTPA device with DevID: ' + str(dev_id) )
        
        return self.devices.copy()

    def release_tparray(self,DevID:int):
        
        # Get device information from the device list
        dev_info = self.devices.loc[[DevID]]
        
        # If more than one devices have the same device id, return error
        if len(dev_info) != 1:
            Exception('Multiple devices have the same device id.')
            
        # Get udp socket
        udp_socket = self.sockets[DevID]
        
        # Create server address
        server_address = (dev_info['IP'].item(), self.port)
        
        # Send message to device to stop streaming bytes
        for i in range(5):
            udp_socket.sendto(bytes('X','utf-8'),server_address)
            time.sleep(10/1000)
        
        # Clean up port
        answ = self._read_port(udp_socket,server_address)
        
        # Send message to release device
        _ = udp_socket.sendto(bytes('x Release HTPA series device','utf-8'),
                              server_address)
        
        # Clean up port
        answ = self._read_port(udp_socket,server_address)
        
        # Release socket
        udp_socket.close()
        
        # Delete socket from dict
        del self.sockets[DevID]
        
        print('Released HTPA device with DevID: ' + str(DevID) )
            
        return answ    
            
    def start_continuous_bytestream(self,dev_id,cmd=b'K'):
        """
        Parameters
        ----------
        dev_id : int
            DevID of a bound device.
        cmd : bytes
            Command that starts the stream. b'K' (default) streams pixel
            values in dK. Configurable so that UDP_Client /
            UDP_ClientProcess can expose the same stream_cmd argument as
            their USB counterparts.
        """

        # Get device information from the device list
        dev_info = self.devices.loc[[dev_id]]

        # If more than one devices have the same device id, return error
        if len(dev_info) != 1:
            Exception('Multiple devices have the same device id.')

        # Get udp socket
        udp_socket = self.sockets[dev_id]

        # Create server address
        server_address = (dev_info['IP'].item(), self.port)

        # Send message to device to start streaming bytes
        cmd = cmd.encode('utf-8') if isinstance(cmd,str) else bytes(cmd)
        _ = udp_socket.sendto(cmd,server_address)

    def stop_continuous_bytestream(self,**kwargs):
        
        dev_id = kwargs.pop('DevID',None)
        ip = kwargs.pop('IP',None)
        
        if (dev_id is None) and (ip is None):
            raise Exception('Either DevID or IP must be provided!')
        
        # Create an UDP socket and a server address
        # If DevID is provided 
        if dev_id is not None:
            # Get device information from the device list
            dev_info = self.devices.loc[[dev_id]]
            
            # If more than one devices have the same device id, return error
            if len(dev_info) != 1:
                Exception('Multiple devices have the same device id.')
            
            # Get udp socket
            udp_socket = self.sockets[dev_id]
                    
            # Create server address
            server_address = (dev_info['IP'].item(), self.port)
        
        # If IP is provided 
        elif ip is not None:
            # Create the udp socket
            udp_socket = socket.socket(socket.AF_INET,          # Internet
                                       socket.SOCK_DGRAM)       # UDP
            
            # Create server address
            server_address = (ip,self._port)
            
            # Set timeout to 1 second
            # socket.setdefaulttimeout(1)
            udp_socket.settimeout(1)
                
        # Send message to device to stop streaming bytes
        for i in range(5):
            udp_socket.sendto(bytes('X','utf-8'),server_address)
            time.sleep(10/1000)
            udp_socket.sendto(bytes('x','utf-8'),server_address)
            time.sleep(10/1000)
        
        # Clean up port by reading all old packages from it
        answ = self._read_port(udp_socket,server_address)
        
        return answ
    
    def read_continuous_bytestream(self,dev_id,first_frame_timeout=None,
                                   return_raw=False):
        """
        Assemble one frame from the continuous bytestream of a bound
        device.

        Parameters
        ----------
        dev_id : int
            DevID of a bound device.
        first_frame_timeout : float, optional
            Timeout in seconds for the read that waits for the device to
            start sending (the sync read, see below). The socket's own
            timeout is sized for a device that is already streaming,
            which a device that was just handed its stream command is
            not: it can take seconds before the first package appears.
            Pass this while waiting for the first frame to keep that
            delay from looking like a lost device. None (default) leaves
            the socket's timeout alone.
        return_raw : bool
            If True, return ``(frame, packages)`` instead of just the
            frame, where ``packages`` is the list of UDP datagrams the
            frame was assembled from. Used by ``UDP_HTPA`` when it was
            constructed with ``keep_raw=True``; the packages roughly
            double the size of a frame once pickled, so this is off by
            default.

        Raises
        ------
        HTPA_UDPTimeout
            If the device sent nothing within the socket's timeout. This
            used to return a frame of zeros instead, which made an
            unplugged device indistinguishable from a working one sending
            black frames -- UDP_Client / UDP_ClientProcess rely on the
            exception to notice that the device is gone.
        """

        # Get device information from the device list
        dev_info = self.devices.loc[[dev_id]]
        
        # Initialize ByteStream Reader for convertings bytes to pandas
        # dataframes
        bytestream_converter = \
            HTPA_ByteStream_Converter(ArrayType=self.ArrayType)
        
        # If more than one device has the same device id, return error
        if len(dev_info) != 1:
            Exception('Multiple devices have the same device id.')
        
        # Get udp socket
        udp_socket = self.sockets[dev_id]

        # The socket's timeout applies between two packages of a stream
        # that is running. Waiting for a device to START sending is a
        # different kind of wait, and a much longer one, so it gets its
        # own timeout if the caller asked for one. Restored below before
        # this method returns or raises, so the stream itself keeps
        # reading at its own pace.
        stream_timeout = udp_socket.gettimeout()
        sync_timeout = (stream_timeout if first_frame_timeout is None
                        else first_frame_timeout)

        # Create variable that indicates if DataFrame was constructed success-
        # fully
        success = False
        while success == False:
            
            # Initialize list for received packages            
            packages = []
            
            # Read incoming packages until one with the package index 1 is received
            sync = False

            # This is the read that may have to wait for the device
            udp_socket.settimeout(sync_timeout)

            while sync == False:

                # Receive package
                try:
                    package = udp_socket.recv(self.tparray._package_size)
                except socket.timeout:
                    udp_socket.settimeout(stream_timeout)
                    raise HTPA_UDPTimeout(f'Socket timeout ({sync_timeout} s) '
                                          f'while waiting for the first '
                                          f'package of a frame from device '
                                          f'{dev_id}.')

                
                # Check package index
                # Exception for (32,32)
                if self.tparray._npsize == (32,32):
                    
                    if len(package) == 1292:
                        package_index = 1
                    else:
                        package_index = 2
                else:
                    # All other arrays
                    package_index = int(package[0])
                
                # Check if it is equal to 1
                if package_index == 1:
                    packages.append(package)
                    sync = True

            # Synced: the device is sending, so the rest of this frame is
            # expected at the stream's own pace again
            udp_socket.settimeout(stream_timeout)

            for p in range(2,self.tparray._package_num+1):

                # Receive package
                try:
                    package = udp_socket.recv(self.tparray._package_size)
                except socket.timeout:
                    raise HTPA_UDPTimeout(f'Socket timeout after package '
                                          f'{p-1} of '
                                          f'{self.tparray._package_num} '
                                          f'from device {dev_id}.')

                # Get index of package
                # Exception for (32,32)
                if self.tparray._npsize == (32,32):
                    if len(package) == 1288:
                        package_index = 2
                    else:
                        package_index = 0
                else:
                    # All other arrays 
                    package_index = int(package[0])
                
                # Check if package index has the expected value
                if package_index == p:
                    packages.append(package)
                
            # In the end check if as many packages were received as expected
            if len(packages) == self.tparray._package_num:
                               
                # If yes, pass the packages to the class that parses the
                # bytes into an np.ndarray or pd.DataFrame
                # df_frame = self.bytestream_reader.bytes_to_df(packages)
                
                frame = bytestream_converter.convert(packages)
                
                success = True
                
            else:
                print('Frame lost.')
                sync = False


        if return_raw:
            return frame, packages

        return frame


    # def read_single_frame(self,dev_id,**kwargs):
        
    #     # Read voltage 'c' or temperature 'k' frame
    #     mode = kwargs.pop('mode','k')
        
    #     # Get device information from the device list
    #     dev_info = self.devices.loc[[dev_id]]
        
    #     # If more than one devices have the same device id, return error
    #     if len(dev_info) != 1:
    #         Exception('Multiple devices have the same device id.')
        
    #     # Get udp socket
    #     udp_socket = self.sockets[dev_id]
        
    #     # Create server address, i.e. tuple of device IP and port 
    #     server_address = (dev_info['IP'].item(), self.port)

    #     # Send message to device to send the single frame
    #     mess = udp_socket.sendto(bytes(mode,'utf-8'),server_address)        
        
    #     # Michaels code is atrocious. Sometimes it sends no frame, sometimes
    #     # 1 frame and sometimes two frames. All these cases have to be considered
    #     # Try to get first frame (works always)
    #     try:
    #         frame = self._receive_udp_frame(udp_socket)
    #     except:
    #         frame = pd.DataFrame(data=[],
    #                              columns = self.tparray.get_serial_data_order())
        
    #     if frame is None:
    #         frame = pd.DataFrame(data=[],
    #                              columns = self.tparray.get_serial_data_order())
        
    #     if len(frame) == 0:
    #         print('Reading frame failed.')
    #     else:
    #         print('Successfully received frame')
        
    #     # Clean up whole port in which a second frame may or may not be
    #     self._read_port(udp_socket)
        
    #     return frame
    
    # def _receive_udp_frame(self,udp_socket):
    #     """
    #     Receives UDP packages and tries to put them together to a frame

    #     Parameters
    #     ----------
    #     udp_socket : TYPE
    #         DESCRIPTION.

    #     Returns
    #     -------
    #     None.

    #     """
        
    #     # Initialize list for received packages            
    #     packages = []
        
    #     # Read incoming packages until one with the package index 1 is received
    #     sync = False
        
    #     while sync == False:
            
    #         # Receive package
    #         package = udp_socket.recv(self.tparray._package_size)
            
    #         # Get index of package
    #         package_index = int(package[0])
            
    #         # Check if it is equal to 1
    #         if package_index == 1:
    #             packages.append(package)
    #             sync = True
        
    #     for p in range(2,self.tparray._package_num+1):
            
    #         # Receive package
    #         package = udp_socket.recv(self.tparray._package_size)
            
    #         # Get index of package
    #         package_index = int(package[0])
            
    #         # Check if package index has the expected value
    #         if package_index == p:
    #             packages.append(package)
                
    #     # In the end check if as many packages were received as expected
    #     if len(packages) == self.tparray._package_num:
                            
    #         frame = self.bytestream_converter.convert(packages)
                        
    #     else:
    #         print('Frame lost.')
    #         frame = None


    #     return frame
        
    
    def device_put_to_sleep(self,dev_id,**kwargs):
        """
        Puts a device to sleep
        """
        # Get device information from the device list
        dev_info = self.devices.loc[[dev_id]]
        
        # If more than one devices have the same device id, return error
        if len(dev_info) != 1:
            Exception('Multiple devices have the same device id.')
        
        # Get udp socket
        udp_socket = self.sockets[dev_id]
                
        # Create server address
        server_address = (dev_info['IP'].item(), self.port)

        # Send message to device to start streaming bytes
        _ = udp_socket.sendto(bytes('s','utf-8'),server_address)
        answer = udp_socket.recv(self.tparray._package_size)
        answer = answer.decode('utf-8')
        
        
        if answer.strip() == 'Module set into sleep mode.':
            print('Device ' + str(dev_id) + ' put to sleep.')
            return True
        else:
            print('Putting device ' + str(dev_id) + ' to sleep failed.')
            return False
        
    def device_wake_up(self,dev_id,**kwargs):
        """
        Wakes device up
        """
        # Get device information from the device list
        dev_info = self.devices.loc[[dev_id]]
        
        # If more than one devices have the same device id, return error
        if len(dev_info) != 1:
            Exception('Multiple devices have the same device id.')
        
        # Get udp socket
        udp_socket = self.sockets[dev_id]
                
        # Create server address
        server_address = (dev_info['IP'].item(), self.port)

        # Send message to device to start streaming bytes
        _ = udp_socket.sendto(bytes('S','utf-8'),server_address)
        answer = udp_socket.recv(self.tparray._package_size)
        answer = answer.decode('utf-8')
        
        
        if answer.strip() == 'Module woke up.':
            print('Device ' + str(dev_id) + ' awake.')
            return True
        else:
            print('Failed waking up device ' + str(dev_id) + '.')
            return False
    
    
    # def _device_is_streaming(self,udp_socket,server_address):
    #     """
    #     Check if the device is already streaming

    #     Parameters
    #     ----------
    #     udp_socket : TYPE
    #         DESCRIPTION.
    #     server_address : TYPE
    #         DESCRIPTION.

    #     Returns
    #     -------
    #     None.

    #     """
        
    #     # Try to read in packages and check if they have the size expected from
    #     # a HTPA device
        
    #     # Try to read in a maximum amount of packages that corresponds to
    #     # 10 frames
    #     p_max = 10 * self.tparray._package_num
        
    #     for p in range(p_max):
            
    #         try:
    #             package = udp_socket.recv(self.tparray._package_size)
        
    #     for i in range()
        
    
    def _callstring_to_information(self,call:bytes):
        """
        Extracts information from the ridiculously long and inefficient answer
        to the device call.

        Parameters
        ----------
        call : bytes
            DESCRIPTION.

        Returns
        -------
        dev_info : TYPE
            DESCRIPTION.

        """

        call = call.decode('utf-8')
        
        # Extract information from callstring
        arraytype = int(call.split('Arraytype')[1].split('MODTYPE')[0])
        mac_id = re.findall(r'\w{2}\.\w{2}\.\w{2}\.\w{2}.\w{2}.\w{2}',call)[0]
        ip = re.findall(r'\d{3}\.\d{3}\.\d{3}.\d{3}',call)[0]
        
        # Remove leading zeros from IP
        ip = '.'.join([str(int(x)) for x in ip.split('.')])
        
        
        dev_id = int(call.split('DevID:')[1].split('Emission')[0])
        
        
        dev_info = pd.DataFrame(data = [],
                                columns = self.col_dict.keys(),
                                index = self.index)
        
        dev_info.loc[dev_id,['IP','MAC-ID','Arraytype']] = \
            [ip,mac_id,arraytype]
        
        dev_info = dev_info.astype(self.col_dict)    
        
        return dev_info
    


# ─────────────────────────────────────────────────────────────────────────────
class UDP_Driver:
    """Marker base class, mirroring I2C_Driver / SPI_Driver / USB_Driver."""
    pass


class UDP_HTPAError(Exception):
    """Raised on UDP communication or data errors."""


# ─────────────────────────────────────────────────────────────────────────────
class UDP_HTPA(UDP_Driver):
    """
    Single-device UDP driver for HTPA series devices.

    One object = one device, so this is the UDP counterpart of ``USB_HTPA``:
    it owns one socket, binds one device, and ``acquire_frame()`` returns the
    frame dictionary of ``hspy.drivers._frame``. ``UDP_Client`` /
    ``UDP_ClientProcess`` only have to drive the lifecycle below -- all
    discovery and binding happens in here.

    Typical usage::

        with UDP_HTPA(IP='192.168.240.80') as sensor:
            sensor.start_stream()
            frame = sensor.acquire_frame()
            sensor.stop_stream()

    ``ArrayType`` does not have to be given: the answer to the bind call
    carries it, so ``init()`` reads it off the device. Pass it only to pin it
    down (it is then checked against what the device reports).

    Thread safety: one instance must only be used from one thread at a time,
    exactly like the other drivers.
    """

    def __init__(self,
                 IP: str = '',
                 DevID: int = -1,
                 Bcast_Addr: str = '',
                 ArrayType: int = None,
                 port: int = 30444,
                 stream_cmd: bytes = b'K',
                 timeout_s: float = 1.0,
                 first_frame_timeout: float = 5.0,
                 keep_raw: bool = False,
                 auto_init: bool = True):
        """
        Parameters
        ----------
        IP : str
            IP of the device to bind. Alternative to DevID + Bcast_Addr.
        DevID : int
            Device number of the device to bind. Requires Bcast_Addr, since
            the device has to be discovered by broadcast first.
        Bcast_Addr : str
            Broadcast address of the subnet the device lives in.
        ArrayType : int, optional
            Array type of the device. Read off the device in init() when not
            given; when given, init() raises if the device disagrees.
        port : int
            UDP port the HTPA devices listen on.
        stream_cmd : bytes
            Command that starts the continuous bytestream, and therefore what
            the pixel values mean. b'K' (default) streams them in dK, which
            end up in frame['pix_dK'] with frame_unit 'dK'; anything else
            ends up in frame['pix'] with frame_unit 'digits'.
        timeout_s : float
            Socket timeout for the reads of a stream that is already running.
        first_frame_timeout : float
            Socket timeout for the read that waits for the device to start
            sending. The device needs a moment to act on the stream command,
            which is not the same wait as the one between two packages of a
            running stream. Used until the first frame has arrived.
        keep_raw : bool
            If True, every frame additionally carries the UDP datagrams it
            was assembled from under frame['raw']. Off by default: the
            packages roughly double the size of a pickled frame.
        auto_init : bool
            If True (default), open() and init() are called right here, so
            the driver is ready to stream as soon as it is constructed --
            same convention as the I2C, SPI and USB drivers.

        Returns
        -------
        None.

        """

        super(UDP_HTPA, self).__init__()

        # Checked here so that a missing argument fails at construction
        # rather than halfway through open()
        if not ((DevID != -1 and len(Bcast_Addr) != 0) or len(IP) != 0):
            raise ValueError('Either IP or DevID and Bcast_Addr of the '
                             'device to be bound have to be specified!')

        self._IP = IP
        self._DevID = DevID
        self._Bcast_Addr = Bcast_Addr
        self._ArrayType_requested = ArrayType
        self._port = port

        # Acquisition options. Plain attributes, so that a caller can change
        # them between frames; acquire_frame() itself takes no arguments, the
        # same as every other driver in hspy.drivers.
        self.stream_cmd = stream_cmd
        self.timeout_s = timeout_s
        self.first_frame_timeout = first_frame_timeout
        self.keep_raw = keep_raw

        # Populated by open() / init()
        self._reader = None
        self._bound = False
        self._streaming = False
        self._MAC = ''
        self.ArrayType = None
        self.tparray = None

        # None until the first frame of the current stream has arrived, which
        # is what switches the socket from first_frame_timeout to timeout_s
        self._t_first_frame = None

        self.open()

        if auto_init:
            try:
                self.init()
            except Exception:
                self.close()
                raise

    def __enter__(self) -> "UDP_HTPA":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __repr__(self) -> str:
        state = ('streaming' if self._streaming
                 else ('bound' if self._bound else 'closed'))
        return (f"UDP_HTPA(DevID={self._DevID}, IP={self._IP!r}, "
                f"ArrayType={self.ArrayType}, {state})")

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def is_open(self) -> bool:
        """True while this driver holds a socket for its device."""
        return self._reader is not None and self._bound

    @property
    def is_connected(self) -> bool:
        """
        True while the device is bound.

        UDP is connectionless, so unlike the USB driver this cannot tell that
        a device has gone away -- only that it was never bound or has been
        released. Noticing a device that stopped answering is the caller's
        job (UDP_Client / UDP_ClientProcess do it with a watchdog over the
        HTPA_UDPTimeout exceptions).
        """
        return self.is_open

    @property
    def is_streaming(self) -> bool:
        return self._streaming

    @property
    def calcdK(self) -> bool:
        """
        True if frames carry 'pix_dK'. Read-only: the device applies its own
        look-up table when streaming with b'K', the host never does.
        """
        return self.stream_cmd == b'K'

    @property
    def DevID(self) -> int:
        return self._DevID

    @property
    def IP(self) -> str:
        return self._IP

    @property
    def serial(self) -> str:
        """The device MAC-ID, which is what identifies it on the network."""
        return self._MAC

    @property
    def port(self) -> int:
        return self._port

    @property
    def width(self):
        return None if self.tparray is None else self.tparray.width

    @property
    def height(self):
        return None if self.tparray is None else self.tparray.height

    @property
    def n_pixels(self):
        return (None if self.tparray is None
                else self.tparray.width * self.tparray.height)

    @property
    def devices(self) -> pd.DataFrame:
        """The device table of the reader underneath (one row)."""
        return (pd.DataFrame() if self._reader is None
                else self._reader.devices)

    # ── Bind / release ───────────────────────────────────────────────────────

    def open(self) -> None:
        """
        Discover (if needed) and bind the device, leaving it with one socket.

        A second call on an already-bound driver does nothing, so that
        ``open()`` can be used to re-open a closed driver without having to
        check first.
        """

        if self.is_open:
            return

        # One HTPA_UDPReader per device. The reader is a multi-device manager
        # and this driver is deliberately not: it is used here purely for the
        # packet-level work (binding, call-string parsing, package assembly),
        # with exactly one device in its table.
        self._reader = HTPA_UDPReader(ArrayType=self._ArrayType_requested,
                                      port=self._port)

        try:
            if (self._DevID != -1) and (len(self._Bcast_Addr) != 0):
                # broadcast() always waits out its full 5 s response timeout,
                # so it is skipped when the device is somehow already known.
                if self._DevID not in self._reader.devices.index:
                    self._reader.broadcast(self._Bcast_Addr)
                self._reader.bind_tparray(DevID=self._DevID)
            else:
                self._reader.bind_tparray(IP=self._IP)

            self._bound = True

            # Read back what was actually bound: binding by IP yields the
            # DevID, binding by DevID yields the IP, and either way the
            # answer to the call carries the MAC-ID and the array type.
            devices = self._reader.devices
            if len(devices) != 1:
                raise UDP_HTPAError(f"expected exactly one bound device, "
                                    f"got {len(devices)}.")

            self._DevID = int(devices.index.item())
            self._IP = str(devices['IP'].item())
            self._MAC = str(devices['MAC-ID'].item())

            # The stream is not running yet, so the socket may as well use
            # the longer of the two timeouts until the first frame arrives.
            self._set_socket_timeout(self.first_frame_timeout)
        except Exception:
            # Nothing is half-bound: give the socket back and let the caller
            # see the original error.
            self._bound = False
            self._reader = None
            raise

    def init(self) -> None:
        """
        Determine the array type and build the TPArray describing the frames.

        The answer to the bind call already carries the array type, so this
        does not talk to the device again. Called by __init__ unless
        auto_init=False.
        """

        if not self.is_open:
            raise UDP_HTPAError('Device is not bound: call open() first.')

        reported = int(self.devices['Arraytype'].item())

        if (self._ArrayType_requested is not None
                and int(self._ArrayType_requested) != reported):
            raise UDP_HTPAError(
                f"Device {self._DevID} reports ArrayType {reported}, but "
                f"ArrayType {self._ArrayType_requested} was requested.")

        self.ArrayType = reported

        # Setting ArrayType on the reader builds its TPArray, which
        # read_continuous_bytestream() needs for the package sizes
        self._reader.ArrayType = reported
        self.tparray = self._reader.tparray

    def close(self) -> None:
        """
        Stop the stream, release the device and close the socket.

        Safe to call more than once and safe to call on a driver whose device
        has gone away -- every step is attempted independently, so one
        failure does not leave the rest undone.
        """

        if self._reader is None:
            return

        reader, self._reader = self._reader, None
        bound, self._bound = self._bound, False
        streaming, self._streaming = self._streaming, False

        if not bound:
            return

        if streaming:
            try:
                reader.stop_continuous_bytestream(DevID=self._DevID)
            except Exception as e:
                print(f"[UDP_HTPA {self._DevID}] Stopping the stream "
                      f"failed: {e}")

        try:
            # Sends the release command and closes the device socket
            reader.release_tparray(self._DevID)
        except Exception as e:
            print(f"[UDP_HTPA {self._DevID}] Releasing the device "
                  f"failed: {e}")

    # ── Stream control ───────────────────────────────────────────────────────

    def start_stream(self, cmd: bytes = None) -> None:
        """
        Start the continuous bytestream of the device.

        Parameters
        ----------
        cmd : bytes, optional
            Overrides (and replaces) self.stream_cmd for this and every
            following stream. Mostly there so that start_stream(b'K') reads
            the same as the USB driver call.
        """

        if not self.is_open:
            raise UDP_HTPAError('Device is not bound: call open() first.')
        if self.tparray is None:
            raise UDP_HTPAError('ArrayType unknown: call init() first.')

        if cmd is not None:
            self.stream_cmd = cmd

        self._reader.start_continuous_bytestream(self._DevID,
                                                 cmd=self.stream_cmd)
        self._streaming = True

        # The first frame of this stream gets the longer timeout again
        self._t_first_frame = None
        self._set_socket_timeout(self.first_frame_timeout)

    def stop_stream(self):
        """Stop the continuous bytestream of the device."""

        if not self.is_open:
            return None

        answer = self._reader.stop_continuous_bytestream(DevID=self._DevID)
        self._streaming = False

        return answer

    # ── Frame acquisition ────────────────────────────────────────────────────

    def acquire_frame(self) -> dict:
        """
        Assemble one frame from the running stream.

        Takes no arguments, like every other driver acquire_frame(): what it
        does is set by the attributes (stream_cmd, timeout_s,
        first_frame_timeout, keep_raw). Intended to be called once per loop
        iteration by UDP_Client / UDP_ClientProcess.

        Returns
        -------
        dict
            The frame dictionary of hspy.drivers._frame. Which pixel stage it
            carries depends on stream_cmd: b'K' gives 'pix_dK' (frame_unit
            'dK'), anything else gives 'pix' (frame_unit 'digits'). Does NOT
            contain 'success' or 'image_id' -- the calling thread / process
            adds those.

        Raises
        ------
        HTPA_UDPTimeout
            The device sent nothing within the socket timeout.
        """

        if not self._streaming:
            raise UDP_HTPAError('Stream not running: call start_stream() '
                                'first.')

        # Until the first frame has arrived the socket keeps the longer
        # timeout set in start_stream(): the device needs a moment to act on
        # the stream command, which is not an error.
        result = self._reader.read_continuous_bytestream(
            self._DevID,
            return_raw=self.keep_raw)

        t = datetime.now()
        t_mono = time.monotonic()

        if self._t_first_frame is None:
            # Streaming for real now, so the reads go back to the pace of a
            # running stream
            self._t_first_frame = t_mono
            self._set_socket_timeout(self.timeout_s)

        groups, packages = result if self.keep_raw else (result, None)

        # TPArray.DataCols spells the groups the way the .bds / .txt column
        # headers do; the frame dict uses the spelling of the drivers
        aux = {AUX_FROM_DATACOLS[group]: values
               for group, values in groups.items()
               if group in AUX_FROM_DATACOLS}

        # b'K' streams object temperatures in dK, every other command streams
        # values the frame dict calls digits
        pixels = groups['pix']
        stage = ({'pix_dK': pixels} if self.stream_cmd == b'K'
                 else {'pix': pixels})

        return new_frame(interface='udp',
                         ArrayType=self.ArrayType,
                         shape=(self.tparray.height, self.tparray.width),
                         DevID=self._DevID,
                         serial=self._MAC,
                         t=t,
                         t_mono=t_mono,
                         eloff=aux.get('eloff'),
                         ptat=aux.get('ptat'),
                         vdd=aux.get('vdd'),
                         atc=aux.get('atc'),
                         Tamb=aux.get('Tamb'),
                         raw=packages,
                         **stage)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _set_socket_timeout(self, timeout: float) -> None:
        """Set the timeout of the socket of this device, if it has one yet."""

        if self._reader is None:
            return

        udp_socket = self._reader.sockets.get(self._DevID)
        if udp_socket is not None:
            udp_socket.settimeout(timeout)
