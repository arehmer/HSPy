# -*- coding: utf-8 -*-
"""
Multiprocessing counterpart to threads.py / ProcessorThread.py.

Contains multiprocessing versions of:
  - UDP_Client             (threads.py)          -> UDP_ClientProcess
  - I2C_Client              (threads.py)          -> I2C_ClientProcess
  - UDP_PickleServer        (threads.py)          -> UDP_PickleServerProcess
  - UDP_PickleClient        (threads.py)          -> UDP_PickleClientProcess
  - ProposalDetectorThread  (ProcessorThread.py)  -> ProposalDetectorProcess

DESIGN NOTE -- why these classes take *_cls / *_kwargs instead of a
ready-made instance
--------------------------------------------------------------------------
The threading versions (UDP_Client, I2C_Client) are constructed by passing
in an already-created, already-opened driver/reader object (an
HTPA_UDPReader with a bound socket, an I2C_HTPA32x32d with an open SMBus
handle). That works for threading because a thread shares the parent
process's memory -- the object, socket and all, is simply usable from the
new thread.

It does NOT work for multiprocessing: multiprocessing.Process pickles
whatever the child needs across the process boundary (or, on Linux with
the default 'fork' start method, silently duplicates open file
descriptors into the child -- which is arguably worse, since both parent
and child then think they own the same socket/bus handle). Either way,
handing a live socket or an open I2C bus handle to a child process is not
safe.

So every class below is constructed with a *class* + a *kwargs dict*
(everything needed to build the underlying driver/reader/detector) rather
than a live instance, and does the actual construction inside `_setup()`,
which -- per processes_base.py -- runs once, inside the child process,
right before the main loop starts. Only picklable configuration (classes,
dicts, strings, numbers) ever crosses the process boundary.

This applies to plain sockets too (UDP_PickleServerProcess /
UDP_PickleClientProcess below): even though a bare, not-yet-bound
socket.socket() often happens to survive being duplicated into a forked
child on Linux, that's start-method-dependent (it does NOT survive the
'spawn'/'forkserver' start methods, which are the default on Windows/macOS
and are frequently required e.g. together with CUDA). So sockets are also
created inside _setup(), never in __init__.
"""

import time
import socket
import struct
import pickle as pkl
from multiprocessing import Queue as MPQueue

import cv2
import numpy as np

from hspy.LuT import LuT
from hspytools.tparray import TPArray
import hsod
from hsod.cv.detectors import build_detector
from hsod.cv.tracktors import build_tracktor

from hspy.ipc.processes_base import WProcess_R1, RProcess_R1, RWProcess_R1


# Same wire-format constants as UDP_PickleServer / UDP_PickleClient in
# threads.py -- kept in sync manually since threads.py is a threading-only
# module and this one has no import dependency on it.
_PICKLE_HEADER_FMT = '!IHH'
_PICKLE_HEADER_SIZE = struct.calcsize(_PICKLE_HEADER_FMT)

# Stay safely below the 65507 byte payload limit of a UDP datagram
_PICKLE_MAX_CHUNK = 60000


class UDP_ClientProcess(WProcess_R1):
    """
    Multiprocessing version of UDP_Client. Runs an HTPA_UDPReader in its
    own process and writes assembled frames into write_buffer.
    """

    def __init__(self,
                 udp_reader_cls,
                 udp_reader_kwargs: dict,
                 write_buffer,
                 IP: str = '',
                 DevID: int = -1,
                 Bcast_Addr: str = '',
                 **kwargs):
        """
        Parameters
        ----------
        udp_reader_cls : type
            The HTPA_UDPReader class (not an instance!) to construct
            inside the child process.
        udp_reader_kwargs : dict
            Keyword arguments to construct udp_reader_cls with, e.g.
            {'listen_port': 30444}.
        write_buffer : Queue / SharedMemoryQueue
            Buffer that assembled frames are put into.
        IP, DevID, Bcast_Addr : see UDP_Client (threads.py) -- same
            binding semantics, just deferred to _setup().

        Returns
        -------
        None.

        """

        self.udp_reader_cls = udp_reader_cls
        self.udp_reader_kwargs = udp_reader_kwargs

        self.IP = IP
        self.DevID = DevID
        self.Bcast_Addr = Bcast_Addr

        super().__init__(name='udp_process',
                         write_buffer=write_buffer,
                         **kwargs)

    def _setup(self):

        # Construct the reader (and therefore open its socket) here,
        # inside the child process.
        self.udp_reader = self.udp_reader_cls(**self.udp_reader_kwargs)

        # Depending on whether the device's IP or the Device ID along
        # with the broadcast address are provided, the process of
        # finding and binding the corresponding htpa device differs
        if (self.DevID != -1) and (len(self.Bcast_Addr) != 0):
            self.udp_reader.broadcast(self.Bcast_Addr)
            self.udp_reader.bind_tparray(DevID=self.DevID)

        elif len(self.IP) != 0:
            self.udp_reader.bind_tparray(IP=self.IP)

            # Save DevID of bound device to attribute
            devices = self.udp_reader.devices
            self.DevID = devices[devices['IP'] == self.IP].index.item()

        else:
            print('Either IP or DevID and Bcast_Addr of the device to be ' +
                  'bound have to be specified!')

        # Start continuous bytestream
        self.udp_reader.start_continuous_bytestream(self.DevID)

        # Set image_id counter
        self.image_id = 0

        # Set time
        self.t0 = time.time()

    def _target(self):

        # Dictionary for storing results in
        result = {}

        try:
            # Try to assemble a frame from the udp packages
            frame = self.udp_reader.read_continuous_bytestream(self.DevID)

            # Set success flag and store frame
            result['success'] = True
            result['frame'] = frame
        except Exception:
            # Set success flag to False in case of failure
            result['success'] = False

        # Store image_id
        result['image_id'] = self.image_id

        # Store device ID
        result['DevID'] = self.DevID

        # Increment image_id
        self.image_id = self.image_id + 1

        return result

    def stop(self):
        self._exit.set()


class I2C_ClientProcess(WProcess_R1):
    """
    Multiprocessing version of I2C_Client. Runs an I2C_HTPA32x32d in its
    own process and writes acquired (+ converted / calibrated / LuT-mapped)
    frames into write_buffer.
    """

    def __init__(self,
                 i2c_driver_cls,
                 i2c_driver_kwargs: dict,
                 write_buffer,
                 lut_csv_path=None,
                 lut_offset: int = 0,
                 active_vdd: bool = True,
                 blind_vdd: bool = False,
                 applyCalib: bool = True,
                 calcdK: bool = True,
                 **kwargs):
        """
        Parameters
        ----------
        i2c_driver_cls : type
            The I2C_HTPA32x32d class (not an instance!) to construct
            inside the child process -- opening the SMBus handle in the
            parent and pickling/duplicating it into the child is not
            safe, so construction is deferred to _setup().
        i2c_driver_kwargs : dict
            Keyword arguments to construct i2c_driver_cls with, e.g.
            {'bus': 1}.
        write_buffer : Queue / SharedMemoryQueue
            Buffer that acquired frames are put into.
        lut_csv_path : str or Path, optional
            If given, a LuT is loaded from this CSV and assigned to the
            driver inside _setup() (mirrors setting i2c_driver.LuT in the
            threading version's calling script).
        lut_offset : int
            Passed through to LuT.from_csv().
        active_vdd, blind_vdd, applyCalib, calcdK :
            Forwarded to i2c_driver.acquire_frame() every iteration.

        Returns
        -------
        None.

        """

        self.i2c_driver_cls = i2c_driver_cls
        self.i2c_driver_kwargs = i2c_driver_kwargs

        self.lut_csv_path = lut_csv_path
        self.lut_offset = lut_offset

        self.active_vdd = active_vdd
        self.blind_vdd = blind_vdd
        self.applyCalib = applyCalib
        self.calcdK = calcdK

        super().__init__(name='i2c_process',
                         write_buffer=write_buffer,
                         **kwargs)

    def _setup(self):

        # Opens the I2C bus (SMBus handle) here, inside the child process.
        # I2C_HTPA32x32d.__init__ already calls open() + init() itself.
        self.i2c_driver = self.i2c_driver_cls(**self.i2c_driver_kwargs)

        if self.lut_csv_path is not None:
            lut = LuT()
            lut.from_csv(self.lut_csv_path, offset=self.lut_offset)
            self.i2c_driver.LuT = lut

        # Set image_id counter
        self.image_id = 0

        # Set time
        self.t0 = time.time()

    def _target(self):

        # Dictionary for storing results in
        result = {}

        try:
            # Acquire and process one frame from the sensor
            data = self.i2c_driver.acquire_frame(active_vdd=self.active_vdd,
                                                  blind_vdd=self.blind_vdd,
                                                  applyCalib=self.applyCalib,
                                                  calcdK=self.calcdK)

            # Set success flag and store frame data
            result['success'] = True
            result.update(data)
        except Exception as e:
            # Set success flag to False in case of failure
            result['success'] = False
            print(f"[{self.name}] {e}")

        # Store image_id
        result['image_id'] = self.image_id

        # Increment image_id
        self.image_id = self.image_id + 1

        return result

    def stop(self):
        self._exit.set()


class CVProcess(RWProcess_R1):
    """
    Multiprocessing version of ProposalDetectorThread (ProcessorThread.py).
    Reads frames from read_buffer, runs object detection, writes the
    result (original data + detections) into write_buffer.
    """

    def __init__(self,
                 name: str,
                 model_dict: dict,
                 write_buffer,
                 read_buffer,
                 **kwargs):
        """
        Parameters
        ----------
        model_dict : dict
            Kwargs passed to build_detector(**model_dict) or 
            build_tracktor(**model_dict )inside
            _setup(). The detector/tracktor is built inside the child process
            rather than being constructed in the parent and pickled
            across, since detector/model objects frequently hold
            resources (loaded weights, a GPU/inference context, ...)
            that either aren't picklable or shouldn't be duplicated
            across a process boundary.
        write_buffer, read_buffer : Queue / SharedMemoryQueue

        Returns
        -------
        None.

        """

        self.model_dict = model_dict

        super().__init__(name=name,
                         read_buffer=read_buffer,
                         write_buffer=write_buffer,
                         **kwargs)

    def _setup(self):
        
        # Check if the provided model is a detector or tracker by name
        # and build the model (load weights / params, allocate whatever
        # runtime context it needs) here, inside the child process
        name = self.model_dict['name']  
        
        if name in hsod.cv.detectors.get_classes():
            print(f'Provided model {name} identified as detector.')
            self.model = build_detector(**self.model_dict)
        elif name in hsod.cv.tracktors.get_classes():
            print(f'Provided model {name} identified as tracktor.')
            self.model = build_tracktor(**self.model_dict)
        else:
            print(f'Provided model {name} could not be identified.')
            return None

        self.tparray = TPArray(SensorType=self.model.SensorType)

        self.num_pix = len(self.tparray._pix)

        self.t0 = time.time()

    def _target(self):

        # Get result from upstream process
        data = self.read_buffer.get()

        # Check success flag of upstream process
        if data['success'] is True:

            if 'pix_dK' in data.keys():
                frame = data['pix_dK']
            elif 'frame' in data.keys():
                frame = data['frame']

            # Reshape if not the proper size
            if frame.ndim == 1:
                frame = frame[0:self.num_pix]
                frame = frame.reshape(self.tparray._npsize)

            try:
                print(frame)
                data_processed = self.model.process({'frame': frame,
                                                     'idx': data['image_id']})
                data.update(data_processed)

            except Exception as e:
                print(f"[{self.name}] {e}")

                # Set success flag to False in case of failure
                data['success'] = False

        else:
            # If upstream process failed, set success flag to False
            data['success'] = False

        return data

    def stop(self):
        self._exit.set()


class UDP_PickleServerProcess(RProcess_R1):
    """
    Multiprocessing version of UDP_PickleServer. Reads arbitrary
    (picklable) data from an upstream buffer, in its own process, and
    sends it out via UDP to a single peer running a UDP_PickleClient /
    UDP_PickleClientProcess.

    Since a pickled payload can easily exceed the size of a single UDP
    datagram, it is split into numbered chunks that are reassembled by
    the receiver. As with any data sent over UDP, delivery is not
    guaranteed: a message of which one or more chunks are lost in transit
    is silently dropped by the receiver.
    """

    def __init__(self,
                 name: str,
                 read_buffer,
                 target_ip: str,
                 target_port: int,
                 **kwargs):
        """
        Parameters
        ----------
        read_buffer : Queue / SharedMemoryQueue
            Buffer that data to be transmitted is read from.
        target_ip, target_port : str, int
            Address of the peer running UDP_PickleClient(Process).

        Returns
        -------
        None.

        """

        self.target_ip = target_ip
        self.target_port = target_port

        super().__init__(name=name,
                         read_buffer=read_buffer,
                         **kwargs)

    def _setup(self):

        # The socket is created here, inside the child process -- not in
        # __init__ -- so that this works regardless of multiprocessing
        # start method ('fork', 'spawn' or 'forkserver').
        self.target_addr = (self.target_ip, self.target_port)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._msg_id = 0

    def _target(self):

        # Get data (e.g. a dict with a numpy array / pandas DataFrame)
        # from the upstream buffer
        data = self.read_buffer.get()

        # Serialize it
        payload = pkl.dumps(data)

        # Split the payload into chunks that fit into a single UDP
        # datagram
        chunks = [payload[i:i + _PICKLE_MAX_CHUNK]
                 for i in range(0, len(payload), _PICKLE_MAX_CHUNK)] or [b'']

        chunk_count = len(chunks)

        # Send every chunk in its own datagram, prefixed with a header
        # that allows the receiver to reassemble the message
        for chunk_index, chunk in enumerate(chunks):
            header = struct.pack(_PICKLE_HEADER_FMT,
                                 self._msg_id,
                                 chunk_index,
                                 chunk_count)
            self._socket.sendto(header + chunk, self.target_addr)

        # Increment (and wrap) the message id
        self._msg_id = (self._msg_id + 1) % 2**32

        return None

    def _teardown(self):
        # Runs inside the child process once the loop has exited. Unlike
        # the threading version, stop() (which runs in the PARENT
        # process) cannot reach into this socket -- it only exists in the
        # child's memory -- so cleanup happens here instead.
        self._socket.close()

    def stop(self):
        self._exit.set()


class UDP_PickleClientProcess(WProcess_R1):
    """
    Multiprocessing version of UDP_PickleClient. Runs in its own process,
    receives data sent by a UDP_PickleServer / UDP_PickleServerProcess,
    reassembles and unpickles it, and writes the resulting object into
    write_buffer for downstream processes to consume.
    """

    def __init__(self,
                 name: str,
                 write_buffer,
                 listen_ip: str = '0.0.0.0',
                 listen_port: int = 0,
                 timeout: float = 1.0,
                 **kwargs):
        """
        Parameters
        ----------
        write_buffer : Queue / SharedMemoryQueue
            Buffer that reassembled messages are put into.
        listen_ip, listen_port : str, int
            Local address to bind the receiving socket to.
        timeout : float
            recvfrom() timeout in seconds. Also bounds how quickly this
            process notices stop() was called (see note in _target()
            below) -- keep this reasonably small if fast shutdown matters.

        Returns
        -------
        None.

        """

        self.listen_ip = listen_ip
        self.listen_port = listen_port
        self.timeout = timeout

        # Small one-shot handshake queue: the child reports the address
        # it actually bound to, once, right after binding in _setup().
        # A plain multiprocessing.Queue (unlike a socket) is fine to
        # create here in the parent -- it's designed to cross the
        # process boundary safely.
        self._addr_queue = MPQueue(maxsize=1)
        self._resolved_addr = None

        super().__init__(name=name,
                         write_buffer=write_buffer,
                         **kwargs)

    def _setup(self):

        # Socket is created and bound here, inside the child process.
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind((self.listen_ip, self.listen_port))
        self._socket.settimeout(self.timeout)

        # Report the address actually bound to back to the parent (this
        # matters if listen_port=0 was passed and the OS assigned an
        # ephemeral port).
        self._addr_queue.put(self._socket.getsockname())

        # Chunks of messages that have not been fully received yet,
        # keyed by message id
        self._pending = {}

    def get_listen_addr(self, timeout: float = 5.0):
        """
        Blocks (from the PARENT process) until the child has bound its
        socket and reports the address back, or `timeout` seconds pass.

        Unlike UDP_PickleClient.listen_addr (a plain property that works
        immediately because the thread shares the parent's memory and the
        socket is created in __init__), the socket here only exists
        inside the child process and is only created once the process
        has actually started running (inside _setup()). So this must be
        called *after* .start(), and it can legitimately take a moment
        before the answer is available.

        Returns
        -------
        tuple (ip, port), or raises queue.Empty if the process hasn't
        bound within `timeout` seconds.
        """
        if self._resolved_addr is None:
            self._resolved_addr = self._addr_queue.get(timeout=timeout)
        return self._resolved_addr

    def _target(self):

        # Keep listening until either a full message has been
        # reassembled or the process is stopped.
        #
        # NOTE on shutdown: self._exit is a multiprocessing.Event, so it
        # IS visible here correctly even though stop() is called from the
        # parent process -- no special handling needed for that part.
        # What differs from the threading version is that stop() there
        # also closes the socket to unblock recvfrom() immediately; here,
        # stop() runs in the parent and has no way to reach into this
        # child's socket. Shutdown therefore relies entirely on the
        # recvfrom() timeout (set above from `timeout`) to periodically
        # re-check self._exit -- so worst-case shutdown latency for this
        # process is ~`timeout` seconds.
        while not self._exit.is_set():

            try:
                packet, _addr = self._socket.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                # Socket was closed (e.g. during shutdown) while blocked in recvfrom
                break

            header = packet[:_PICKLE_HEADER_SIZE]
            chunk = packet[_PICKLE_HEADER_SIZE:]

            msg_id, chunk_index, chunk_count = struct.unpack(_PICKLE_HEADER_FMT,
                                                              header)

            chunks = self._pending.setdefault(msg_id, {})
            chunks[chunk_index] = chunk

            # Once every chunk of this message has arrived, reassemble
            # it in order and hand it back to be put in the write buffer
            if len(chunks) == chunk_count:
                payload = b''.join(chunks[i] for i in range(chunk_count))
                del self._pending[msg_id]
                return pkl.loads(payload)

        return None

    def _teardown(self):
        self._socket.close()

    def stop(self):
        self._exit.set()


class ImshowProcess(RProcess_R1):
    """
    Multiprocessing version of Imshow. Reads frames (+ detections) from
    read_buffer and displays them in a cv2 window, running entirely in
    its own process.

    Note this is a case where multiprocessing is arguably a *better* fit
    than threading, not just a parallel option: GUI toolkits (including
    OpenCV's HighGUI backend) can be finicky about which thread they're
    driven from. Giving the display loop its own process sidesteps that
    class of issue entirely, at the cost of the usual IPC overhead.
    """

    def __init__(self,
                 name: str,
                 read_buffer,
                 ArrayType: int = None,
                 SensorType: int = None,
                 window_name: str = 'Sensor stream',
                 **kwargs):

        if ArrayType is None and SensorType is not None:
            self._tparray_kwargs = {'SensorType': SensorType}
        elif ArrayType is not None and SensorType is None:
            self._tparray_kwargs = {'ArrayType': ArrayType}
        else:
            raise Exception('Either ArrayType or SensorType must be provided.')

        self.window_name = window_name

        super().__init__(name=name,
                         read_buffer=read_buffer,
                         **kwargs)

    def _setup(self):

        # TPArray construction and window creation both happen here,
        # inside the child process.
        self.tparray = TPArray(**self._tparray_kwargs)
        self.num_pix = len(self.tparray._pix)

        self.t0 = time.time()

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def _target(self):

        # Get result from upstream process
        data = self.read_buffer.get()

        if data['success'] is not True:
            return

        frame = data['pix_dK']

        # Reshape if not the proper size
        if frame.ndim == 1:
            frame = frame[0:self.num_pix]
            frame = frame.reshape(self.tparray._npsize)

        # Convert to opencv type
        frame = cv2.normalize(frame, frame, 0, 255, cv2.NORM_MINMAX)
        frame = frame.astype(np.uint8)

        # Scale it up by a factor of 10
        sf = 10
        frame = np.repeat(np.repeat(frame, sf, axis=0), sf, axis=1)

        # Convert frame to RGB to be able to plot colored boxes
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        # Draw bounding boxes / annotations, if present
        if 'annot_frame' in data.keys():

            annotations = data['annot_frame'].annotations
            annotations['confirmed'] = False
            annotations.loc[annotations['label'] == 1, 'confirmed'] = True
            bboxes = annotations

            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.4
            fontColor = (255, 255, 0)
            thickness = 1
            lineType = 2

            for b in bboxes.index:

                box = bboxes.loc[[b]]

                if box['confirmed'].item() != True:
                    continue

                x, y = int(box['xtl'].item()), int(box['ytl'].item())
                w = int(box['xbr'].item()) - int(box['xtl'].item())
                h = int(box['ybr'].item()) - int(box['ytl'].item())

                # Scale coordinates
                x = int(sf * x); y = int(sf * y); w = int(sf * w); h = int(sf * h)

                frame = cv2.rectangle(frame,
                                      (x, y),
                                      (x + w, y + h),
                                      (255, 255, 0), 1)

                # Write score if it exists
                if box['score'].item() != -99:
                    score = f"{box['score'].item():.2}"
                    cv2.putText(frame, score, (x + 1, y + 10), font,
                               fontScale, fontColor, thickness, lineType)

            # Write the number of persons in the upper left corner
            num_persons = sum(bboxes['confirmed'] == True)

            cv2.putText(frame,
                        f'Number of persons: {num_persons}',
                        (0, 20), font, fontScale, fontColor, thickness, lineType)

        cv2.imshow(self.window_name, frame)
        cv2.waitKey(1)

    def _teardown(self):
        # Window is destroyed here, inside the child process, once the
        # loop has exited -- this always runs (unlike the threading
        # version's `if self._exit == True:` check before destroying the
        # window, which relied on self._exit being reassigned to a plain
        # bool in stop() -- that reassignment breaks Event semantics and
        # isn't carried over here; self._exit is a proper
        # multiprocessing.Event throughout).
        cv2.destroyWindow(self.window_name)

    def stop(self):
        self._exit.set()
