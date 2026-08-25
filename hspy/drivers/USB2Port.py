import socket
import struct
import threading
import queue
from enum import IntEnum
import numpy as np
from hspy.drivers.HSComUSB import HS_USBCom
import time

class MsgType(IntEnum):
    CMD         = 0x01  # GUI -> Python: USB command
    CMD_RESP    = 0x02  # Python -> GUI: command response
    STREAM      = 0x03  # Python -> GUI: frame data
    STREAM_START= 0x04  # GUI -> Python: start stream
    STREAM_STOP = 0x05  # GUI -> Python: stop stream
    DEVICE_LIST  = 0x10  # Python -> GUI: device list after connect
    CALIB_CHUNK  = 0x06  # Python -> GUI: Kalibrier-Chunk
    CALIB_DONE   = 0x07  # Python -> GUI: alle Chunks gesendet
    CALIB_START  = 0x08  # GUI -> Python: start calibration read    
    CALIB_WRITE  = 0x09  # GUI -> Python: write calibration

class HTPAServer:
    def __init__(self, host='127.0.0.1', port=54321):
        self.host = host
        self.port = port
        self.devices = {}        # device_id -> HS_USBCom
        self.cmd_queues = {}     # device_id -> Queue (GUI->USB)
        self.streaming = {}      # device_id -> bool
        self.conn = None
        self.conn_lock = threading.Lock()
        # print(struct.calcsize('<BBI'))  # muss 6 sein
        # print(struct.calcsize('BBH'))  # war 4

    def send_message(self, msg_type, device_id, payload: bytes):
        header = struct.pack('<BBI', int(msg_type), device_id, len(payload))
        print(f"send_message: type=0x{int(msg_type):02X} device_id={device_id} payload_len={len(payload)} header_len={len(header)}")
        with self.conn_lock:
            try:
                self.conn.sendall(header + payload)
            except Exception as e:
                print(f"Send error: {e}")

    def recv_exact(self, n):
        buf = b''
        while len(buf) < n:
            chunk = self.conn.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("Client disconnected")
            buf += chunk
        return buf

    def device_worker(self, device_id, serial_number):
        """Ein Thread pro USB-Device"""
        with HS_USBCom(serial_number=serial_number) as com:
            self.devices[device_id] = com
            cmd_queue = self.cmd_queues[device_id]

            while True:
                if self.streaming.get(device_id):
                    datalength = self.datalength.get(device_id)
                    if datalength is None:
                        print(f"datalength not set for device {device_id}, stopping stream")
                        self.streaming[device_id] = False
                        continue
                # Stream-Modus: Frame lesen, zwischendurch Commands prüfen
                    frame = com.read_frame(datalength)
                    if frame is not None:
                        self.send_message(MsgType.STREAM, device_id, frame.tobytes())

                    # Commands non-blocking prüfen
                    try:
                        msg_type, payload = cmd_queue.get_nowait()
                        if msg_type == MsgType.STREAM_STOP:
                            self.streaming[device_id] = False
                            com.send(payload if payload else b'x')  # Stop-Command
                            # evtl. noch einen Frame lesen der zwischen STOP und Antwort gesendet wurde
                            leftover_frame = com.read_frame(self.datalength.get(device_id, 0))
                            if leftover_frame is not None:
                                self.send_message(MsgType.STREAM, device_id, leftover_frame.tobytes())                            
                            # "STOP!" Response lesen
                            stop_resp = com.read_raw_response(timeout=2000)
                            if stop_resp:
                                print(f"Stream stop response: {stop_resp.strip()}")
                            else:
                                # falls Frame und STOP! zusammen ankamen, nochmal versuchen
                                stop_resp = com.read_raw_response(timeout=1000)
                                if stop_resp:
                                    print(f"Stream stop response (2nd try): {stop_resp.strip()}")
                                
                        elif msg_type == MsgType.CMD:
                            resp = com.send_receive(payload)
                            self.send_message(MsgType.CMD_RESP, device_id,
                                            resp.encode() if resp else b'')
                    except queue.Empty:
                        pass

                else:
                    # Command-Modus: blockierend auf Commands warten
                    try:
                        msg_type, payload = cmd_queue.get(timeout=1.0)
                        if msg_type == MsgType.STREAM_START:
                            # erste 4 Bytes = frame_size, letztes Byte = command
                            frame_size = struct.unpack('I', payload[:4])[0]
                            cmd        = payload[4:5]  # b'K' oder b't'
                            
                            self.datalength[device_id] = frame_size
                            print(f"Stream start: device={device_id} frame_size={frame_size} cmd={cmd}")
                            
                            resp = com.send_receive(cmd)
                            print(f"Stream start response: {resp}")
                            self.streaming[device_id] = True
                        elif msg_type == MsgType.CMD:
                            resp = com.send_receive(payload)
                            if resp:
                                self.send_message(MsgType.CMD_RESP, device_id,
                                                resp.encode() if isinstance(resp, str) else resp)                            
                            # self.send_message(MsgType.CMD_RESP, device_id,
                            #                 resp.encode() if resp else b'')
                        elif msg_type == MsgType.CALIB_START:
                            # erste 4 Bytes = calib_size, letztes Byte = command
                            calib_size = struct.unpack('I', payload[:4])[0]
                            cmd        = payload[4:5]  # z.B. b'n'
                            chunk_size = 1024  # feste Chunk-Größe des Geräts
                            
                            com.send(cmd)
                            total_received = 0

                            while total_received < calib_size:
                                remaining = calib_size - total_received
                                read_size = min(remaining, chunk_size)  # letzter Chunk kann kleiner sein
                                chunk = com.read_frame_raw(max_bytes=read_size, timeout=3000)
                                if chunk is None:
                                    print(f"Timeout at calib read, received {total_received}/{calib_size}")
                                    break
                                if len(chunk) == 0:  # leere Chunks ignorieren
                                    continue                                
                                self.send_message(MsgType.CALIB_CHUNK, device_id, chunk)
                                total_received += len(chunk)
                            # Prüfen ob Gerät noch was schickt
                            leftover = com.read_raw_response(timeout=500)
                            if leftover:
                                print(f"Leftover after calib read: {leftover.strip()}")                                
                      
                        elif msg_type == MsgType.CALIB_WRITE:
                            calib_size = struct.unpack('I', payload[:4])[0]
                            calib_data = payload[4:]  # komplettes File im Payload
                            chunk_size = 1024

                            # Schritt 1: Write-Mode einleiten
                            resp = com.send_receive(b'Set EEPROM data\r\n', timeout=3000)
                            if resp is None or 'Setting BCC' not in resp:
                                print(f"CALIB_WRITE: unexpected response: {resp}")
                                self.send_message(MsgType.CMD_RESP, device_id, b'ERROR')
                                continue

                            # Schritt 2: Chunks senden, nach jedem auf WrCxx warten
                            total_sent = 0
                            chunk_nr   = 0
                            success    = False

                            while total_sent < calib_size:
                                remaining  = calib_size - total_sent
                                this_chunk = min(remaining, chunk_size)
                                chunk_data = calib_data[total_sent:total_sent + this_chunk]

                                com.send(chunk_data)

                                # Auf WrCxx oder "Write was successful." warten
                                ack = com.read_raw_response(timeout=5000)
                                if ack is None:
                                    print(f"Timeout waiting for ack at chunk {chunk_nr}")
                                    break

                                ack = ack.strip()
                                print(f"Device ack: {ack}")

                                if 'Write was successful' in ack:
                                    total_sent += this_chunk
                                    success = True
                                    break
                                elif 'WrC' in ack:
                                    total_sent += this_chunk
                                    chunk_nr   += 1
                                    # Fortschritt an GUI melden
                                    progress = int((total_sent / calib_size) * 100)
                                    self.send_message(MsgType.CMD_RESP, device_id,
                                                    f'PROGRESS {progress}'.encode())
                                else:
                                    print(f"Unexpected ack: {ack}")
                                    break

                            if success:
                                self.send_message(MsgType.CMD_RESP, device_id, b'WRITE_DONE')
                            else:
                                self.send_message(MsgType.CMD_RESP, device_id, 
                                                f'WRITE_ERROR {total_sent}/{calib_size}'.encode())


                        elif msg_type is None:
                            break  # shutdown
                    except queue.Empty:
                        continue

    def gui_receiver(self):
        """Empfängt Commands von der GUI"""
        while True:
            try:
                header = self.recv_exact(6)
                msg_type, device_id, payload_len = struct.unpack('<BBI', header)
                payload = self.recv_exact(payload_len) if payload_len > 0 else b''

                # print(f"gui_receiver: type=0x{msg_type:02X} device_id={device_id} payload_len={payload_len}")

                try:
                    mt = MsgType(msg_type)
                    print(f"gui_receiver: resolved to {mt}")
                except ValueError:
                    print(f"gui_receiver: unknown MsgType 0x{msg_type:02X}")

                if device_id in self.cmd_queues:
                    self.cmd_queues[device_id].put((MsgType(msg_type), payload))
                else:
                    print(f"Unknown device_id: {device_id}")

            except ConnectionError:
                print("GUI disconnected")
                break
            except Exception as e:
                print(f"Receiver error: {e}")
                break

    def send_device_list(self, serials):
        """Schickt Geräteliste nach Connect:
        Format: [count(1B)] + pro Device: [id(1B)][sn_len(1B)][sn(N Bytes)]
        """
        payload = bytes([len(serials)])
        for device_id, sn in enumerate(serials):
            sn_bytes = sn.encode('ascii')
            payload += bytes([device_id, len(sn_bytes)]) + sn_bytes

        self.send_message(MsgType.DEVICE_LIST, 0xFF, payload)            

    def run(self):
        # Devices finden und Threads starten
        serials = HS_USBCom.find_all_devices()
        if not serials:
            print("No devices found!")
            return

        print(f"Found {len(serials)} device(s): {serials}")

        # Windows braucht Zeit nach dispose_resources()
        time.sleep(0.3)        

        # Device-IDs zuweisen und Queues anlegen
        for i, sn in enumerate(serials):
            self.cmd_queues[i] = queue.Queue()
            self.streaming[i] = False
            # datalength muss noch ermittelt werden - hier vereinfacht
            self.datalength = {}

        # TCP Server starten
        # srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # srv.bind((self.host, self.port))
        # srv.listen(1)
        # print(f"Waiting for GUI on {self.host}:{self.port}...")
        # self.conn, addr = srv.accept()
        # print(f"GUI connected from {addr}")
        # self.send_device_list(serials)  # <-- sofort senden

        # Client statt Server
        print(f"Connecting to GUI on {self.host}:{self.port}...")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        # Warten bis GUI bereit ist
        while True:
            try:
                sock.connect((self.host, self.port))
                break
            except ConnectionRefusedError:
                print("GUI not ready, retrying in 1s...")
                time.sleep(1)
        
        self.conn = sock
        print("Connected to GUI")

        # Geräteliste sofort nach Connect senden
        self.send_device_list(serials)  
        # Threads erst nach weiterem kurzen Delay starten
        time.sleep(0.3)              

        # Device-Threads starten
        threads = []
        for i, sn in enumerate(serials):
            t = threading.Thread(target=self.device_worker, args=(i, sn), daemon=True)
            t.start()
            threads.append(t)
            time.sleep(0.1)  # Threads gestaffelt starten, nicht alle auf einmal

        # GUI-Empfang im Main-Thread
        self.gui_receiver()

        # Shutdown
        for q in self.cmd_queues.values():
            q.put((None, b''))
        for t in threads:
            t.join(timeout=3)
        sock.close()

if __name__ == '__main__':
    server = HTPAServer()
    server.run()