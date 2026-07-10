import usb.core
import usb.util
import usb.backend.libusb1
import libusb_package
import re
import numpy as np
import sys

# pip install libusb
# pip install libusb-package


class HS_USBCom:
    def __init__(self, idVendor=0x32a7, idProduct=0x0003, timeout=300, serial_number=None):
        self.idVendor = idVendor
        self.idProduct = idProduct
        self.timeout = timeout
        self.serial_number = serial_number
        self.dev = None
        self.ep_in = None
        self.ep_out = None
        self.SYNC_WORD_0 = 0xB0D0
        self.SYNC_WORD_1 = 0xF046

    @staticmethod
    def find_all_devices(idVendor=0x32a7, idProduct=0x0003):
        """Returns list of all found serial numbers of the specific device"""
        backend = usb.backend.libusb1.get_backend(find_library=libusb_package.find_library)
        devices = list(usb.core.find(idVendor=idVendor, idProduct=idProduct,
                                    find_all=True, backend=backend))
        serials = []
        for d in devices:
            sn=None
            try:
                sn = usb.util.get_string(d, d.iSerialNumber, langid=0x0409)     # 0x0409 = English
                print(f"Found: VID=0x{idVendor:04X} PID=0x{idProduct:04X} SN={sn}")
            except Exception as e:
                print(f"Warning reading SN: {e}")
            finally:
                try:
                    usb.util.dispose_resources(d)  # free, will be opened in open() again
                except Exception:
                    pass
            if sn is not None:  #only pass correct serials
                serials.append(sn)
        return serials


    def open(self):
        # explicit giving Backend, should work on Windows and Linux
        backend = usb.backend.libusb1.get_backend(find_library=libusb_package.find_library)
        if backend is None:
            raise RuntimeError("libusb Backend not found")

        if self.serial_number is not None:
            # open a specific device
            devices = list(usb.core.find(idVendor=self.idVendor, idProduct=self.idProduct,
                                        find_all=True, backend=backend))
            self.dev = None
            for d in devices:
                try:
                    # set_configuration() first, then read string descriptor
                    if sys.platform != 'win32':
                        if d.is_kernel_driver_active(0):
                            d.detach_kernel_driver(0)
                    d.set_configuration()
                    sn = usb.util.get_string(d, d.iSerialNumber)
                    if sn == self.serial_number:
                        self.dev = d
                        break
                    else:
                        usb.util.dispose_resources(d)  # not the required device, free
                except Exception as e:
                    print(f"Warning at read of SN: {e}")
                    continue
        else:
            self.dev = usb.core.find(idVendor=self.idVendor, idProduct=self.idProduct, backend=backend)
            # Kernel-driver only relevant for Linux
            if sys.platform != 'win32':
                if self.dev.is_kernel_driver_active(0):
                    self.dev.detach_kernel_driver(0)
                # #access must be granted, create rule: 
                # #sudo nano /etc/udev/rules.d/99-htpa.rules
                # #content: SUBSYSTEM=="usb", ATTRS{idVendor}=="32a7", ATTRS{idProduct}=="0003", MODE="0666"                    
            self.dev.set_configuration()

        if self.dev is None:
            raise ValueError("Device not found! (VID=0x{:04X}, PID=0x{:04X}, SN={})".format(
                self.idVendor, self.idProduct, self.serial_number))

        # Endpoints ermitteln
        cfg = self.dev.get_active_configuration()
        intf = cfg[(0, 0)]

        self.ep_out = usb.util.find_descriptor(intf,
            custom_match=lambda e:
                usb.util.endpoint_direction(e.bEndpointAddress) == usb.util.ENDPOINT_OUT)

        self.ep_in = usb.util.find_descriptor(intf,
            custom_match=lambda e:
                usb.util.endpoint_direction(e.bEndpointAddress) == usb.util.ENDPOINT_IN)

        if self.ep_out is None or self.ep_in is None:
            raise ValueError("Bulk-Endpoints not found")

        print("Connected: ep_out=0x{:02X}, ep_in=0x{:02X}".format(
            self.ep_out.bEndpointAddress, self.ep_in.bEndpointAddress))


    def close(self):
        if self.dev is None:
            return
        try:
            usb.util.release_interface(self.dev, 0)
            if sys.platform != 'win32':            
                try:
                    self.dev.attach_kernel_driver(0)
                except Exception:
                    pass  # no Kernel-driver present
        finally:
            usb.util.dispose_resources(self.dev)
            self.dev = None
            self.ep_in = None
            self.ep_out = None

    def send_receive(self, cmd, timeout=None):
        if self.dev is None:
            raise RuntimeError("Device not opened!")

        t = timeout if timeout is not None else self.timeout

        try:
            self.ep_out.write(cmd if isinstance(cmd, bytes) else cmd.encode())
            answer = self.ep_in.read(1024, timeout=t)
            return bytes(answer).decode('utf-8', errors='replace')

        except usb.core.USBTimeoutError:
            print("Timeout - clearing endpoint...")
            try:
                self.dev.clear_halt(self.ep_in)
                self.dev.clear_halt(self.ep_out)
            except Exception:
                pass
            return None

        except usb.core.USBError as e:
            print("USB error: {} - trying reset...".format(e))
            try:
                self.dev.reset()
                self.dev = usb.core.find(idVendor=self.idVendor, idProduct=self.idProduct)
                self.dev.set_configuration()
            except Exception:
                pass
            return None
        

    def send(self, cmd):
        """Sends command without waiting for an answer"""
        if self.dev is None:
            raise RuntimeError("Device not open")
        self.ep_out.write(cmd if isinstance(cmd, bytes) else cmd.encode())

    def read_frame(self, data_length, timeout=None):
        """Reads binary frame and returns numpy-Array with uint16"""
        t = timeout if timeout is not None else self.timeout
        byte_count = data_length * 2  # 16bit = 2 Bytes 
        max_packet = self.ep_in.wMaxPacketSize

        # try:
        #     raw = self.ep_in.read(byte_count, timeout=t)
        #     return np.frombuffer(bytes(raw), dtype=np.uint16)
        try:
            raw = bytearray()
            remaining = byte_count        
            while remaining > 0:
                chunk_size = min(remaining, max_packet * 64)  # reasonable chunk size
                chunk = self.ep_in.read(chunk_size, timeout=t)
                raw.extend(chunk)
                remaining -= len(chunk)
            
            if len(raw) != byte_count:
                print(f"Warning: Expected {byte_count} bytes, got {len(raw)}")
                return None
            
            frame = np.frombuffer(bytes(raw), dtype=np.uint16)   
            # Sync-Check
            if frame[-2] != self.SYNC_WORD_0 or frame[-1] != self.SYNC_WORD_1:
                print(f"Sync-Error: Expected 0x{self.SYNC_WORD_0:04X} 0x{self.SYNC_WORD_1:04X}, "
                    f"got 0x{frame[-2]:04X} 0x{frame[-1]:04X}")
                return None

            return frame[:-2]  # ohne Sync-Wörter                 

        except usb.core.USBTimeoutError:
            print("Timeout at frame read")
            try:
                self.dev.clear_halt(self.ep_in)
            except Exception:
                pass
            return None

        except usb.core.USBError as e:
            print("USB error at frame-read: {}".format(e))
            return None        

    # Context Manager Support
    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False



class StringHelper:
    def GetArrayType(str):
        match = re.search(r'Arraytype\s+(\d+)', str)
        if match:
            array_type = int(match.group(1))
            return array_type
        else:
            print("ArrayType not found")
            return 0xFF
        
    