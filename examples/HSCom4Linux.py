import threading
import queue
import usb.core
import hspy
import cv2
import numpy as np
from hspy.tparray import TPArray
from hspy.drivers.HSComUSB import HS_USBCom, StringHelper


stop_event = threading.Event()
frame_queue = queue.Queue(maxsize=10)  # maxsize prevents overflow when display methods are slow

def device_thread(serial_number, window_name):
    with HS_USBCom(serial_number=serial_number) as com:
        print(f"[{serial_number}] {com.send_receive(b'v')}")
        configstr = com.send_receive(b'G')
        # print(configstr)
        at = StringHelper.GetArrayType(configstr)
        htpad = TPArray(ArrayType=at)

        if htpad.DesignGen <= 3:
            datalength = (htpad.width * htpad.height
                          + (htpad.DevConst['NROFBLOCKS'] * htpad.DevConst['NROFPTAT'])
                          + (htpad.width * htpad.height // htpad.DevConst['NROFBLOCKS'])
                          + htpad.DevConst['NROFATC'] + 2)
        elif htpad.DesignGen == 4:
            datalength = (htpad.width * htpad.height
                          + (htpad.DevConst['NROFBLOCKS'] + htpad.DevConst['NROFPTAT']) + 2)
        datalength += 2     #sync words 0xb0d0 0xf04g  

        #now send command to start temperature stream
        print(com.send_receive(b'K'))

        while not stop_event.is_set():
            frame = com.read_frame(datalength)
            if frame is not None:
                # frame is numpy-Array with uint16-values, cut off irrelevant data for image, process it. eventually chip temperature could be extracted if of interest
                sensframe = frame[0:htpad.width * htpad.height]
                #print(sensframe.min(), sensframe.max())
                # 1D -> 2D reshape                
                img = sensframe.reshape(htpad.height, htpad.width)
                img_normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                #optionally: Color mapping
                img_color = cv2.applyColorMap(img_normalized, cv2.COLORMAP_INFERNO)
                try:
                    frame_queue.put_nowait((window_name, img_color))  # non-blocking
                except queue.Full:
                    print("Frame skipped")
                    pass  # Display is to slow, skip it

        com.send(b'x')

# Find all devices
serials = HS_USBCom.find_all_devices()
print(f"serials: {serials}")  
# filter None
serials = [s for s in serials if s is not None]

if not serials:
    print("No devices found!")
    exit()

print(f"{len(serials)} Device(s) found: {serials}")

# Start a thread per device
threads = []
for i, sn in enumerate(serials):
    t = threading.Thread(target=device_thread, args=(sn, f"HTPA {sn}"), daemon=True)
    t.start()
    threads.append(t)

# Main-Thread: Here imshow and waitKey 
while True:
    # show all pending frames from the queue
    while not frame_queue.empty():
        window_name, img = frame_queue.get_nowait()
        cv2.imshow(window_name, img)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        stop_event.set()
        break

for t in threads:
    t.join(timeout=3)

cv2.destroyAllWindows()




