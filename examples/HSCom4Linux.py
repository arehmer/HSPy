import usb.core
import hspy
import cv2
import numpy as np
from hspy.tparray import TPArray
from hspy.drivers.HSComUSB import *
print(hspy.drivers.HSComUSB.__file__)  # zeigt den exakten Pfad der geladenen Datei


with HS_USBCom() as com:
    print(com.send_receive(b'v'))
    configstr = com.send_receive(b'G')
    print(configstr)
    at = StringHelper.GetArrayType(configstr)
    htpad = TPArray(ArrayType=at)
    if htpad.DesignGen<=3:
        datalength = htpad.width*htpad.height+ (htpad.DevConst['NROFBLOCKS']* htpad.DevConst['NROFPTAT'])+(htpad.width*htpad.height//htpad.DevConst['NROFBLOCKS'])+htpad.DevConst['NROFATC']+2
    elif htpad.DesignGen==4:
        datalength = htpad.width*htpad.height+ (htpad.DevConst['NROFBLOCKS']+htpad.DevConst['NROFPTAT'])+2
    datalength+=2 #sync words 0xb0d0 0xf04g        
    #now send command to start temperature stream
    #com.send(b'K')  # Start Streaming
    print(com.send_receive(b'K'))
    while True:
        frame = com.read_frame(datalength)
        if frame is not None:
            # frame is numpy-Array with uint16-values, process it
            sensframe=frame[0:htpad.width*htpad.height]
            #print(sensframe.min(), sensframe.max())
            # 1D -> 2D reshape
            img = sensframe.reshape(htpad.height, htpad.width)
            img_normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            #optionally: Color mapping
            img_color = cv2.applyColorMap(img_normalized, cv2.COLORMAP_INFERNO)
            cv2.imshow("HTPA Stream", img_color)
        #ESC to stop
        if cv2.waitKey(1) & 0xFF == 27:
            com.send(b'x')  # Stop-Command
            break

    cv2.destroyAllWindows()   



