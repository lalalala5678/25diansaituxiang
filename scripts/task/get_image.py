#!/usr/bin/env python3
"""
拍照
"""

import time
import cv2
from picamera2 import Picamera2
import numpy as np



def capture_image() -> np.ndarray:
    """使用 Picamera2 拍一张照片并返回 ndarray"""
    cam = Picamera2()
    cam.configure(cam.create_still_configuration())
    cam.start()
    time.sleep(2)                     # 预热
    frame = cam.capture_array()
    cam.close()
    return frame


if __name__ == "__main__":
    # 1) 拍摄一张图并且保存
    
    original = capture_image()
    cv2.imwrite("original.jpg", original) 

