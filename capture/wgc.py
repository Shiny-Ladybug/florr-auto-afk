from zbl import Capture
import cv2
import numpy as np
import ctypes


def wgc_capture(hwnd):
    # ctypes.windll.user32.SetProcessDPIAware()
    with Capture(window_handle=hwnd, is_cursor_capture_enabled=True) as cap:
        frame = next(cap.frames())
        img_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        img_np = np.array(img_bgr, dtype=np.uint8)
        return img_np
