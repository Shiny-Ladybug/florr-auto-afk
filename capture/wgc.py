import ctypes
import pygetwindow as gw
import time
import cv2
import numpy as np
from ctypes import windll
import win32gui
import win32ui
import win32print
import win32api
import win32con


def get_fixed_window_rect(hwnd):
    hDC = win32gui.GetDC(0)
    real_w = win32print.GetDeviceCaps(hDC, win32con.DESKTOPHORZRES)
    apparent_w = win32api.GetSystemMetrics(0)
    scale_ratio = real_w / apparent_w
    origin_window_rect = win32gui.GetWindowRect(hwnd)
    client_rect = win32gui.GetClientRect(hwnd)
    client_offset = win32gui.ClientToScreen(hwnd, (0, 0))
    fixed_window_rect = [
        int(client_offset[0] * scale_ratio),
        int(client_offset[1] * scale_ratio),
        int((client_offset[0] + client_rect[2]) * scale_ratio),
        int((client_offset[1] + client_rect[3]) * scale_ratio),
    ]
    return fixed_window_rect


def wgc_capture(hwnd):
    ctypes.windll.user32.SetProcessDPIAware()
    fixed_rect = get_fixed_window_rect(hwnd)
    left, top, right, bottom = fixed_rect
    w = right - left
    h = bottom - top
    hwnd_dc = win32gui.GetWindowDC(hwnd)
    mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
    save_dc = mfc_dc.CreateCompatibleDC()
    bitmap = win32ui.CreateBitmap()
    bitmap.CreateCompatibleBitmap(mfc_dc, w, h)
    save_dc.SelectObject(bitmap)
    windll.user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), 3)
    bmpinfo = bitmap.GetInfo()
    bmpstr = bitmap.GetBitmapBits(True)
    img = np.frombuffer(bmpstr, dtype=np.uint8).reshape(
        (bmpinfo["bmHeight"], bmpinfo["bmWidth"], 4))
    win32gui.DeleteObject(bitmap.GetHandle())
    save_dc.DeleteDC()
    mfc_dc.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwnd_dc)
    return img
