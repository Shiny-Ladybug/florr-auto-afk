import time
from ctypes import windll, byref, c_ubyte
from ctypes.wintypes import RECT, HWND
import numpy as np
import cv2
import pygetwindow as gw

GetDC = windll.user32.GetDC
CreateCompatibleDC = windll.gdi32.CreateCompatibleDC
GetClientRect = windll.user32.GetClientRect
CreateCompatibleBitmap = windll.gdi32.CreateCompatibleBitmap
SelectObject = windll.gdi32.SelectObject
BitBlt = windll.gdi32.BitBlt
SRCCOPY = 0x00CC0020
GetBitmapBits = windll.gdi32.GetBitmapBits
DeleteObject = windll.gdi32.DeleteObject
ReleaseDC = windll.user32.ReleaseDC

windll.user32.SetProcessDPIAware()


def bitblt_capture(handle: HWND):
    hDC = GetDC(0)
    real_w = windll.gdi32.GetDeviceCaps(hDC, 118)  # DESKTOPHORZRES
    apparent_w = windll.user32.GetSystemMetrics(0)  # SM_CXSCREEN
    scale_ratio = real_w / apparent_w
    r = RECT()
    GetClientRect(handle, byref(r))
    width, height = r.right, r.bottom

    # 修正宽度和高度以考虑 DPI 缩放
    width = int(width * scale_ratio)
    height = int(height * scale_ratio)

    # 创建兼容的 DC 和位图
    dc = GetDC(handle)
    cdc = CreateCompatibleDC(dc)
    bitmap = CreateCompatibleBitmap(dc, width, height)
    SelectObject(cdc, bitmap)

    # 执行 BitBlt 操作
    BitBlt(cdc, 0, 0, width, height, dc, 0, 0, SRCCOPY)

    # 获取位图数据
    total_bytes = width * height * 4
    buffer = bytearray(total_bytes)
    byte_array = c_ubyte * total_bytes
    GetBitmapBits(bitmap, total_bytes, byte_array.from_buffer(buffer))

    # 清理资源
    DeleteObject(bitmap)
    DeleteObject(cdc)
    ReleaseDC(handle, dc)

    # 转换为 NumPy 数组
    img = np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 4)

    return img
