# -*- mode: python ; coding: utf-8 -*-

import os
import onnxruntime

onnxruntime_dll_path = os.path.join(
    os.path.dirname(onnxruntime.__file__), "capi", "onnxruntime_providers_shared.dll"
)

binaries = []
if os.path.exists(onnxruntime_dll_path):
    binaries.append((onnxruntime_dll_path, './onnxruntime/capi'))

a = Analysis(
    ['./segment.py'],
    pathex=[],
    binaries=binaries,
    datas=[],
    hiddenimports=['shapely', 'pyclipper', 'pyscreeze', 'pytweening', 'pymsgbox', 'psutil', 'onnxruntime', 'plyer', 'playsound'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='segment',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    version='./file_version_info.txt',
    icon=['./gui/icon.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='segment',
)