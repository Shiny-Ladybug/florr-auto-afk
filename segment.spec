# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ['./segment.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=['shapely', 'pyclipper', 'pyscreeze', 'pytweening', 'pymsgbox', 'psutil', 'playsound', 'win11toast', 'fastapi', 'uvicorn.lifespan.off', 'uvicorn.lifespan.on', 'uvicorn.lifespan', 'uvicorn.protocols.websockets.auto', 'uvicorn.protocols.websockets.wsproto_impl', 'uvicorn.protocols.websockets_impl', 'uvicorn.protocols.http.auto',
                   'uvicorn.protocols.http.h11_impl', 'uvicorn.protocols.http.httptools_impl',
                   'uvicorn.protocols.websockets', 'uvicorn.protocols.http', 'uvicorn.protocols',
                   'uvicorn.loops.auto', 'uvicorn.loops.asyncio', 'uvicorn.loops.uvloop', 'uvicorn.loops',
                   'uvicorn.logging', 'pyperclip'],
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
    version='./file_version_info_ex.txt',
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
