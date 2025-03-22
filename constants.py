VERSION_INFO = "1.0.7"
VERSION_TYPE = "Release"
RELEASE_DATE = "2025-03-16 14:26:34"
DEFAULT_CONFIG = {
    "runningCountDown(min)": -1,
    "showLogger": False,
    "moveMouse": True,
    "useOBS": False,
    "verbose": True,
    "moveAfterAFK": False,
    "epochInterval": 8,
    "executeBinary": {
        "runBeforeAFK": "",
        "runAfterAFK": ""
    },
    "optimizeQuantization": 1,
    "rdpEpsilon": 5,
    "extendLength": 30,
    "mouseSpeed": 100,
    "skipUpdate": False,
    "environment": False,
    "windowSizeTolerance": 0.1,
    "windowSizeRatio": [
        0.787,
        1
    ],
    "yoloConfig": {
        "segModel": "./models/afk-seg.pt",
        "detModel": "./models/afk-det.pt"
    }
}
