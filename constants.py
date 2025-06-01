import base64
ASSET_REPO = "Shiny-Ladybug/assets"
PROJECT_REPO = "Shiny-Ladybug/florr-auto-afk"
DATASET_REPO = "Shiny-Ladybug/florr-afk"
VERSION_INFO = "1.2.9"
VERSION_TYPE = "Pre-Release"
assert VERSION_TYPE in ["Release", "Pre-Release", "Dev"]
SUB_VERSION = "0"
RELEASE_DATE = "2025-06-01 15:08:00"  # 哼，我不管，你们都要准备艾草x
GITHUB_TOKEN_BASE64 = "V2pKc01HRklWbWxZTTBKb1pFWTRlRTFWU2xCT1ZVcE5VakZyZDJWdWIzZGxhM2cyVWpCMFVWUXdUbmxZTWs1TVpWVmtSMVF6U210WGFrSlZVa2RvV2xGWFVYbFNSR3gxV2tVeFNHRkdTWGRYUjA1VVRtdFNkR0V3ZHpKa2FsSkRVa2RLZWxKR1JsZFNhelZPVkhwV1ZFNUZkekZqUlZJeVQwUk9SRU5uUFQwPQ=="
# just remind that token above has only `rw` access to `only` the dataset repo, so idc if it leaks or not
GITHUB_TOKEN = GITHUB_TOKEN_BASE64
for i in range(3):
    GITHUB_TOKEN = base64.b64decode(GITHUB_TOKEN).decode("utf-8").strip()
CHANGELOG = {
    "1.2.9": ["use dijkstra for default afk path finding",
              "`start` and `end` are now inferred from exposure image",
              "fix `WWGC`",
              "model update"],
    "1.2.8": ["fix critical bug which cause thread throttling",
              "Microsoft Edge is no longer supported",
              "fix `locate_ready`"],
    "1.2.7": ["performance improvement",
              "runtime error fix for paths with no end detected"],
    "1.2.6": ["auto-upload dataset to GitHub",
              "object oriented for AFK_Segment and AFK_Path"],
    "1.2.5": ["fix starting point issue",
              "update GUI and change the notify mp3 audio",
              "notify and press 'ready' when server close or afk fail"],
    "1.2.4": ["export afk data to `./train` for semi-supervised training"],
    "1.2.3": ["fix IoU issue",
              "add moving exposure",
              "add tips",
              "I wont say if i'll stop updating from now on"],
    "1.2.2": ["literally final update ig"],
    "1.2.1": ["various launch page backgrounds qwq",
              "run button no longer cause lag",
              "notify and sound when afk detected"],
    "1.2.0": ["add background afk check support",
              "fix run button bug"],
    "1.1.1": [
        "fix GUI conflict",
        "add history"
    ]
}
