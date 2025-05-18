ASSET_REPO = "Shiny-Ladybug/assets"
PROJECT_REPO = "Shiny-Ladybug/florr-auto-afk"
DATASET_REPO = "Shiny-Ladybug/florr-afk"
VERSION_INFO = "1.2.6"
VERSION_TYPE = "Pre-Release"
assert VERSION_TYPE in ["Release", "Pre-Release", "Dev"]
SUB_VERSION = "1"
RELEASE_DATE = "2025-05-18 13:53:00"
GITHUB_TOKEN = "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"
# just remind that token above has only `rw` access to `only` the dataset repo, so idc if it leaks or not
CHANGELOG = {
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
