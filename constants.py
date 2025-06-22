import base64
ASSET_REPO = "Shiny-Ladybug/assets"
PROJECT_REPO = "Shiny-Ladybug/florr-auto-afk"
DATASET_REPO = "Shiny-Ladybug/florr-afk"
VERSION_INFO = "1.3.0"
VERSION_TYPE = "Pre-Release"
assert VERSION_TYPE in ["Release", "Pre-Release", "Dev"]
SUB_VERSION = "6"
RELEASE_DATE = "2025-06-22 12:22:00"
GITHUB_TOKEN_BASE64 = "VEZNd2RFeFRNVU5TVldSS1ZHbENVMVV3UldkVlJrcEtWbXRHVlZKVFFreFNWbXQwVEZNd2RFeFJjRTVUVld4R1lqSmtTbEZyUmtKVE1FNUNWVlZXUW1KdGJEVk5XRVpaWkRCa1dXSlhiM0pPYlZKUFdUSk9hR1ZGYjNKWFYyeHFWa2RvY2s1c1ZYcGpiRkozVGxjNWFGUnNSbkJSVjJSSVlsaGtkMkZ1VWxCRGFtUkVUVmN4U2s0elVsUmFhbFpRVTJrNE1HRXpVbFJXV0UxNVkwUmFNVXQ2VGtSYWFrNVhZVWhvUldWc2FHaFpNV3d4VFd4a1JGa3pRak5XZWtreVdsVkZkMlJIZUVkT1ZYQkxVV3hqZVdGc1ZqSlVWbEZMVkdwb1VsRlVWa05QUjJoeVlrVm9Ra3g2WkdoaFZGSTFZVmRhVEZReVJrdGliR1JZVG5wS2JWWnJTbEpXV0VaUVRIcE9RbE16YUhoT1NFMDBUVVpKZVdGSGIzaFdiVkpxVTFVNVVVMUdhRmhUV0ZWNVRIZHdkR05yVW0xTU1ERlFaVWhDVms1SWJFSmhSRW8xVGxjMWVXSnFaRVJVYkZJeFdUSTRlVmt3ZEVOU2JFSm9WbTVGTVZaclRrbE5WRVpDV1d4R01HVnJNSGRYUlRWQ1VtMVpNVTlGWkVOV1JURkdXVlpXVEVOdE9XcGlSV1JxWVVkM2VWbHBkRVZTUlZwYVUxZHpNMkl5YkhCUFJXUmFVVzVrVG1OSVVrOVhSR3Q0WlcwNU5rMXVVbFJWYWtwelZFaGthbEV4Um5Wak1EZ3pWa1ZXV0ZKR2NHNWpWMmgwWld0T2VFNUlUVXRYVjNOeVMzbHpNbGx0VlhsWGJGSndWRlpzZDA5WGRIUlpWR1JEVmtVeGVsWkVZM3BoYld3MlVrWktSVTB5YkRKaE1VWktVa1ZHVWxGVlNrSmlNR3hEVVZWU01VMVVUbEpPUms1RFQwWlpkMUpzWkdsaVozQjNaVmM1TlZWVGRIaGxhM041VGxWR1JVOUhiR2xrTVdoTVpHNVNhMlZxVGxWVE0wNHdWRE5PWVZKcWJGZFNXRXBDWVcxYVZFNXVaM3BTYW1oSlRsVjBkMk5GYUZwaVNGSlhWRk01Um1Sc2FERmlNREZGUTJ4R1NsWlVaSE5XTVdRMlVsaE9WRTB5UmpSamJHczBUVVprUlZORmVFdFBWa1ozWkZVMGNrdDZUa0pYUms0MldUQlNVbUl4UmpSaWJWWmhUakZPV0ZscmRFbGpSMlJUVFZaV1IxWlhkRmxaTTFKcVVtMVJTMDFyVW05bFIzUlJVMFpXTUZwdWJEVmlWR040WTBob01WVnBkRWhrUjNCUFYwVTFNRTlXU21Ga1dGRTFVWHBWZDFsVWJFTmFWMUpUVmtkU2FrNVlTbkZhTVhBd1ZVVktjbHByVWxGWGJFbzFUbWwwUlZKM2NHdFZWMDV3WVd4c2NHRnVSalZoUXpselV6Rm9UbVF4U210T1JscFhXbGN3TTFwSVVUTlJiRll5VWxSU1ZFOUhOVkJTTW5nelVURktiRXN4U2pSTE1uaFdWbFZ3VFUwelFtbGtWV2MxWVZoR2QyUXhRWEpEYmxvMVVUQkdhazlIT0RGVU1tOHlWMWRzV1dWdVVtMVNSRXB3WWxab1YxTnJjSEJXTUhReFdXMWFSMUpIUlRGUmJrWkdaVlU1Y1ZsdWFFZGlSM1JRWlcxb1JVOUZUa3hpUmsxNVUxaHZlbFJ0VW5CalZWbExaRzFLYkZkV1ZqUlNWVTV1VjFWV1FtVnJSbkJWUmxwYVdYcE5NbE5FUVhaU00xVjVaRWhTVTFsclJuZGxWbHBZVFRObmRsRXpaRWRsVkZsM1VrWk9ObVJyWjNaUFUzUkZWbGROZWxwSWJHOWlWMGwyVFhkd2NXVlVUbGxQUlhSMlRsWktiMk13Y0ZOVFJFazBVVmhvU2xWdVFYSlJXRTF5VEhwU1VGbHBkRVpOTUd4dlRsVldVbVJXVWs5TmJFWndXV3RHY0ZGdVJtNVRSRVkwWkRBeE1sTnJXa3hVYldnMlpFWldTRU5zV2s5U1JXdDNXa1JhTmxFeGFFVmlSRWx5VWpOT05sRllXa2xPVjNNMFRURldORnA2V1hoVFJtUlBVM3BWY21ReWFHMU5hMmgyV210V2FHUnJWVEpQV0Zrd1YxVlpOVTE2YUVSYU1XeEdVVmhvZFZkVk5FdGlibXhMWWpKa01WVXlUbTFOVjBWM1lubDBiVmxZV25OTk1XUjRWRVJXYlZOR1FrMVhWRVZ5VmxSYWFGZFVXakZUYkdoaFUxaFZOV1JZYUd4V01IQXhWWHBvZVZaWFpGQlRSRnB4VVhwU1RFMUdaRkJOVVhCeVlUQjBNbHBWV2toU1EzTXdZek53VmxOWGRGQlhSbHBvWWxkU2IwNVlTbWxaYmtwaFlWWlZjazF1WkRaTmVtZDJZVWRLZUUxWGRIcFVlWFJIVjI1S2NWWnFWakpSTW5SaFN6QTFNRlZWY0VKaGVrMHlRMnBDYzFZeVpHMVZiV2hSWTBoT01WcHVRbGhqYTFKWFZUSkdRMDlXWkVwVlJFNDBZVlJTTUU5VlJucFZNbEpEWTBac1VFOUZUbTVYVlVaVlVtNXNkV1ZWYkhsaGExWnJVV3M1V1Zwc1kzbGliRzkyVFd4alMyTkhhRkJMTTJoT1YxVkdTbGxVUWt0U2JXeFhWRVZrYTA1cVJtMWpWWE13WldwUmRrd3hTbnBqYmxwRFkxZE9VMlZXY0RGWlZra3lWbXhPVjAxRlduSlVNRTV3VmpOa1NrNTZRVFZWV0djMFpIcEtTVmRuY0ZCTmF6aDJZMjVDUldFd2JHbE5lVGxPVlVWYVNGTkZhR2hqTVVsNVZsVmFTVlpzU25OYVYxWkdZVWhvZFZwRlVqVlBSMFpPWkVSV2JGTXhRWGRQVjFveFVUTkJNRkpIYkhOamEwNVJWbXhzVGs0eVZrZERibVJZV2tVNE1HRlVUalZqUmtwRVZVWmFiVTR3VWtaVWF6QjVVakZHVEZGdFpFUk5TRXBVVmxad2NHVlhNREphTUd4YVRsVnZNbGxZVFhkWGJWcEZWbXBTZVdKSFVsQmpNRTVaVDFaR2VGUXhRbUZTV0dOTFpXdE9kV1ZYZUcxVWEzZ3hWbFpTZVZWdGNHcGxSRTR3VlVWa2RHUXdPSGRqU0ZwMFkwVTRkbGRYWjNaTlZtaHRVbXBaTTAxSVJUSldWRUkyVTBaR2Qwc3djRVpSYmsxNVlsaHNSbFpIY0ZWV1JHaEhUa0Z3UzJKRlRteFhTRVkxVlVaR1RsWkVhRFJpUm1oVVltdGtlazFzVGtOa1dFWnRZMnR3ZVdKR1NUSmxWWEIwWVd4T1ZrNXFXbXhrTWtaT1dqSmFXV1ZZV1hsa2JtaEVXVlprUzFkdFZtOVNWR016VVcxc1VrTnVWbmhqZWxaQ1lqQmtRbEpVVVRGWk1uY3pUakJPYzJKVlRrZFJiRnAyVFVoQ1ZXRkVaR0ZUYlVweFZqQndjMW94UmpOTk1qVkdaRzB3ZUVzeFJucFNWMnhFV1RCVk0yUjZSbXRWVkdnMVYyeENibU5IYTB0V1JrRjVVVlZzYkdOclRrTlRla2t5WVRGb1dVNUVUVEprVldSb1pFaENkbEpYYkhKa2JXTjVZV3hPUmxKV1RrUlRSVVp1V2pGUk1sVnNhSFZpUnpSNFV6TmFUVk5JV25WT2FsbDJWbTFXYVUxVk1YTldRWEEwVFVaSmVHRkhNSGRXTVVwU1RqQTVkRTV1YnpGamVtUjJZMVJvYkZWdE5WUmtWVkpUVFZWMGRrMXFWa3hsVkVKdlUwZEdhbGRYYUd0aVYwWnJXVzFHY2xreWN6bERhVEIwVEZNd2RGSlZOVVZKUmtwVVVWTkNVVlZyYkZkUlZsSkdTVVYwUmxkVE1IUk1VekIw"
# just remind that bot private key above has only content `rw` access to `only` the dataset repo, so idc if it leaks or not
GITHUB_TOKEN = GITHUB_TOKEN_BASE64
for i in range(3):
    GITHUB_TOKEN = base64.b64decode(GITHUB_TOKEN).decode("utf-8").strip()
CHANGELOG = {
    "1.3.0": ["add custom extension (auto switch Sponge as demo)",
              "extension release (chat, background remove while afk, etc)",
              "install extension.js via Tampermonkey to enable extension client",
              "ollama server is required to deploy manually",
              "add ai chat"
              ],
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
