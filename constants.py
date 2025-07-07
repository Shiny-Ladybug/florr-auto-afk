import base64
ASSET_REPO = "Shiny-Ladybug/assets"
PROJECT_REPO = "Shiny-Ladybug/florr-auto-afk"
DATASET_REPO = "Shiny-Ladybug/florr-afk"
VERSION_INFO = "1.3.2"
VERSION_TYPE = "Release"
assert VERSION_TYPE in ["Release", "Pre-Release", "Dev"]
SUB_VERSION = "0"
RELEASE_DATE = "2025-07-07 12:27:33"
GITHUB_TOKEN_BASE64 = "VEZNd2RFeFRNVU5TVldSS1ZHbENVMVV3UldkVlJrcEtWbXRHVlZKVFFreFNWbXQwVEZNd2RFeFJjRTVUVld4R1lqSmtTbEZyUmtKVE1FNUNWVlZXUW1JeVNrZGxWbVJWVFVoT1VGVjZiRFJqYWtaV1ducHNUV0Z0TUhoV1JXYzFWMGRrTW1Nd1VsWk9iRWx5VGtSU1YxZHFVbFZrU0UwMVkwaEtVbGRyT1RCRGFrNVVWVEI0YTJFeVpHdFJiRUUwVjIwNU1WRllSbWhSTTJSMlZWWkdTVTVVV25SWk1qVnFWMGN4YUZKRWJISk5NRGxQWlZSU1YxUnNiRmxXVjNRMFZGUldlbUZVWkU5T1JscHZZVVJzUmxSNlZqRk9WVmxMWlVSR1RHTldSbEpUUkVwWlRsYzVlazV0T0hwVWVtd3hUakJHUTFKRldsRmtlbGw2WkVoV1VsVnFRak5VTTJoVlRXdEdiMlZJWTNsVU1Wa3pZbFZOZUZkWVJrOVVNa1pUVm14Sk5WbFZOVWRpVlRWWlMzZHdhVlJYYkRKVFJGcHFZMnQwU0ZaVlRtaFRhMUp1WVZadmQxb3pWbk5UTWxwV1YyMDBNbFJWVmxGaVJtaE1ZVVpCZDFaVVpFdFRWR2hFWVVoU1lVMXBkRXRYYkU1TFkydE9NMDFXVG5SVFZtOTNZVWhSZDBOdGFHdFZibFpHVFRKR2JtRnRTWGhUVm14TVlsZEdURXg2U2pGV1ZtUkRZbXBDY1Uwd1ZYbGhhemx2Vmtaa01XSldjR3hoVjFaWlVXeEdRbUo2Umt4YVZVbDVWMFJrVTAxSWIzbGFhMDVDWkRCa1UxUkhkMHRXYkZwRVRqSTVhVkl5ZURKaWFteFdZVzVHTm1GSVVuVlJNR1JDWkZVMGVtSnJTa2RsYlRsMFdtcEdhVmRXY0RKVmJFWktVa1ZHVWxGVlNrSmlNR3hEVVZWb01GZFhTa0psVjBwWFQwVjNkazVyVlhKUmQzQlBaV3M0TUZkV2F6SlBWWEJwVlhwYWNVNUdhRkZVUlVwMFlXcEZNMlF6YkdwV1dFSkdVbGhDU1dGRVVrVk5NVlp2VlVkS2NsTXdSVFJUYms0MlkydGtkR0pGTVVOWmEyUXhUbXBTUlZaSVJrbE1NSE41UTIxV2VtSXlaRkZPTWpsSVRWZHJORlZ0Um5kV1ZWSldUbFJhTkV3eVZUVmpXRkowVFROYVJtUkhkSEZYYmxaUVYxaE9iV1F3Y0ZWV1JYTTBUa1ZHZVZKdWFEQk9SM1JaWlZWd1ZXSjZXVFZPUm1oUVUydHJTMHd5VlhKTlZFcEpXbTVzYUZVeGNGQlVSRmx5VG1wU2RGUXpWbTlTVlVZMFYwWmtZV1JJWkhOYU1rVjJVek5hUTA1dGQzaFdSbU13VWpOU2RGSXlXbE5UZW14NlRURmpjbFV3T1U1T2JXaFJUV3BCZDJKbmNFNWFibkJoVTNwYU0yTlZNSGhSYlhCUVVWWnNSbGRyVW1wTlZsRnlWVVJPVFdWdVVuQlRSazR3VDBoSmVsSklWbEpaYld4VVlWVktkR0pyT1Raa1JrWkpZa1JPYlZscWEzWk9NalZRVmpOQmVVMURPVlJEYW1Sc1VtMTBhazlHU2pWaFIxWk5XbFZhYjJJeGJEUmFWa1l4VDBaQ2FXTnJUa1ZYVmxwSlV6QldSV0pXY0hKaGExWTJUREpXZFZKVlVUVmhTRkpxWVVaTk1VeDZWazFrYkVaNlQxVldNR1ZzV2pOVlJsVkxZMVZ3ZEU1WFJsbGhNRTV1VjFWV1FtVnRhM0pWTWxwWVVXdDRXVkl4VVhaUmFrcFlWV3hHTlZveVdrdE5SR2hMWTNwR1dsUnFSbTFqVmxaeVVsUldkbUZyWkZOYU0wWnNZMnhrVG1WdFpHaFJWR1JvVmxGd1drMXJkRWhpUXpnd1RVY3hiVkl4YnpGTU0xSndUbTB3TUdJeU9XNU9WM1JXWW01b1VWUjZhSHBWTVhCdllUQmpNbGRIZHpST01sWnNXakZPYldOcWF6VlZiVFZRVlVkR1dFOUhVWGhWYlZFMVUzcEpja05yWkVSbFZXczBZMWRHZDAxVlkzbGpWV2hJVTBoYU0wNHhXak5aVm14S1ZsWldUV0l4YnpSaVJsa3lXbTA0ZVZSNlVYZFZiRkpoVGtaV1RWUnJXbXhVVm5CclRqSndiMWR1VGtSYU1XeEdVVmhzVGxOVmIwdGthVGxWWlc1a05XSkZaRVpOVkZwM1lWaFdVbVJVVmtOWFYwNWFUVWRyZUdGR1dsbGlTRVpIVlVOMFMyUkZXVFZoVkZKT1ZsYzBNbG96U2sxWFJYZDNXWHBPVkdGRmRIZFdiV2QyVTJ0S2FHTXhXa0pPVVhCUVdrZGthRTlYYzNaV1dHeEdUbGh3VjJSRWFGZGxSVTVPWlZka2EwNVhXakJMTVVKUFVXMHhZV05JWkRCV2ExcFpZMVZPTmxOSVNrdGlWRTVzV2taQ1UxZHJVbWxUYTNodlZFWndRbEpXY0hKbFIyaHBRMnR3YTFOR1ZsaE5iV2hRWkZoV1VGSkZOREZOTVdST1ZUQjBVMkY2UWxsa1YyeFVZVzFrYWxreWREUlBWVWwzWVRJMVMwOUZUbTVYVlVwcFUwaFNWbFJIYkdGaWJYUnRVbXRvY21WcE9VNWxSM0JwVXpOVlMwNXNjRkpqZWxwSFZVUk9Ra3N4YUZSVFJXeFZVVzVhZVU1dGVHeFdSRkl4WTBoQ1dGRlhOVXRoYkhCeFpXNUNWRlZVV25oTlYxWnRaRVJzYTFaRlRsUk5SVXAyWkZNNVIxUnJValpVTUd4NVZtcGFWRk4zYnpSVFIwWktZVmRrVGswd1drcE5WbWhEVTJ0S2MxWXdiRVpOZWxWNFRUTnNSbU5JV2pKVlJsSlhZekpOTUdJeFpGUmpWMUY0Vm14QmVVd3phM2hsU0ZJMllsaEdiR0ZIY0hKT2EzZDJWWHBHY1dWWFRqVkRiVkpMV1dwU1VrMUZlSEpOVjFvMFZGaENkbEZyZEhoVlZsVnlVa2hrVEZGdFpFWldiV3N3Wld0b1NsVkZlRkZPUmtaUlZGWk5NVmxyU2xCYVZFNUxaVzVhYkU1RlpEUlNWRlpLVlRBNE0yTnFRazVsVjI5TFRURk9NR05FV1RKYWJWVjZWRlprUTJKRk1VaFVNM0F4VW10Uk0yRnJVWHBsUkZaRVREQTFTVm96VG5STGVrSkZUVEJvTTA1dVZsTmxWVGxXVVc1c1UxUXlTbFJrYTJ4dlVWWktkRlJ1WkdwWlZFNVBXbEZ3YkZWRmNIQmtNakZaV1ZoS1ZXRXdZekpTYm1SWlRXcHNXbFF3ZEVoaFNHaERWREJTVjFNemNGQmFNR1JWVVcxR2MxcEZORFJVUlRnelQxVk5lbUpyVGtKTE1VNVBUbXRHZVZkcVdsRlNWWE13VDBoa1NFTnRaSFpNTTBwQ1lqQmtRbFpVU2s5TE1IQnpVakprTVZGcVJYcE5NbWgxWkRCV2NXUnRaSGRPUjJneFlsUmFWbUZZVFRWVGJXUlhWVmRGZUZReGFFZE5ha28yVFd4b1QwOVdTa1pUUm13eFpHMTBWMk5JU1V0VlZsWlZVVE5PYWs1VlZqTlZSa0pEVlZWNGEwNXRXbWhUYTJnMFRsVktNVXN6YkZaYWJFNTZUVVZvVTFWNU9WTmlSbVJJWkVoRmQySnVRbFJXYVRrMlUwVk9Oa3N3U2tkWmF6VnRXVmhaZWxsdFVYSmpRWEF3WVc1S2QwNHljM0pTVm1SeVVXNVdUazVGVmxCbFZrNUZWbFZvTWxaVk1VbFdSbVJTVG0wNU1XSllXWGRUV0d3MFdYcE9TMVpXVGpOU1ZVNTNZbFpTYWxGV2F6bERhVEIwVEZNd2RGSlZOVVZKUmtwVVVWTkNVVlZyYkZkUlZsSkdTVVYwUmxkVE1IUk1VekIw"
# just remind that bot private key above has only content `rw` access to `only` the dataset repo, so idc if it leaks or not
GITHUB_TOKEN = GITHUB_TOKEN_BASE64
for i in range(3):
    GITHUB_TOKEN = base64.b64decode(GITHUB_TOKEN).decode("utf-8").strip()
CHANGELOG = {
    "1.3.2": [
        "update remote dataset upload type",
        "add OpenCV-Pro method",
        "add more extension schedule support"
    ],
    "1.3.1": ["add special background [爱心][星星眼]",
              "add move, join squad action",
              "add so-called aspac",
              "fix extension while 2k resolution",
              "better response action (debounce)"],
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
