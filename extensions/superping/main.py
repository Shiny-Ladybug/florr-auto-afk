import pyautogui
import asyncio  # this is a must
import pyperclip

keywords = {
    "A tower of thorns rises from the sands...": "Cactus",
    "You hear someone whisper faintly...\"just... one more game...\"": "Gambler",
    "You hear lightning strikes coming from a far distance...": "Jellyfish",
    "Something mountain-like appears in the distance...": "Rock",
    "There's a bright light in the horizon": "Firefly",
    "A big yellow spot shows up in the distance...": "Hornet",
    "A buzzing noise echoes through the sewer tunnels": "Fly",
    "You sense ominous vibrations coming from a different realm...": "Hel"
}


async def send_florr_message(message: str, scope: str, user: str):
    messages = message.split("<<SEP>>")
    for msg in messages:
        if scope.lower() != "whisper":
            msg = f"/{scope.lower()} {msg.strip()}"
            pyperclip.copy(msg)
            pyautogui.press('enter')
            await asyncio.sleep(0.1)
            pyautogui.hotkey(
                'ctrl', 'v', interval=0.1)
            await asyncio.sleep(0.1)
            pyautogui.press('enter')
        else:
            msg = f"/{scope.lower()} {user} {msg.strip()}"
            pyperclip.copy(msg)
            pyautogui.press('enter')
            await asyncio.sleep(0.1)
            pyautogui.hotkey(
                'ctrl', 'v', interval=0.1)
            await asyncio.sleep(0.1)
            pyautogui.press('enter')


async def main(chat_ping, server, websocket):
    if chat_ping["content"]["area"] == "$system":
        mob = None
        region = server["name"]
        if not region:
            return
        if "A Super " in chat_ping["content"]["message"] and " has spawned" in chat_ping["content"]["message"]:
            mob = chat_ping["content"]["message"].split(
                "A Super ")[1].split(" has spawned")[0]
        elif keywords.get(chat_ping["content"]["message"]):
            mob = keywords[chat_ping["content"]["message"]]

        if mob:
            print(f"Super mob detected: {mob} in {region}")
            await send_florr_message(
                f"{region} {mob}", "guild", None)
