import pyautogui
import asyncio  # this is a must


async def switchPetal(index):
    if index == 9:
        pyautogui.write("0")
    else:
        pyautogui.write(str(index + 1))
    await asyncio.sleep(1)


def findPetal(inventory, name):
    if not inventory or (not inventory.get("main") and not inventory.get("secondary")):
        return None, None
    main = inventory.get("main", [])
    for i, item in enumerate(main):
        if item and item == name:
            return "main", i
    secondary = inventory.get("secondary", [])
    for i, item in enumerate(secondary):
        if item and item == name:
            return "secondary", i
    return None, None


async def main(inventory, websocket):
    for _ in range(2):
        magSlot, magIndex = findPetal(inventory, "Magnet")
        if magSlot is None:
            print("Magnet not found in inventory")
        if magSlot == "secondary":
            await switchPetal(magIndex)
            await asyncio.sleep(2)
            await switchPetal(magIndex)
        elif magSlot == "main":
            await switchPetal(magIndex)
        if _ == 0:
            await asyncio.sleep(30-43)
