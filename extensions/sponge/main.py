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


async def main(health, health_speed, inventory):
    spongeSlot, spongeIndex = findPetal(inventory, "Sponge")
    if health_speed < 0:
        etaZero = health / -health_speed
        print("ETA: ", etaZero, "HP: ", health)
        if etaZero < 1.5 and spongeIndex and spongeSlot == "main":
            if spongeSlot == "main":
                print(f"Switching to Sponge at main:{spongeIndex}")
                await switchPetal(spongeIndex)
                await asyncio.sleep(0.5)  # await swap
                await switchPetal(spongeIndex)
            await asyncio.sleep(0.1)  # prevent throttling
        elif etaZero < 1.5:
            print("No Sponge found")
