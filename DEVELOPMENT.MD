# Extensions Docs (v1.3.0)

The docs are for someone who want to create their own extensions, or maybe ~~Macros~~

Anyway I want to say is comments are not allowed in JSON, the comments here are only used to let you know further about those key-value things

## Structures

> ./extensions  
> └─ demo  
>         main.py  
>         registry.json  

### registry.json

```json
{
    "name": "demo", // the extension name has to be the same as the folder name or it will throw an error
    "description": "your description that will display on the info column",
    "author": "<YOUR NAME>",
    "version": "1.1.4.5.1.4 or other something",
    "enabled": true,
    "events": [], // to declare when the extension should run, available choices are "florrHealth", "florrSlots", "florrMessages", "florrSquads" and "florrPosition"
    "args": [], // what args in Python will pass to the extension
}
```

### main.py

```python
import asyncio # must import asyncio to prevent throttling
import pyautogui # then import modules manually here

async def main(args):
    asyncio.sleep(0.1) # must use asyncio.sleep instead of time.sleep
    print(args)
```

## Internal Commands

```python
await websocket.send_json({"command": "showInfo", "info": "Hello World", "color": "#0969da50", "duration": 3000})
```

## Events & Args

### florrHealth

The Tampermonkey extension installed in your browser will upload florr health data every `1` second via WebSocket

All available variables are `health_ping` and  `health_speed`

#### health_ping: dict

Format:

```json
{"type": "florrHealth", "health": 100, "shield": 0}
```

The `health` and `shield` values are the percentage of the health

#### health_speed: float

The increase/decrease health speed of the flower

### florrSlots

The Tampermonkey extension installed in your browser will upload florr slot data every `1` second via WebSocket

All available variables are `slot_ping` and `inventory`

#### slot_ping: dict

Format:

```json
{
    "type": "florrSlots",
    "texts": [
        {
            "text": "<Petal name>",
            "x": <float>,
            "y": <float>
        },
        ...
    ]
}
```

#### inventory: dict

The inventory is auto assigned by Python with slot_ping value

It may not so accurate when one slot is empty because I have to adapt all slot numbers (for 6-10)

Format:

```json
{
    "main": [
        "Basic",
        "Basic",
        "Basic",
        "Basic",
        "Rose"
    ],
    "secondary": [
        "Egg",
        "Egg",
        None, // null in JSON
        "Egg",
        "Egg"
    ]
}
```

### florrMessages

Upload chat messages when there's a new chat messsage

All available variables are `chat_ping` and `assembled_chat`

#### chat_ping: dict

Format:

```json
{
    "type": "florrMessages",
    "content": {
        "area": "Local", // "Local", "Squad", "Guild"
        "user": "<username>",
        "message": "<text>",
        "userPosition": { // this may be null (None)
            "x": 214.35,
            "y": 124.14,
            "distance": 247.7
        }
    }
}
```

#### assembled_chat: str

Format:

```
[Local] <username>: <text>
```

### florrSquads

Positions: [0, 62000] x+, y+

The squad information

All available variable is `squad_ping`

#### squad_ping: dict

This is an 2-people squad as an example

Format:

```json
{
    "type": "florrSquads",
    "positions": [
        {
            "x": 21590.51,
            "y": 25295.6,
            "type": "self" // this is your own position
        },
        {
            "x": 28251.93,
            "y": 25392.24,
            "type": "squad" // this is one of your squad mate's position
        }
    ]
}
```

### florrPosition

Positions: [0, 62000] x+, y+

The position you are at

All available variables are `position_ping`, `self_position` and `position_speed`

#### position_ping: dict

Format:

```json
{"type": "florrPosition", "position": {"x": 21638.91, "y": 25333.28}}
```

#### self_position: tuple

Format:

```
(21638.91, 25333.28)
```

#### position_speed: float

Just float huh.

### updateTrack

Users cannot control what to track.

@server.py

```python
await websocket.send_json({"command": "track", "players": ["BF52", "IDontLikeToExist"]})
```

#### Return: dict

```json
{
    "type": "updateTrack",
    "Positions": {
        "BF52": {
            "x": -34.78,
            "y": -221.57,
            "distance": "224.28"
        },
        "IDontLikeToExist": null // None in Python
    }
}
```

## Example

An example extension of automatically switch `sponge`

### registry.json

```json
{
    "name": "sponge", // good name
    "description": "Auto switch sponge when health is low", // good description
    "author": "114514", // Yes it's 114514 not 1919810
    "version": "1.1.4", // good using semantic version
    "enabled": true, // Yes, it's turned on
    "events":[ 
        "florrHealth" // to know when to switch sponge, you have to listen on Health instead of Slot Updates
    ],
    "args": [
        "health_ping['health']", // passing Health Percentage value
        "health_speed", // to know whether it's a good time to switch sponge
        "inventory" // to know where the sponge is
    ]
}
```

### main.py

```python
import time
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
    print("Health: ", health, "Health Speed: ", health_speed)
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

```
