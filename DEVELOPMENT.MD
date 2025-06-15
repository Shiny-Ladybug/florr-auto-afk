# Extensions Docs

The docs are for someone who want to create their own extensions, or maybe ~~Macros~~

Anyway I want to say is comments are not allowed in JSON, the comments here are only used to let you know further about those key-value things

## Structures

> ./extensions  
> └─ demo  
>         main.lua  
>         registry.json  

### registry.json

```json
{
    "name": "demo", // the extension name has to be the same as the folder name or it will throw an error
    "description": "your description that will display on the info column",
    "author": "<YOUR NAME>",
    "version": "1.1.4.5.1.4 or other something",
    "enabled": true,
    "events": [], // to declare when the extension should run, available choices are "florrHealth" and "florrSlots"
    "args": [], // what args in Python will pass to the extension
    "imports": [] // what packages to import from Python, so that you can execute PyAutoGUI in Lua
}
```

### main.lua

```lua
function main(args) -- the args are declared in registry.json
	print(args)
    print("I am running")
end
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
        None, // null or nil in JSON and Lua
        "Egg",
        "Egg"
    ]
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
    ],
    "imports": [
        "pyautogui", // import PyAutoGUI to control keyboard via Lua
        "time" // `time` to sleep
    ]
}
```

### main.lua

```lua
-- example switch load
local function switchLoad(num)
    pyautogui.keyDown("l") -- call Python functions for you have import it in registry.json
    time.sleep(0.1)
    pyautogui.write(tostring(num))
    time.sleep(0.1)
    pyautogui.keyUp("l")
end

local function switchPetal(index)
    if index == 9 then
        pyautogui.write("0")
    else
        pyautogui.write(tostring(index + 1))
    end
    time.sleep(1)
end

local function userData2List(userdata)
    if type(userdata) == "userdata" then
        local list = {}
        for i = 1, 10 do
            local ok, value = pcall(function() return userdata[i] end)
            if ok then
                list[i] = value
            else
                break
            end
        end
        return list
    end
end

local function findPetal(inventory, name)
    if not inventory or (not inventory.main and not inventory.secondary) then return nil, nil end
    local main = userData2List(inventory.main)
    if main then
        for i, item in ipairs(main) do
            if item and item == name then
                return "main", i
            end
        end
    end
    local secondary = userData2List(inventory.secondary)
    if secondary then
        for i, item in ipairs(secondary) do
            if item and item == name then
                return "secondary", i
            end
        end
    end
    return nil, nil
end


function main(health, health_speed, inventory) -- these args are passed via Python by the registry.json
    local spongeSlot, spongeIndex = findPetal(inventory, "Sponge")
    if health_speed < 0 then
        etaZero = health / -health_speed
        print("ETA: " .. etaZero, "HP:" .. health)
        if etaZero < 1.5 and spongeIndex and spongeSlot == "main" then
            if spongeSlot == "main" then
                print("Swapping Sponge at " .. spongeSlot .. ":" .. spongeIndex)
                switchPetal(spongeIndex)
                time.sleep(0.5) -- await swap
                switchPetal(spongeIndex)
            end
            time.sleep(0.1) -- prevent throttling
        elseif etaZero < 1.5 then
            print("No Sponge found")
        end
    end
end
```

