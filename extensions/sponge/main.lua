local function switchLoad(num)
    pyautogui.keyDown("l")
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


function main(health, health_speed, inventory)
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
