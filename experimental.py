import extension
import numpy as np
import pyperclip
from itertools import combinations
import math
import asyncio
from sys import _getframe
from datetime import datetime
from os import path, listdir
import requests
from json import load, dump, loads
from os.path import exists
from time import time, sleep
from rich.console import Console
import pyautogui
import re
from random import choice
import asyncio
import functools


console = Console()
ALL_SERVERS = []


def get_config() -> dict:
    with open("./config.json", "r", encoding="utf-8") as f:
        return load(f)


def log(event: str, type: str, show: bool = True, save: bool = True, chat: bool = False):
    back_frame = _getframe().f_back
    if back_frame is not None:
        back_filename = path.basename(back_frame.f_code.co_filename)
        back_funcname = back_frame.f_code.co_name
        back_lineno = back_frame.f_lineno
    else:
        back_filename = "Unknown"
        back_funcname = "Unknown"
        back_lineno = "Unknown"
    now = datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    logger = f"[{time}] <{back_filename}:{back_lineno}> <{back_funcname}()> {type}: {event}"
    if type.lower() == "info":
        style = "green"
    elif type.lower() == "error":
        style = "red"
    elif type.lower() == "warning":
        style = "yellow"
    elif type.lower() == "critical":
        style = "bold red"
    elif type.lower() == "event":
        style = "#ffab70"
    elif type.lower() == "chat":
        style = "#d46183"
    else:
        style = ""

    if show:
        console.print(logger, style=style)
    if save:
        with open('latest.log', 'a', encoding='utf-8') as f:
            f.write(f'{logger}\n')
    if chat:
        with open('./chat.log', 'a', encoding='utf-8') as f:
            f.write(f'{logger}\n')


last_request = 0
last_cache = None


def load_history():
    with open('./conversation.json', 'r', encoding="utf-8") as file:
        history = load(file)
    for message in history:
        message["content"] = message["content"].replace(
            "{{username}}", get_config()['extensions']["autoChat"]['selfUsername'])
    return history


history = load_history()

default = {
    "block_alpha": False,
    "send": False,
    "connected": False,
    "aspac": False
}


def initiate_swap():
    if not exists('./extension.swap.json'):
        with open('./extension.swap.json', 'w', encoding="utf-8") as file:
            dump(default, file, indent=4)
            return
    with open('./extension.swap.json', 'w', encoding="utf-8") as file:
        dump(default, file, indent=4)
        return


def load_swap():
    global last_request, last_cache
    if time() - last_request > get_config()['extensions']['swapInterval']:
        last_request = time()
        with open('./extension.swap.json', 'r', encoding="utf-8") as file:
            data = load(file)
        last_cache = data
        return data
    else:
        return last_cache


last_cache = load_swap()


def format_response(resp: str):
    try:
        answer = loads(resp)
        return answer
    except:
        if "```" in resp:
            resp = resp.replace(
                "```json\n", "").replace("\n```", "")
        try:
            answer = loads(resp)
            return answer
        except:
            try:
                answer = eval(resp)
                return answer
            except:
                return None


def set_connected(type: bool):
    with open('./extension.swap.json', 'r', encoding="utf-8") as file:
        data = load(file)
        data['connected'] = type

    with open('./extension.swap.json', 'w', encoding="utf-8") as file:
        dump(data, file, indent=4)


def get_connected() -> bool:
    return load_swap()['connected']


def get_aspac() -> bool:
    return load_swap()['aspac']


def set_aspac(type: bool):
    with open('./extension.swap.json', 'r', encoding="utf-8") as file:
        data = load(file)
        data['aspac'] = type

    with open('./extension.swap.json', 'w', encoding="utf-8") as file:
        dump(data, file, indent=4)


def query(prompt: str, history: list[dict] = [], return_think=True):
    if get_config()['extensions']['autoChat']['chatType'] == "random":
        response = choice(
            get_config()['extensions']['autoChat']['chatRandomResponses'])
        messages = history.copy()
        response = '{"action": "chat", "content": "' + response + '"}'
        messages.append({"role": "assistant", "content": response})
        return response, messages

    url = get_config()['extensions']['autoChat']['chatEndpoint']
    headers = {
        "Content-Type": "application/json"
    }
    if get_config()['extensions']['autoChat']['chatAPIKey'] != "<YOUR_API_KEY>":
        headers["Authorization"] = f"Bearer {get_config()['extensions']['autoChat']['chatAPIKey']}"
    messages = history.copy()
    if not get_config()['extensions']['autoChat']["chatEnableThink"]:
        prompt += " /no_think"
    messages.append({"role": "user", "content": prompt})
    payload = {
        "model": get_config()['extensions']['autoChat']['chatModel'],
        "messages": messages,
        "stream": False,
        "keep_alive": get_config()['extensions']['autoChat']["chatKeepAlive"]
    }
    resp = requests.post(
        url, json=payload, headers=headers)
    if resp.status_code != 200:
        raise Exception(
            f"Request failed with status code {resp.status_code}: {resp.json()}")
    response_json = resp.json()
    if get_config()['extensions']['autoChat']['chatType'] == "openai":
        response = response_json['choices'][0]['message']['content']
    elif get_config()['extensions']['autoChat']['chatType'] == "ollama":
        response = response_json['message']['content']
    messages.append({"role": "assistant", "content": response})
    if "</think>" in response and not return_think:
        response = response.split("</think>")[-1].strip()
    return response, messages


def embed_prompt(message_stack, squad_ping, inventory, health_ping):
    now = datetime.now()
    date_text = now.strftime("%Y年%m月%d日 %H点%M分%S秒")
    squad_text = ""
    inventory_text = ""
    health_text = ""
    message_text = ""
    if squad_ping:
        squad_count = len(squad_ping["positions"])
        if squad_count == 1:
            squad_text += f"你现在在一个组队中，但是只有自己一个人。"
        elif 1 < squad_count < 4:
            squad_text += f"你现在在一个组队中，队伍中有{squad_count}人，还有{4 - squad_count}个空位。"
        elif squad_count == 4:
            squad_text += f"你现在在一个组队中，队伍中有{squad_count}人，已经满了。"
    else:
        squad_text += "你现在没有组队。"
    if inventory:
        inventory_text = f'''你的物品栏主槽中有这些物品：`{", ".join(inventory["main"]) if inventory["main"] else "空"}；你的物品栏副槽中有这些物品：`{", ".join(inventory["secondary"]) if inventory["secondary"] else "空"}。'''
    if health_ping:
        health_text = f"你现在的血量为{health_ping['health']}/100。"
    for chat_ping in message_stack:
        message_text += f"你收到了来自{chat_ping['content']['user']}的{chat_ping['content']['area']}消息，内容为:“{chat_ping['content']['message']}”。\n"
    return f"现在是北京时间{date_text}。\n{squad_text}\n{inventory_text}\n{health_text}\n\n{message_text}".strip()


def switch_block_alpha(type: bool):
    with open('./extension.swap.json', 'r', encoding="utf-8") as file:
        data = load(file)
        data['block_alpha'] = type

    with open('./extension.swap.json', 'w', encoding="utf-8") as file:
        dump(data, file, indent=4)


def get_block_alpha():
    return load_swap()['block_alpha']


def switch_send(type: bool):
    with open('./extension.swap.json', 'r', encoding="utf-8") as file:
        data = load(file)
        data['send'] = type

    with open('./extension.swap.json', 'w', encoding="utf-8") as file:
        dump(data, file, indent=4)


def get_send():
    return load_swap()['send']


def get_installed_extensions():
    extensions = listdir("./extensions")
    response = []
    for ext in extensions:
        if path.exists(f"./extensions/{ext}/registry.json") and path.exists(f"./extensions/{ext}/main.py"):
            with open(f"./extensions/{ext}/registry.json", "r", encoding="utf-8") as f:
                registry = load(f)
            if registry["name"].strip() == ext.strip():
                response.append(registry)
            else:
                log(
                    f"Extension \"{registry['name']}\" registry name mismatch folder name \"{ext}\"", "ERROR", save=False)
    return response


def assign_petals_to_slots(petals, slots_num, median_delta_x):
    petals_sorted = sorted(petals, key=lambda p: p['x'])
    if not petals_sorted:
        return [None] * slots_num
    start_x = petals_sorted[0]['x']
    half_delta = median_delta_x / 2
    slots = []
    for i in range(slots_num):
        target_x = start_x + i * median_delta_x
        found = None
        for p in petals_sorted:
            if abs(p['x'] - target_x) <= half_delta:
                found = p['text']
                break
        slots.append(found)
    return slots


def parse_inventory(inventory):
    inventory_ = {}
    slot_main = []
    slot_secondary = []
    odd_flag = False
    for petal in inventory:
        petal['x'] = round(petal['x'], 2)
        petal['y'] = round(petal['y'], 2)
        if petal['y'] < -50:
            if petal['x'] == 0:
                odd_flag = True
            slot_main.append(petal)
        else:
            slot_secondary.append(petal)

    slot_main.sort(key=lambda x: x['x'])
    delta_x = []
    if len(slot_main) > 1:
        for i in range(1, len(slot_main)):
            delta_x.append(
                slot_main[i]['x'] - slot_main[i-1]['x'])
        median_delta_x = sorted(delta_x)[len(delta_x) // 2]
        if median_delta_x != 0:
            if odd_flag:
                slots_num = round(-slot_main[0]
                                  ['x']/median_delta_x)*2 + 1
            else:
                slots_num = round(-slot_main[0]
                                  ['x']/median_delta_x)*2 + 2
            inventory_["main"] = assign_petals_to_slots(
                slot_main, slots_num, median_delta_x
            )[:10]
    else:
        inventory_["main"] = assign_petals_to_slots(
            slot_main, len(slot_main), 0
        )

    slot_secondary.sort(key=lambda x: x['x'])
    delta_x_sec = []
    if len(slot_secondary) > 1:
        for i in range(1, len(slot_secondary)):
            delta_x_sec.append(
                slot_secondary[i]['x'] -
                slot_secondary[i-1]['x']
            )
        median_delta_x_sec = sorted(delta_x_sec)[
            len(delta_x_sec)//2]
        if median_delta_x_sec != 0:
            odd_flag_sec = any(p['x'] == 0 for p in slot_secondary)
            if odd_flag_sec:
                slots_num_sec = round(
                    -slot_secondary[0]['x']/median_delta_x_sec
                )*2 + 1
            else:
                slots_num_sec = round(
                    -slot_secondary[0]['x']/median_delta_x_sec
                )*2 + 2
            inventory_["secondary"] = assign_petals_to_slots(
                slot_secondary, slots_num_sec, median_delta_x_sec
            )[:10]
    else:
        inventory_["secondary"] = assign_petals_to_slots(
            slot_secondary, len(slot_secondary), 0
        )
    return inventory_


def distance(a, b):
    return math.hypot(a['x'] - b['x'], a['y'] - b['y'])


def midpoint(a, b):
    return {'x': (a['x'] + b['x']) / 2, 'y': (a['y'] + b['y']) / 2}


def fermat_point(a, b, c):
    def angle(u, v, w):
        ux, uy = u['x'], u['y']
        vx, vy = v['x'], v['y']
        wx, wy = w['x'], w['y']
        a = math.hypot(ux - vx, uy - vy)
        b = math.hypot(ux - wx, uy - wy)
        c = math.hypot(vx - wx, vy - wy)
        return math.acos((a*a + b*b - c*c) / (2*a*b))
    A = a
    B = b
    C = c
    angles = [angle(A, B, C), angle(B, A, C), angle(C, A, B)]
    for i, ang in enumerate(angles):
        if ang >= 2 * math.pi / 3:
            return [A, B, C][i]
    sqrt3 = math.sqrt(3)
    x = (A['x'] + B['x'] + C['x'] + sqrt3 * (A['y'] -
         B['y'] + B['y'] - C['y'] + C['y'] - A['y'])) / 3
    y = (A['y'] + B['y'] + C['y'] + sqrt3 * (B['x'] -
         A['x'] + C['x'] - B['x'] + A['x'] - C['x'])) / 3
    return {'x': x, 'y': y}


def suggest_position(squad_data, threshold=1000):
    positions = squad_data['positions']
    self_pos = next(p for p in positions if p['type'] == 'self')
    mates = [p for p in positions if p['type'] == 'squad']
    if len(mates) < 2:
        return None

    if len(mates) == 3:
        close_pairs = []
        for a, b in combinations(mates, 2):
            if distance(a, b) < threshold:
                close_pairs.append((a, b))
        if len(close_pairs) == 3:
            return fermat_point(mates[0], mates[1], mates[2])
        elif len(close_pairs) >= 1:
            a, b = close_pairs[0]
            return midpoint(a, b)
        else:
            return None
    elif len(mates) == 2:
        return midpoint(mates[0], mates[1])
    else:
        return None


def reset_keyboard():
    pyautogui.keyUp('w')
    pyautogui.keyUp('a')
    pyautogui.keyUp('s')
    pyautogui.keyUp('d')


async def move_to_position(get_self_pos, websocket, target_pos, stop_distance=150, min_tolerance=20):
    while True:
        self_pos, self_velocity = get_self_pos()
        if self_pos is None or target_pos is None:
            await asyncio.sleep(0.1)
            continue
        dx = target_pos[0] - self_pos[0]
        dy = target_pos[1] - self_pos[1]
        dist = (dx**2 + dy**2) ** 0.5

        if dist < stop_distance + max(50, self_velocity * 30):
            reset_keyboard()
            if dist < stop_distance:
                break

        if dx > 0:
            if abs(dx) < min_tolerance:
                pyautogui.keyUp('d')
                pyautogui.press('d', interval=0.05)
            else:
                pyautogui.keyUp('a')
                pyautogui.keyDown('d')
        elif dx < 0:
            if abs(dx) < min_tolerance:
                pyautogui.keyUp('a')
                pyautogui.press('a', interval=0.05)
            else:
                pyautogui.keyUp('d')
                pyautogui.keyDown('a')
        if dy > 0:
            if abs(dy) < min_tolerance:
                pyautogui.keyUp('s')
                pyautogui.press('s', interval=0.05)
            else:
                pyautogui.keyUp('w')
                pyautogui.keyDown('s')
        elif dy < 0:
            if abs(dy) < min_tolerance:
                pyautogui.keyUp('w')
                pyautogui.press('w', interval=0.05)
            else:
                pyautogui.keyUp('s')
                pyautogui.keyDown('w')

        await asyncio.sleep(0.05)

    reset_keyboard()
    await websocket.send_json({"command": "switchInterval", "interval": 1000})
    return


def re_match(text: str, patterns: list[str]) -> str:
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return True
    return False


def filter_emoji(desstr, restr=''):
    try:
        co = re.compile(u'[\U00010000-\U0010ffff]')
    except re.error:
        co = re.compile(u'[\uD800-\uDBFF][\uDC00-\uDFFF]')
    return co.sub(restr, desstr)


def send_florr_message(message: str, scope: str, user: str):
    messages = message.split("<<SEP>>")
    for msg in messages:
        if scope.lower() != "whisper":
            msg = f"/{scope.lower()} {filter_emoji(msg).strip()}"
            pyperclip.copy(msg)
            pyautogui.press('enter')
            sleep(0.1)
            pyautogui.hotkey(
                'ctrl', 'v', interval=0.1)
            sleep(0.1)
            pyautogui.press('enter')
        else:
            msg = f"/{scope.lower()} {user} {filter_emoji(msg).strip()}"
            pyperclip.copy(msg)
            pyautogui.press('enter')
            sleep(0.1)
            pyautogui.hotkey(
                'ctrl', 'v', interval=0.1)
            sleep(0.1)
            pyautogui.press('enter')


def send_florr_command(command: str):
    pyperclip.copy(command)
    pyautogui.press('enter')
    sleep(0.1)
    pyautogui.hotkey('ctrl', 'v', interval=0.1)
    sleep(0.1)
    pyautogui.press('enter')


def find_target_petal(inventory_, target_text):
    inventory = inventory_.copy()
    target_text = target_text.strip().lower()
    inventory['main'] = [petal.lower()
                         for petal in inventory['main'] if petal is not None]
    inventory['secondary'] = [petal.lower()
                              for petal in inventory['secondary'] if petal is not None]

    def calculate_levenshtein_distance(pending, needle):
        m, n = len(pending), len(needle)
        dp = np.zeros((m + 1, n + 1))

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pending[i - 1] == needle[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(dp[i - 1][j], dp[i][j - 1],
                                   dp[i - 1][j - 1]) + 1
        return dp[m][n]/len(pending) if len(pending) > 0 else float('inf')

    target_text = target_text.strip()
    best_matches = []
    best_distance = float('inf')
    for idx, petal in enumerate(inventory['main']):
        if petal is None:
            continue
        distance = calculate_levenshtein_distance(petal, target_text)
        if distance <= best_distance:
            best_distance = distance
            best_matches.append(
                {'text': petal, 'slot': idx, 'type': 'main', 'distance': distance})

    for idx, petal in enumerate(inventory['secondary']):
        if petal is None:
            continue
        distance = calculate_levenshtein_distance(petal, target_text)
        if distance <= best_distance:
            best_distance = distance
            best_matches.append(
                {'text': petal, 'slot': idx, 'type': 'secondary', 'distance': distance})

    best_matches.sort(key=lambda x: x['distance'])
    min_dist = best_matches[0]['distance'] if best_matches else float('inf')
    best_matches = [
        match for match in best_matches if match['distance'] == min_dist]
    return best_matches


async def move_to_chat_target(chat_ping, steps=get_config()['extensions']['autoChat']['fullControl']['steps'], velocity=400):
    if chat_ping and chat_ping['content']['userPosition'] is not None:
        rel_x = chat_ping['content']['userPosition']['x']
        rel_y = chat_ping['content']['userPosition']['y']
        if rel_x == 0 and rel_y == 0:
            return

        key_x = 'd' if rel_x > 0 else 'a'
        key_y = 's' if rel_y > 0 else 'w'

        angle = math.atan2(abs(rel_y), abs(rel_x))
        x_steps = int(steps * math.cos(angle))
        y_steps = int(steps * math.sin(angle))
        duration = (rel_x**2 + rel_y**2)**0.5 / velocity
        for i in range(max(x_steps, y_steps)):
            if i < x_steps:
                pyautogui.keyDown(key_x)
            if i < y_steps:
                pyautogui.keyDown(key_y)
            await asyncio.sleep(duration / steps)
            if i < x_steps:
                pyautogui.keyUp(key_x)
            if i < y_steps:
                pyautogui.keyUp(key_y)


async def follow_target(get_track_ping, target, websocket, steps=get_config()['extensions']['autoChat']['fullControl']['steps'], velocity=400, duration_=get_config()['extensions']['autoChat']['fullControl']['followDuration']):
    await websocket.send_json({"command": "switchInterval", "interval": 200})
    start_time = time()
    while True:
        track_ping = get_track_ping()
        if target not in list(track_ping.get('Positions', {}).keys()):
            await asyncio.sleep(0.5)
            continue
        if track_ping['Positions'][target] is None:
            await asyncio.sleep(1)
            continue
        rel_x = track_ping['Positions'][target]['x']
        rel_y = track_ping['Positions'][target]['y']
        key_x = 'd' if rel_x > 0 else 'a'
        key_y = 's' if rel_y > 0 else 'w'

        angle = math.atan2(abs(rel_y), abs(rel_x))
        x_steps = int(steps * math.cos(angle))
        y_steps = int(steps * math.sin(angle))
        duration = (rel_x**2 + rel_y**2)**0.5 / velocity
        for i in range(max(x_steps, y_steps)):
            if i < x_steps:
                pyautogui.keyDown(key_x)
            if i < y_steps:
                pyautogui.keyDown(key_y)
            await asyncio.sleep(duration / steps)
            if i < x_steps:
                pyautogui.keyUp(key_x)
            if i < y_steps:
                pyautogui.keyUp(key_y)
        if time() - start_time > duration_:
            break
    await websocket.send_json({"command": "switchInterval", "interval": 1000})


async def move_direction(direction: str, duration: float):
    if direction == "left":
        pyautogui.keyDown("a")
        await asyncio.sleep(duration)
        pyautogui.keyUp("a")
    elif direction == "right":
        pyautogui.keyDown("d")
        await asyncio.sleep(duration)
        pyautogui.keyUp("d")
    elif direction == "up":
        pyautogui.keyDown("w")
        await asyncio.sleep(duration)
        pyautogui.keyUp("w")
    elif direction == "down":
        pyautogui.keyDown("s")
        await asyncio.sleep(duration)
        pyautogui.keyUp("s")


async def execute_task(tasklist: list, inventory: dict, message_stack: list, get_track_ping, websocket=None):
    goto_task_holder = {
        'task': None
    }
    for task in tasklist:
        if task['action'] == 'chat':
            send_florr_message(
                task['content'], message_stack[-1]['content']['area'], message_stack[-1]['content']['user'])
        elif task['action'] == 'inv' and get_config()['extensions']['autoChat']['fullControl']:
            petals = find_target_petal(inventory, task['petal'])
            if petals:
                if task['type'] == 'use':
                    for petal in petals:
                        if petal['type'] == 'secondary':
                            pyautogui.press(
                                f"{petal['slot'] + 1 if petal['slot'] != 9 else '0'}")
                elif task['type'] == 'stop':
                    for petal in petals:
                        if petal['type'] == 'main':
                            pyautogui.press(
                                f"{petal['slot'] + 1 if petal['slot'] != 9 else '0'}")
        elif task['action'] == 'goto' and get_config()['extensions']['autoChat']['fullControl']:
            if message_stack[-1] and task['target'] == message_stack[-1]['content']['user']:
                if websocket is not None:
                    if goto_task_holder is not None and goto_task_holder['task'] is not None and not goto_task_holder['task'].done():
                        goto_task_holder['task'].cancel()
                    goto_task_holder['task'] = asyncio.create_task(move_to_chat_target(message_stack[-1])
                                                                   )
        elif task['action'] == 'follow' and get_config()['extensions']['autoChat']['fullControl']:
            await websocket.send_json({"command": "track", "players": [task['target']]})
            if websocket is not None:
                if goto_task_holder is not None and goto_task_holder['task'] is not None and not goto_task_holder['task'].done():
                    goto_task_holder['task'].cancel()
                goto_task_holder['task'] = asyncio.create_task(
                    follow_target(get_track_ping, task['target'], websocket)
                )
        elif task['action'] == 'join_squad' and get_config()['extensions']['autoChat']['fullControl']:
            await websocket.send_json({"command": "showInfo", "info": f"Join squad: {task['code']}", "color": "#d5167c75", "duration": 3000})
            send_florr_command(f"/squad-join {task['code']}")
        elif task['action'] == 'move' and get_config()['extensions']['autoChat']['fullControl']:
            await move_direction(task['direction'], 1)
        await asyncio.sleep(0.1)


async def query_async(prompt: str, history: list[dict] = [], return_think=True):
    loop = asyncio.get_running_loop()
    func = functools.partial(query, prompt, history, return_think)
    return await loop.run_in_executor(None, func)


def check_ollama():
    if get_config()['extensions']['autoChat']['chatType'] == "ollama":
        endpoint = get_config()['extensions']['autoChat']['chatEndpoint']
        endpoint = endpoint.replace("/api/chat", "/api/tags")
        try:
            resp = requests.get(endpoint)
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                for model in models:
                    if model['name'] == get_config()['extensions']['autoChat']['chatModel']:
                        log(f"Found Ollama model: {model['name']}",
                            "INFO", save=False)
                        return True
                log(f"Model {get_config()['extensions']['autoChat']['chatModel']} not found on Ollama server. Run `ollama pull {get_config()['extensions']['autoChat']['chatModel']}` to download it.",
                    "ERROR", save=False)
                return False
            else:
                log("Ollama server is not running or not reachable.",
                    "ERROR", save=False)
                return False
        except requests.RequestException as e:
            log(f"Error checking Ollama server: {e}", "ERROR", save=False)
            log("Please check if Ollama is running and accessible.",
                "ERROR", save=False)
            log("Visit https://ollama.com/download to download Ollama.",
                "ERROR", save=False)
            return False


async def send_notify(websocket, title, info):
    await websocket.send_json({"command": "showNotification", "title": title, "info": info, "icon": "data:image/png;base64,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", "duration": 3000})


def match_cron_field(field_val, current_val):
    if field_val == '*':
        return True
    if field_val.startswith('*/'):
        step = int(field_val[2:])
        return current_val % step == 0
    return str(current_val) == field_val


def check_schedule(cron_expr, now=None):
    if not cron_expr or not isinstance(cron_expr, str):
        return False
    if now is None:
        now = datetime.now()

    fields = cron_expr.strip().split()
    if len(fields) != 5:
        return False

    minute, hour, dom, mon, wday = fields
    mapping = [
        (minute, now.minute),
        (hour, now.hour),
        (dom, now.day),
        (mon, now.month),
        (wday, now.isoweekday() % 7)
    ]

    for cron_val, current_val in mapping:
        if not match_cron_field(cron_val, current_val):
            return False
    return True


async def update_servers_loop():
    global ALL_SERVERS
    MATRIX = [
        {"id": "map-0", "endpoint": "florrio-map-0", "tags": {"dungeon": "garden"}},
        {"id": "map-1", "endpoint": "florrio-map-1", "tags": {"dungeon": "desert"}},
        {"id": "map-2", "endpoint": "florrio-map-2", "tags": {"dungeon": "ocean"}},
        {"id": "map-3", "endpoint": "florrio-map-3", "tags": {"dungeon": "jungle"}},
        {"id": "map-4", "endpoint": "florrio-map-4",
            "tags": {"dungeon": "ant_hell"}},
        {"id": "map-5", "endpoint": "florrio-map-5", "tags": {"dungeon": "hel"}},
        {"id": "map-6", "endpoint": "florrio-map-6", "tags": {"dungeon": "sewers"}},
        {"id": "map-7", "endpoint": "florrio-map-7", "tags": {"dungeon": "factory"}},
        {"id": "map-8", "endpoint": "florrio-map-8", "tags": {"dungeon": "pyramid"}},
    ]
    server_keys = {
        "vultr-miami": "us",
        "vultr-frankfurt": "eu",
        "vultr-tokyo": "as"
    }
    while True:
        servers = []
        for server in MATRIX:
            found = {"us": None, "eu": None, "as": None}
            tries = 0
            while tries < 5 and not all(found.values()):
                endpoint = f"https://api.n.m28.io/endpoint/{server['endpoint']}-green/findEach/"
                tags = server['tags']
                try:
                    resp = await asyncio.to_thread(requests.get, endpoint, params=tags)
                    if resp.status_code == 200:
                        data = resp.json()
                        servers_in_response = data.get('servers', {})
                        for key, val in servers_in_response.items():
                            name = server_keys.get(key)
                            if name and not found[name]:
                                found[name] = val.get('id')
                    else:
                        await asyncio.sleep(1)
                        tries += 1
                        continue
                except Exception:
                    await asyncio.sleep(1)
                    tries += 1
                    continue
                await asyncio.sleep(1)
                tries += 1
            for name, sid in found.items():
                if sid:
                    servers.append({
                        "id": sid,
                        "name": name,
                        "map": server["id"],
                        "mapName": server["tags"]["dungeon"]
                    })
        ALL_SERVERS = servers
        await asyncio.sleep(10)


def find_server_by_id(server_id, all_servers=None):
    if all_servers is None:
        all_servers = ALL_SERVERS
    if not all_servers:
        return {"id": server_id, "name": None, "map": None, "mapName": None}
    for server in all_servers:
        if server['id'] == server_id:
            return server
    return {"id": server_id, "name": None, "map": None, "mapName": None}
