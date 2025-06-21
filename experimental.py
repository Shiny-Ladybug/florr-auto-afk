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


def get_config() -> dict:
    with open("./config.json", "r", encoding="utf-8") as f:
        return load(f)


def log(event: str, type: str, show: bool = True, save: bool = True):
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
    else:
        style = ""

    if show:
        console.print(logger, style=style)
    if save:
        with open('latest.log', 'a', encoding='utf-8') as f:
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
    "connected": False
}

if not exists('./extension.swap.json'):
    with open('./extension.swap.json', 'w', encoding="utf-8") as file:
        dump(default, file, indent=4)


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


def get_connected():
    return load_swap()['connected']


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


def embed_prompt(chat_ping, squad_ping, inventory, health_ping):
    now = datetime.now()
    date_text = now.strftime("%Y年%m月%d日 %H点%M分%S秒")
    squad_text = ""
    inventory_text = ""
    health_text = ""
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
        inventory_text = f'''你的物品栏主槽中有这些物品：`{", ".join(inventory["main"])}；你的物品栏副槽中有这些物品:`{", ".join(inventory["secondary"])}。'''
    if health_ping:
        health_text = f"你现在的血量为{health_ping['health']}/100。"
    message_text = f"你收到了来自{chat_ping['content']['user']}的{chat_ping['content']['area']}消息，内容为:“{chat_ping['content']['message']}”。"
    return f"现在是北京时间{date_text}。\n{squad_text}\n{inventory_text}\n{health_text}\n\n{message_text}"


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
            with open(f"./extensions/{ext}/registry.json", "r") as f:
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


def send_florr_message(message: str, scope: str):
    messages = message.split("<<SEP>>")
    for msg in messages:
        msg = f"/{scope.lower()} {filter_emoji(msg).strip()}"
        pyperclip.copy(msg)
        pyautogui.press('enter')
        sleep(0.1)
        pyautogui.hotkey(
            'ctrl', 'v', interval=0.1)
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


async def execute_task(tasklist: list, inventory: dict, chat_ping: dict, get_track_ping, websocket=None):
    goto_task_holder = {
        'task': None
    }
    for task in tasklist:
        if task['action'] == 'chat':
            send_florr_message(
                task['content'], chat_ping['content']['area'] if chat_ping else "Local")
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
            if chat_ping and task['target'] == chat_ping['content']['user']:
                if websocket is not None:
                    if goto_task_holder is not None and goto_task_holder['task'] is not None and not goto_task_holder['task'].done():
                        goto_task_holder['task'].cancel()
                    goto_task_holder['task'] = asyncio.create_task(move_to_chat_target(chat_ping)
                                                                   )
        elif task['action'] == 'follow' and get_config()['extensions']['autoChat']['fullControl']:
            await websocket.send_json({"command": "track", "players": [task['target']]})
            if websocket is not None:
                if goto_task_holder is not None and goto_task_holder['task'] is not None and not goto_task_holder['task'].done():
                    goto_task_holder['task'].cancel()
                goto_task_holder['task'] = asyncio.create_task(
                    follow_target(get_track_ping, task['target'], websocket)
                )
        await asyncio.sleep(0.1)


async def query_async(prompt: str, history: list[dict] = [], return_think=True):
    loop = asyncio.get_running_loop()
    func = functools.partial(query, prompt, history, return_think)
    return await loop.run_in_executor(None, func)
