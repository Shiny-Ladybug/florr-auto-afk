from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from sys import _getframe
import asyncio
import experimental
from experimental import get_config
import extension
from datetime import datetime
from uvicorn import run
from os import path, listdir
from json import load
from rich.console import Console
from time import monotonic
import traceback

console = Console()


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


def get_installed_extensions():
    extensions = listdir("./extensions")
    response = []
    for ext in extensions:
        if path.exists(f"./extensions/{ext}/registry.json") and path.exists(f"./extensions/{ext}/main.lua"):
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


def start_extension_server():
    app = FastAPI()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.websocket(get_config()['extensions']['defaultRoute'])
    async def endpoint(websocket: WebSocket):
        await websocket.accept()
        sender = websocket.client
        log(
            f"WebSocket connection from {sender[0]}:{sender[1]} established", "EVENT")

        should_block_alpha = experimental.get_block_alpha()

        health_ping = ""
        slot_ping = ""
        health_speed = 0.0
        inventory = {"main": [], "secondary": []}
        last_health = 0.0
        last_time = monotonic()

        extensions = get_installed_extensions()

        try:
            while True:
                await asyncio.sleep(0.1)
                if experimental.get_block_alpha() != should_block_alpha:
                    log("Block alpha setting changed", "EVENT", save=False)
                    should_block_alpha = experimental.get_block_alpha()
                    await websocket.send_json({"block_alpha": should_block_alpha})

                if experimental.get_send():
                    experimental.switch_send(False)
                    await websocket.send_json({"command": "send"})
                    message = await websocket.receive_json()

                message: dict = await websocket.receive_json()
                print(message)
                if message.get("type") == "florrHealth":
                    health_ping = message
                    health = float(message['health'])
                    now = monotonic()
                    dt = now - last_time
                    if dt > 0:
                        health_speed = (health - last_health) / dt
                    last_health = health
                    last_time = now

                elif message.get("type") == "florrSlots":
                    slot_ping = message
                    inventory = parse_inventory(message['texts'])

                for ext in extensions:
                    if ext["enabled"] and message.get("type") in ext["events"]:
                        registry, code = extension.load_extension(
                            ext["name"])
                        args = []
                        for arg_expr in registry["args"]:
                            args.append(
                                eval(arg_expr, globals(), locals()))
                        extension.execute_extension(registry, code, args)

        except Exception as e:
            traceback.print_exc()
    log(
        f"Extension server started on {get_config()['extensions']['host']}:{get_config()['extensions']['port']}", "EVENT")
    run(app, host=get_config()['extensions']['host'], port=get_config()[
        'extensions']['port'], log_level="error")


if __name__ == "__main__":
    start_extension_server()
