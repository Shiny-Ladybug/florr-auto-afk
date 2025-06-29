from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import experimental
from experimental import get_config
import extension
from uvicorn import run
from time import monotonic
from starlette.websockets import WebSocketDisconnect
from experimental import log
import traceback
import time
import constants
from json import load, dumps
from contextlib import asynccontextmanager

current_path_task = None
connected = False


def start_extension_server():
    app = FastAPI()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    experimental.set_connected(False)

    @app.websocket(get_config()['extensions']['defaultRoute'])
    async def endpoint(websocket: WebSocket):
        await websocket.accept()
        sender = websocket.client
        log(
            f"WebSocket connection from {sender[0]}:{sender[1]} established", "EVENT")
        global connected
        connected = True
        experimental.set_connected(True)

        should_block_alpha = experimental.get_block_alpha()
        health_ping = {}
        slot_ping = {}
        chat_ping = {}
        squad_ping = {}
        position_ping = {}
        track_ping = {}

        track_list = []
        assembled_chat = ""
        health_speed = 0.0
        inventory = {"main": [], "secondary": []}
        last_health = 0.0
        last_health_time = monotonic()
        self_position: tuple[float, float] = (None, None)
        last_position: tuple[float, float] = (None, None)
        last_position_time = monotonic()
        position_speed = 0.0
        squad_position: list[tuple[float, float]] = []

        message_stack = []
        debounce_task = None

        async def process_message_stack(websocket):
            nonlocal message_stack
            if message_stack:
                log(f"Responding to {len(message_stack)} messages", "EVENT")
                prompt = experimental.embed_prompt(
                    message_stack, squad_ping, inventory, health_ping)
                response, experimental.history = await experimental.query_async(
                    prompt, experimental.history, return_think=False)

                if len(experimental.history) > get_config()['extensions']['autoChat']['historyMaxLength']:
                    experimental.history.pop(1)

                if response:
                    response = experimental.format_response(
                        response)
                    if response:
                        task_t = time.time()
                        log(f"Queried {task_t} tasks: {response}", "INFO")

                        def get_track_ping():
                            return track_ping
                        for task in response:
                            await experimental.send_notify(
                                websocket, "Task", f"Executing {task}")
                        await experimental.execute_task(
                            response, inventory, message_stack, get_track_ping, websocket)

                        log(f"{task_t} tasks executed", "EVENT")
                message_stack = []

        async def process_message(websocket):
            await asyncio.sleep(get_config()['extensions']['autoChat']['chatCooldown'])
            await process_message_stack(websocket)

        try:
            while True:
                extensions = experimental.get_installed_extensions()
                if experimental.get_block_alpha() != should_block_alpha:
                    log("Block alpha setting changed", "EVENT", save=False)
                    should_block_alpha = experimental.get_block_alpha()
                    await websocket.send_json({"block_alpha": should_block_alpha})

                if experimental.get_send():
                    experimental.switch_send(False)
                    await websocket.send_json({"command": "send"})
                    message = await websocket.receive_json()
                message: dict = await websocket.receive_json()

                if message.get("type") == "florrHealth":
                    health_ping = message
                    health = float(message['health'])
                    now = monotonic()
                    dt = now - last_health_time
                    if dt > 0:
                        health_speed = (health - last_health) / dt
                    last_health = health
                    last_health_time = now

                elif message.get("type") == "florrSlots":
                    slot_ping = message
                    inventory = experimental.parse_inventory(message['texts'])

                elif message.get("type") == "florrMessages":
                    chat_ping = message
                    assembled_chat = f"[{chat_ping['content']['area']}] {chat_ping['content']['user']}: {chat_ping['content']['message']}"
                    log(assembled_chat, "CHAT", save=False)
                    if get_config()['extensions']['autoChat']["enable"]:
                        if get_config()['extensions']['autoChat']['selfUsername'] != "enter <username> here or chat will respond to your own messages":
                            if chat_ping['content']['area'] in get_config()['extensions']['autoChat']['chatScope'] and chat_ping['content']['user'] != get_config()['extensions']['autoChat']['selfUsername']:
                                if (get_config()['extensions']['autoChat']["chatMaxDistance"] and chat_ping['content']['userPosition'] is not None and chat_ping['content']['userPosition']['distance'] < get_config()['extensions']['autoChat']["chatMaxDistance"]) or not get_config()['extensions']['autoChat']["chatMaxDistance"]:
                                    if (get_config()["extensions"]['autoChat']['chatWhitelist'] != [] and experimental.re_match(chat_ping['content']['user'], get_config()["extensions"]['autoChat']['chatWhitelist'])) or (get_config()["extensions"]['autoChat']['chatWhitelist'] == [] and not experimental.re_match(chat_ping['content']['user'], get_config()["extensions"]['autoChat']['chatBlacklist'])):
                                        message_stack.append(chat_ping)
                                        await experimental.send_notify(websocket, chat_ping['content']['user'], chat_ping['content']['message'])
                                        log(
                                            f"Message added to stack: {assembled_chat}", "EVENT", save=False)
                                        if debounce_task and not debounce_task.done():
                                            debounce_task.cancel()
                                        debounce_task = asyncio.create_task(
                                            process_message(websocket))

                elif message.get("type") == "florrSquads":
                    global current_path_task
                    message['positions'] = [
                        x for x in message['positions'] if x['x'] >= 0 and x['y'] >= 0]
                    if position_ping and len(message['positions']) > 0:
                        dist = float('inf')
                        for idx, member in enumerate(message['positions']):
                            if ((member['x']-self_position[0])**2+(member['y']-self_position[1])**2) < dist:
                                dist = ((member['x']-self_position[0])
                                        ** 2+(member['y']-self_position[1])**2)
                                if idx != 0:
                                    message['positions'][idx -
                                                         1]['type'] = "squad"
                                    message['positions'][idx]['type'] = "self"
                                else:
                                    message['positions'][idx]['type'] = "self"
                            else:
                                message['positions'][idx]['type'] = "squad"
                        squad_ping = message
                        squad_position = message['positions']

                        if get_config()['extensions']["autoCalibrate"]:
                            best_pos = experimental.suggest_position(
                                squad_ping)
                            best_pos = (
                                best_pos['x'], best_pos['y']) if best_pos else None
                            if best_pos:
                                if (not current_path_task or current_path_task.done() or
                                        getattr(current_path_task, "target_pos", None) != best_pos):
                                    if current_path_task and not current_path_task.done():
                                        current_path_task.cancel()
                                        experimental.reset_keyboard()

                                    def get_self_velocity():
                                        return (self_position[0], self_position[1]), position_speed
                                    await websocket.send_json({"command": "switchInterval", "interval": 200})
                                    current_path_task = asyncio.create_task(
                                        experimental.move_to_position(get_self_velocity, websocket, best_pos))
                                    current_path_task.target_pos = best_pos
                            else:
                                if current_path_task:
                                    current_path_task.cancel()
                                    experimental.reset_keyboard()
                                    current_path_task = None

                elif message.get("type") == "florrPosition":
                    position_ping = message
                    self_position = (
                        message['position']['x'],
                        message['position']['y']
                    )
                    now = monotonic()
                    if last_position[0] is not None and last_position[1] is not None:
                        dt = now - last_position_time
                        dx = self_position[0] - last_position[0]
                        dy = self_position[1] - last_position[1]
                        dist = (dx ** 2 + dy ** 2) ** 0.5
                        if dt > 0:
                            position_speed = dist / dt
                    last_position = self_position
                    last_position_time = now

                elif message.get("type") == "updateTrack":
                    track_ping = message

                elif message.get("type") == "aspac":
                    experimental.set_aspac(True)

                for ext in extensions:
                    if ext["enabled"] and message.get("type") in ext["events"]:
                        registry = extension.load_extension(
                            ext["name"])
                        args = []
                        for arg_expr in registry["args"]:
                            args.append(
                                eval(arg_expr, globals(), locals()))
                        asyncio.create_task(
                            extension.execute_extension(registry, args))

        except WebSocketDisconnect:
            experimental.set_connected(False)
            connected = False
            log(
                f"WebSocket connection from {sender[0]}:{sender[1]} disconnected", "EVENT")
        except Exception as e:
            traceback.print_exc()
    log(
        f"Extension server started on {get_config()['extensions']['host']}:{get_config()['extensions']['port']}", "EVENT")

    @app.get("/ping")
    def ping():
        return {"status": "ok"}

    if get_config()['extensions']['autoChat']['enable'] and get_config()['extensions']['autoChat']['chatType'] == "ollama":
        status = experimental.check_ollama()
        if not status:
            log("Ollama wasn't configured properly, `autoChat` won't work correctly",
                "CRITICAL", save=False)

    experimental.initiate_swap()
    run(app, host=get_config()['extensions']['host'], port=get_config()[
        'extensions']['port'], log_level="critical")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        global current_path_task, connected
        experimental.set_connected(False)
        if current_path_task and not current_path_task.done():
            current_path_task.cancel()
            experimental.reset_keyboard()

    app.router.lifespan_context = lifespan


if __name__ == "__main__":
    print(
        f"florr-auto-afk-server v{constants.VERSION_INFO}({constants.VERSION_TYPE}.{constants.SUB_VERSION}) {constants.RELEASE_DATE}")
    start_extension_server()
