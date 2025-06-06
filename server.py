from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from sys import _getframe
import asyncio
import experimental
from datetime import datetime
from uvicorn import run
from os import path
from rich.console import Console

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


def get_config() -> dict:
    with open("./config.json", "r", encoding="utf-8") as f:
        return experimental.load(f)


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
        try:
            while True:
                await asyncio.sleep(2)
                if experimental.get_block_alpha() != should_block_alpha:
                    log("Block alpha setting changed", "EVENT", save=False)
                    should_block_alpha = experimental.get_block_alpha()
                    await websocket.send_json({"block_alpha": should_block_alpha})
                if experimental.get_send() == True:
                    experimental.switch_send(False)
                    await websocket.send_json({"command": "send"})
                    message = await websocket.receive_json()
                    log(f"Received message: {message}", "EVENT")
        except Exception:
            pass

    log(
        f"Extension server started on {get_config()['extensions']['host']}:{get_config()['extensions']['port']}", "EVENT")
    run(app, host=get_config()['extensions']['host'], port=get_config()[
        'extensions']['port'], log_level="error")


if __name__ == "__main__":
    start_extension_server()
