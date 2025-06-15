from json import load, dump
from os.path import exists
from time import time


def get_config() -> dict:
    with open("./config.json", "r", encoding="utf-8") as f:
        return load(f)


last_request = 0
last_cache = None


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


default = {
    "block_alpha": False,
    "send": False
}

if not exists('./extension.swap.json'):
    with open('./extension.swap.json', 'w', encoding="utf-8") as file:
        dump(default, file, indent=4)
