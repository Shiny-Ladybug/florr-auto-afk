from json import load, dump
from os.path import exists


def switch_block_alpha(type: bool):
    with open('./extension.swap.json', 'r', encoding="utf-8") as file:
        data = load(file)
        data['block_alpha'] = type

    with open('./extension.swap.json', 'w', encoding="utf-8") as file:
        dump(data, file, indent=4)


def get_block_alpha():
    with open('./extension.swap.json', 'r', encoding="utf-8") as file:
        data = load(file)
        return data['block_alpha']


def switch_send(type: bool):
    with open('./extension.swap.json', 'r', encoding="utf-8") as file:
        data = load(file)
        data['send'] = type

    with open('./extension.swap.json', 'w', encoding="utf-8") as file:
        dump(data, file, indent=4)


def get_send():
    with open('./extension.swap.json', 'r', encoding="utf-8") as file:
        data = load(file)
        return data['send']


default = {
    "block_alpha": False,
    "send": False
}

if not exists('./extension.swap.json'):
    with open('./extension.swap.json', 'w', encoding="utf-8") as file:
        dump(default, file, indent=4)
