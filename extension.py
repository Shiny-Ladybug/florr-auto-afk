import json
from lupa import LuaRuntime
import traceback
lua = LuaRuntime(unpack_returned_tuples=True)


def load_extension(name):
    with open(f"./extensions/{name}/registry.json", "r") as f:
        registry = json.load(f)

    with open(f"./extensions/{name}/main.lua", "r") as f:
        code = f.read()
    return registry, code


def execute_extension(registry, code, args):
    try:
        lua.execute(code)
        imports = registry.get("imports", [])
        if isinstance(imports, str):
            imports = [imports]
        for module_name in imports:
            module = __import__(module_name)
            lua.globals()[module_name] = module

        return lua.globals().main(*args)
    except Exception as e:
        traceback.print_exc()
        return None
