import json
import traceback
import asyncio


def load_extension(name):
    with open(f"./extensions/{name}/registry.json", "r") as f:
        registry = json.load(f)
    return registry


async def execute_extension(registry, args):
    try:
        module = __import__(
            f"extensions.{registry['name']}.main", fromlist=[''])
        if hasattr(module, 'main'):
            ret = await module.main(*args)
            return ret
        else:
            raise AttributeError(
                f"The module '{registry['name']}' does not have a 'main' function.")
    except ImportError as e:
        print(
            f"Error importing module '{registry['name']}': {e}")
        return None
    except Exception as e:
        print(
            f"An error occurred while executing the extension '{registry['name']}': {e}")
        traceback.print_exc()
    return None
