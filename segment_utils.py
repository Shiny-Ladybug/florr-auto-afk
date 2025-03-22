from constants import *
import requests
from tqdm import tqdm
from matplotlib import use
from ultralytics import YOLO
import cv2
import base64
import pyautogui
from json import load, dump
from sys import _getframe
from os import path, mkdir, system
import numpy as np
from datetime import datetime
from time import sleep, time, localtime
from rich.console import Console
from scipy.interpolate import interp1d
from scipy.spatial import distance
import multiprocessing
from re import match
from rdp import rdp
from traceback import print_exc

multiprocessing.freeze_support()

console = Console()

use('Agg')


print(f"florr-auto-afk v{VERSION_INFO}({VERSION_TYPE}) {RELEASE_DATE}")


def get_config():
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


def initiate():
    if not path.exists("./models"):
        mkdir("./models")
    if not path.exists("./images"):
        mkdir("./images")
    if not path.exists("./latest.log"):
        with open("latest.log", "w") as f:
            f.write("")
    if not path.exists("./config.json"):
        with open("config.json", "w") as f:
            dump(DEFAULT_CONFIG, f, ensure_ascii=False, indent=4)
    if not path.exists("./imgs"):
        with open('./imgs/test.png', 'wb') as file:
            img = base64.b64decode(TEST_IMAGE)
            file.write(img)
    log("Initiated", "INFO")


def download_file(url, filename):
    response = requests.get(url, stream=True, timeout=2)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    t = tqdm(total=total_size, unit='iB', unit_scale=True, desc=filename)
    with open(filename, "wb") as f:
        for data in response.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()


def update_models():
    repo = "Shiny-Ladybug/florr-auto-afk"
    tag_name = "models"
    url = f"https://api.github.com/repos/{repo}/releases/tags/{tag_name}"
    response = requests.get(url)
    if response.status_code != 200:
        log("Trying GitHub mirror API", "WARNING")
        url = f"https://gh.llkk.cc/https://api.github.com/repos/{repo}/releases/tags/{tag_name}"
        response = requests.get(url)
        if response.status_code != 200:
            log("Failed to get release info", "ERROR")
            return
    release_date = response.json()["published_at"]
    if not path.exists("./models/version"):
        with open("./models/version", "w") as f:
            f.write("")
    with open("./models/version", "r") as f:
        current_date = f.read()
    if current_date == release_date:
        log("Models are up to date", "INFO")
        return
    else:
        log("Updating models", "INFO")
    assets = response.json()["assets"]
    for asset in assets:
        try:
            download_file(asset["browser_download_url"],
                          "./models/"+asset["name"])
        except:
            download_file("https://gh.llkk.cc/" +
                          asset["browser_download_url"], "./models/"+asset["name"])
    with open("./models/version", "w") as f:
        f.write(release_date)
    log("Models updated", "INFO")


initiate()
if get_config()["skipUpdate"] == False:
    try:
        update_models()
    except:
        print_exc()
        log("Cannot update models", "WARNING")
afk_seg_model = YOLO(get_config()["yoloConfig"]["segModel"])
afk_det_model = YOLO(get_config()["yoloConfig"]["detModel"])


def color(count, index):
    percent = index / count
    r = int(255 * (1 - percent))
    g = int(255 * percent)
    b = 0
    return (r, g, b)


def reorder_points_by_distance(start, points):
    if start != None:
        sorted_points = [start]
    else:
        sorted_points = [points[0]]
    while points:
        last_point = sorted_points[-1]
        min_distance = float('inf')
        next_point = None
        for point in points:
            try:
                dist = distance.euclidean(last_point, point)
            except:
                print(last_point, point)
            if dist < min_distance:
                min_distance = dist
                next_point = point
        sorted_points.append(next_point)
        points.remove(next_point)
    return sorted_points


def truncate_points(start, points, end):
    sorted_points = []
    if end != None:
        sorted_points = [end]
    else:
        sorted_points = [points[-1]]
    while points:
        last_point = sorted_points[-1]
        min_distance = float('inf')
        for point in points:
            dist = distance.euclidean(last_point, point)
            if dist < min_distance:
                min_distance = dist
                next_point = point
        sorted_points.append(next_point)
        if distance.euclidean(next_point, start) < 2:
            sorted_points.append(start)
            break
        points.remove(next_point)
    sorted_points = sorted_points[::-1]
    return sorted_points


def extend_line(line):
    end = line[-1]
    last = line[-2]
    l_l2_dist = distance.euclidean(end, last)
    sine_theta = (end[1] - last[1]) / l_l2_dist
    cosine_theta = (end[0] - last[0]) / l_l2_dist
    delta_x = get_config()["extendLength"] * cosine_theta
    delta_y = get_config()["extendLength"] * sine_theta
    end = (round(end[0] + delta_x), round(end[1] + delta_y))
    return line + [end]


def apply_mouse_movement(points, speed=get_config()["mouseSpeed"]):
    pyautogui.moveTo(points[0][0], points[0][1], duration=0.5)
    pyautogui.mouseUp(button="left")
    pyautogui.mouseUp(button="right")
    pyautogui.doubleClick()
    pyautogui.mouseDown(button="left")
    for i in range(1, len(points), 1):
        distance_ = distance.euclidean(points[i], points[i-1])
        if distance_ < 5:
            pyautogui.moveTo(points[i][0], points[i][1])
        else:
            duration = distance_ / speed
            pyautogui.moveTo(points[i][0], points[i][1], duration=duration)
    pyautogui.mouseUp(button="left")


def obs(type):
    if type == "start":
        pyautogui.hotkey("ctrl", "alt", "shift", "o", interval=0.1)
    elif type == "stop":
        pyautogui.hotkey("ctrl", "alt", "shift", "p", interval=0.1)


def crop_image(left_top_bound, right_bottom_bound, image):
    return image[left_top_bound[1]:right_bottom_bound[1], left_top_bound[0]:right_bottom_bound[0]]


def validate_line(line):
    new = []
    for pos in line:
        try:
            new.append((int(pos[0]), int(pos[1])))
        except:
            pass
    return new


def test_environment():
    if get_config()["environment"]:
        return
    initiate()
    log("Testing environment", "INFO")
    image = cv2.imread("./imgs/test.png")
    results = {}
    log("Testing YOLO", "INFO")
    result = afk_seg_model.predict(image, retina_masks=True, verbose=False)
    masks = result[0].masks
    if masks:
        log(f"YOLO passed", "INFO")
        results['yolo'] = True
    else:
        log("YOLO failed", "ERROR")
        results['yolo'] = False
    log("Testing PyAutoGUI", "INFO")
    log(f"PyAutoGUI screen size: {pyautogui.size()}", "INFO")
    log(f"Mouse position: {pyautogui.position()}", "INFO")
    log("Mouse will move to (100, 100) in 1 second", "INFO")
    pyautogui.moveTo(100, 100, duration=1)
    if pyautogui.position() == (100, 100):
        log("PyAutoGUI passed", "INFO")
        results['pyautogui'] = True
    else:
        log("PyAutoGUI failed", "ERROR")
        results['pyautogui'] = False
    if all(results.values()):
        log("Environment passed", "INFO")
    else:
        log("Environment failed", "WARNING")
    log(f"Environment test finished with {results}", "INFO")
    configs = get_config()
    configs['environment'] = True
    with open('config.json', 'w', encoding='utf-8') as f:
        dump(configs, f, ensure_ascii=False, indent=4)


def save_image(image, sub_type, type):
    if not path.exists("./images"):
        mkdir("./images")
    if not path.exists(f"./images/{type}"):
        mkdir(f"./images/{type}")
    cv2.imwrite(
        f"./images/{type}/{sub_type}_{datetime.strftime(datetime.now(), '%Y-%m-%dT%H_%M_%SZ')}.png", image)


def yolo_detect(model, img):
    result = model.predict(img, verbose=False)[0]
    names = result.names
    boxes = result.boxes
    things = boxes.data.tolist()
    detected = []
    for method in things:
        new_method = []
        for i in method:
            new_method.append(round(i))
        x_1 = new_method[0]
        y_1 = new_method[1]
        x_2 = new_method[2]
        y_2 = new_method[3]
        x_avg = (x_1 + x_2) / 2
        y_avg = (y_1 + y_2) / 2
        name = new_method[5]
        confidence = new_method[4]
        object = names[name]
        detected.append({"name": object, "x_1": x_1,
                        "y_1": y_1, "x_2": x_2, "y_2": y_2, "x_avg": x_avg, "y_avg": y_avg, "confidence": confidence})
    return detected, img


def detect_afk(img):
    things = yolo_detect(afk_det_model, img)
    windows_pos = None
    for thing in things[0]:
        if thing['name'] == 'Window':
            windows_pos = ((thing['x_1'], thing['y_1']),
                           (thing['x_2'], thing['y_2']))
    if windows_pos == None:
        return None
    window_width = windows_pos[1][0] - windows_pos[0][0]
    window_height = windows_pos[1][1] - windows_pos[0][1]
    ratio = window_width / window_height
    if (get_config()["windowSizeRatio"][0]*(1-get_config()["windowSizeTolerance"]) < ratio < get_config()["windowSizeRatio"][0]*(1+get_config()["windowSizeTolerance"])) or (get_config()["windowSizeRatio"][1]*(1-get_config()["windowSizeTolerance"]) < ratio < get_config()["windowSizeRatio"][1]*(1+get_config()["windowSizeTolerance"])):
        afk_window_img = img[windows_pos[0][1]:windows_pos[1][1],
                             windows_pos[0][0]:windows_pos[1][0]]
        things_afk = yolo_detect(afk_det_model, afk_window_img)
        start_pos = end_pos = None
        start_max_confidence = end_max_confidence = 0
        for thing in things_afk[0]:
            if thing['name'] == 'Start' and thing['confidence'] > start_max_confidence:
                start_pos = (thing['x_avg'], thing['y_avg'])
                start_max_confidence = thing['confidence']
            if thing['name'] == 'End' and thing['confidence'] > end_max_confidence:
                end_pos = (thing['x_avg'], thing['y_avg'])
                end_max_confidence = thing['confidence']
        return start_pos, end_pos, windows_pos
    return None


def calculate_degree(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi


def remove_duplicate_points(points):
    new_points = []
    for point in points:
        if point not in new_points:
            new_points.append(point)
    return new_points


def segment_path(masks, start, end, left_top_bound):
    for mask in masks.data:
        mask = mask.cpu().numpy()
        mask = (mask * 255).astype(np.uint8)
        skeleton = cv2.ximgproc.thinning(mask)
        contours, _ = cv2.findContours(
            skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[0]
    line = []
    for point in contour:
        line.append((point[0][0] + left_top_bound[0],
                    point[0][1] + left_top_bound[1]))
    line = remove_duplicate_points(line)
    if start != None and end != None:
        line = truncate_points(start, line, end)
    else:
        line = reorder_points_by_distance(start, line)
    return line


def move_a_bit(interval=0.1):
    pyautogui.keyDown("w")
    sleep(interval)
    pyautogui.keyUp("w")
    pyautogui.keyDown("d")
    sleep(interval)
    pyautogui.keyUp("d")
    pyautogui.keyDown("s")
    sleep(interval)
    pyautogui.keyUp("s")
    pyautogui.keyDown("a")
    sleep(interval)
    pyautogui.keyUp("a")
