import requests
from tqdm import tqdm
from matplotlib import use
import cv2
import pyautogui
from psutil import Process, virtual_memory, cpu_freq
from json import load, dump
from sys import _getframe, getwindowsversion
from os import path, mkdir, system, remove, listdir
import numpy as np
from datetime import datetime
from time import sleep, time, localtime
from rich.console import Console
from scipy.spatial import distance
from re import match
from rdp import rdp
from traceback import print_exc
import constants
import torch
import json
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize as skimage_skeletonize
from capture import wgc, bitblt
from ultralytics import YOLO
from tarfile import open as tar_open
from github import Github
import uuid

console = Console()

use('Agg')


def get_config() -> dict:
    with open("./config.json", "r", encoding="utf-8") as f:
        return load(f)


def debugger(msg, *args) -> None:
    if get_config()["advanced"]["debug"]:
        frame = _getframe().f_back
        now = datetime.now()
        time = now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        arg_names = {id(value): name for name, value in frame.f_locals.items()}
        mappings = {}
        for arg in args:
            mappings[arg_names.get(id(arg), "CONSTANT")] = str(arg)
        if mappings == {}:
            message = f"[{time}] {msg}"
        else:
            message = f"[{time}] {msg} {mappings}"
        with open('debug.log', 'a', encoding='utf-8') as f:
            f.write(f'{message}\n')


def log_ret(event: str, type: str, logger_=[], show: bool = True, save: bool = True):
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
        tag = "info"
    elif type.lower() == "error":
        style = "red"
        tag = "error"
    elif type.lower() == "warning":
        style = "yellow"
        tag = "warning"
    elif type.lower() == "critical":
        style = "bold red"
        tag = "critical"
    elif type.lower() == "event":
        style = "#ffab70"
        tag = "event"
    elif type.lower() == "notice":
        style = "#0969da"
        tag = "notice"
    else:
        style = ""
        tag = "default"

    if show:
        console.print(logger, style=style)
    if save:
        with open('latest.log', 'a', encoding='utf-8') as f:
            f.write(f'{logger}\n')
    logger_.append({'logger': logger, 'type': tag})
    return {'logger': logger, 'type': tag}


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
    flag = False
    if not path.exists("./models"):
        mkdir("./models")
        flag = True
    if not path.exists("./images"):
        mkdir("./images")
        flag = True
    if not path.exists("./latest.log"):
        with open("latest.log", "w") as f:
            f.write("")
        flag = True
    if not path.exists("./imgs"):
        mkdir("./imgs")
        flag = True
    if flag:
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
    repo = constants.ASSET_REPO
    url = f"https://api.github.com/repos/{repo}/releases/latest"
    response = requests.get(url)
    if response.status_code != 200:
        log("Trying GitHub mirror API", "WARNING")
        url = f"https://gh.llkk.cc/https://api.github.com/repos/{repo}/releases/latest"
        response = requests.get(url)
        if response.status_code != 200:
            log("Failed to get latest release info", "ERROR")
            return
    release_date = response.json()["published_at"]
    if not path.exists("./models/version"):
        with open("./models/version", "w") as f:
            f.write("")
    with open("./models/version", "r") as f:
        current_date = f.read()
    if current_date == release_date:
        log("Models are up to date", "INFO", save=False)
        return
    else:
        log("Updating models", "INFO")
    assets = response.json()["assets"]
    assets = [asset for asset in assets if asset["name"]
              in ["afk-seg.pt", "afk-det.pt"]]
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


def check_update():
    def parse_version(version):
        version = version.removeprefix("v")
        version_parts = version.split('.')
        return tuple(map(int, version_parts))

    def compare_versions(version1, version2):
        return version1 < version2

    repo = constants.PROJECT_REPO
    url = f"https://api.github.com/repos/{repo}/releases/latest"
    response = requests.get(url)
    if response.status_code != 200:
        log("Trying GitHub mirror API", "WARNING")
        url = f"https://gh.llkk.cc/https://api.github.com/repos/{repo}/releases/latest"
        response = requests.get(url)
        if response.status_code != 200:
            log("Failed to get latest release info", "ERROR")
            return
    remote_version = parse_version(response.json()["tag_name"])
    local_version = parse_version(constants.VERSION_INFO)
    if compare_versions(local_version, remote_version):
        log(f"New version available", "WARNING", save=False)
        log(
            f"You are at v{constants.VERSION_INFO} with remote latest {response.json()['tag_name']}", "WARNING", save=False)
    else:
        log("You are at the latest version", "INFO", save=False)
    return compare_versions(local_version, remote_version), remote_version


class AFK_Path:
    def __init__(self, raw_points: list[tuple[int, int]], start_p=None, end_p=None) -> None:
        self.raw_points: list[tuple[int, int]] = raw_points
        self.start_p: tuple[int, int] = start_p
        self.end_p: tuple[int, int] = end_p
        self.rdp_ed: bool = False
        self.rdp_points: list[tuple[int, int]] = None
        self.extend_point: tuple[int, int] = None
        self.sorted: bool = False
        self.sorted_points: list[tuple[int, int]] = None
        self.length: float = None

    def sort(self, width=20) -> None:
        if width is None or width <= 0:
            width = 20
        if self.end_p is not None:
            unused = set(self.raw_points)
            sorted_points = []
            current_point = self.end_p
            while unused:
                next_point = min(
                    unused, key=lambda p: distance.euclidean(current_point, p))
                sorted_points.append(next_point)
                unused.remove(next_point)
                current_point = next_point
            if self.start_p is not None:
                truncated_points = []
                dist = last_dist = float("inf")
                for point in sorted_points:
                    dist = distance.euclidean(point, self.start_p)
                    if not ((dist > last_dist) and (dist < width/2)):
                        truncated_points.append(point)
                        last_dist = dist
                    else:
                        truncated_points.append(self.start_p)
                        break
            else:
                truncated_points = sorted_points
            self.sorted_points = truncated_points[::-1]
            self.sorted_points.append(self.end_p)
            self.sorted = True
        else:
            if self.start_p is not None:
                unused = set(self.raw_points)
                sorted_points = []
                current_point = self.start_p
                while unused:
                    next_point = min(
                        unused, key=lambda p: distance.euclidean(current_point, p))
                    sorted_points.append(next_point)
                    unused.remove(next_point)
                    current_point = next_point
                truncated_points = sorted_points
                self.sorted = True
            else:
                self.sorted_points = self.raw_points
                self.sorted = False

    def rdp(self, epsilon=1) -> None:
        if self.sorted:
            if not self.rdp_points:
                self.rdp_points = rdp(
                    self.sorted_points, epsilon=epsilon)
                self.rdp_ed = True
        else:
            raise ValueError(
                "Path is not sorted. Please sort the path before applying RDP.")

    def extend(self, length) -> None:
        if self.rdp_ed:
            end = self.rdp_points[-1]
            last = self.rdp_points[-2]
            l_l2_dist = distance.euclidean(end, last)
            sine_theta = (end[1] - last[1]) / l_l2_dist
            cosine_theta = (end[0] - last[0]) / l_l2_dist
            self.extend_point = (
                end[0] + length * cosine_theta, end[1] + length * sine_theta)
        elif self.sorted:
            end = self.sorted_points[-1]
            last = self.sorted_points[-2]
            l_l2_dist = distance.euclidean(end, last)
            sine_theta = (end[1] - last[1]) / l_l2_dist
            cosine_theta = (end[0] - last[0]) / l_l2_dist
            self.extend_point = (
                end[0] + length * cosine_theta, end[1] + length * sine_theta)
        else:
            raise ValueError(
                "Path is not sorted. Please sort the path before extending.")

    def get_final(self, top_left_bound: tuple[int, int] = (0, 0), precise=True) -> list[tuple[int, int]]:
        if self.rdp_ed:
            if self.extend_point is not None:
                self.rdp_points.append(self.extend_point)
            ret = [(p[0] + top_left_bound[0], p[1] + top_left_bound[1])
                   for p in self.rdp_points]
        elif self.sorted:
            if self.extend_point is not None:
                self.sorted_points.append(self.extend_point)
            ret = [(p[0] + top_left_bound[0], p[1] + top_left_bound[1])
                   for p in self.sorted_points]
        else:
            ret = []
        if precise:
            return ret
        else:
            return [(int(p[0]), int(p[1])) for p in ret]

    def get_length(self) -> float:
        if self.length is not None:
            return self.length
        if self.rdp_ed:
            length = 0
            for i in range(len(self.rdp_points)-1):
                length += distance.euclidean(
                    self.rdp_points[i], self.rdp_points[i+1])
            self.length = length
            return length
        elif self.sorted:
            length = 0
            for i in range(len(self.sorted_points)-1):
                length += distance.euclidean(
                    self.sorted_points[i], self.sorted_points[i+1])
            self.length = length
            return length
        else:
            raise ValueError(
                "Path is not sorted. Please sort the path before getting length.")


class AFK_Segment:
    def __init__(self, afk_window_image: cv2.Mat, mask: torch.Tensor, start: tuple[int, int], end: tuple[int, int], start_size: int) -> None:
        self.image = afk_window_image
        self.mask = mask
        self.start = start
        self.end = end
        self.width = None
        self.start_size = start_size
        self.segmented_path = None
        self.start_color = tuple(int(i) for i in tuple(
            afk_window_image[int(start[1]), int(start[0])]))
        self.inverse_start_color = tuple(int(i) for i in tuple(
            255 - afk_window_image[int(start[1]), int(start[0])]))

    def save_start(self) -> None:
        if self.mask.ndim == 2:
            H, W = self.mask.shape
        elif self.mask.ndim == 3:
            C, H, W = self.mask.shape
        else:
            pass
        device = self.mask.device
        dtype = self.mask.dtype
        radius = self.start_size / 2.0
        radius_sq = radius ** 2
        cx, cy = int(self.start[0]), int(self.start[1])
        y_coords, x_coords = torch.meshgrid(torch.arange(H, device=device),
                                            torch.arange(W, device=device),
                                            indexing='ij')
        dist_sq = (x_coords.float() - cx)**2 + (y_coords.float() - cy)**2
        circle_mask_bool = dist_sq <= radius_sq
        circle_mask = circle_mask_bool.to(dtype)
        if self.mask.ndim == 3:
            circle_mask = circle_mask.unsqueeze(0)
        self.mask = torch.maximum(self.mask, circle_mask)

    def segment_path(self) -> list[tuple[int, int]]:
        mask = self.mask.cpu().numpy()
        mask = (mask * 255).astype(np.uint8)
        skeleton = cv2.ximgproc.thinning(mask)
        contours, _ = cv2.findContours(
            skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = contours[0]
        line = [(point[0][0], point[0][1]) for point in contour]
        line = list(set(line))
        self.segmented_path = line
        return line

    def get_width(self):
        if self.width is None:
            mask_np = self.mask.cpu().numpy()
            mask_bin = (mask_np > 0).astype(np.uint8)
            skeleton = skimage_skeletonize(mask_bin).astype(np.uint8)
            dist_map = distance_transform_edt(mask_bin)
            thickness_list = dist_map[skeleton > 0] * 2
            thickness_list = dist_map[skeleton > 0] * 2
            self.width = np.mean(thickness_list)
        return self.width


def apply_mouse_movement(points, speed=get_config()["advanced"]["mouseSpeed"], active=False):
    pyautogui.moveTo(points[0][0], points[0][1], duration=0.5)
    pyautogui.mouseUp(button="left")
    pyautogui.mouseUp(button="right")
    if active:
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


def test_environment(afk_seg_model):
    if get_config()["advanced"]["environment"]:
        return
    initiate()
    log("Testing environment", "INFO")
    image = cv2.imread("./imgs/test.png")
    results = {}
    log("Testing YOLO", "INFO")
    result = afk_seg_model.predict(image, retina_masks=True, verbose=False)
    masks = result[0].masks
    if masks is not None:
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
    configs["advanced"]['environment'] = True
    with open('config.json', 'w', encoding='utf-8') as f:
        dump(configs, f, ensure_ascii=False, indent=4)


def save_image(image, sub_type, type):
    if get_config()["advanced"]["saveImage"]:
        if not path.exists("./images"):
            mkdir("./images")
        if not path.exists(f"./images/{type}"):
            mkdir(f"./images/{type}")
        cv2.imwrite(
            f"./images/{type}/{sub_type}_{datetime.strftime(datetime.now(), '%Y-%m-%dT%H_%M_%SZ')}.png", image)


def save_test_image(image, sub_type, test_time: str):
    if not path.exists("./test"):
        mkdir("./test")
    if not path.exists(f"./test/{test_time}"):
        mkdir(f"./test/{test_time}")
    cv2.imwrite(
        f"./test/{test_time}/{sub_type}_{datetime.strftime(datetime.now(), '%Y-%m-%dT%H_%M_%SZ')}.png", image)


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


def detect_afk(img, afk_det_model, caller="main", test_time=None):
    things = yolo_detect(afk_det_model, img)
    windows_pos = None
    for thing in things[0]:
        if thing['name'] == 'Window':
            windows_pos = ((thing['x_1'], thing['y_1']),
                           (thing['x_2'], thing['y_2']))
    if windows_pos is None:
        return None
    window_width = windows_pos[1][0] - windows_pos[0][0]
    window_height = windows_pos[1][1] - windows_pos[0][1]
    ratio = window_width / window_height
    if (get_config()["advanced"]["windowSizeRatio"][0]*(1-get_config()["advanced"]["windowSizeTolerance"]) < ratio < get_config()["advanced"]["windowSizeRatio"][0]*(1+get_config()["advanced"]["windowSizeTolerance"])) or (get_config()["advanced"]["windowSizeRatio"][1]*(1-get_config()["advanced"]["windowSizeTolerance"]) < ratio < get_config()["advanced"]["windowSizeRatio"][1]*(1+get_config()["advanced"]["windowSizeTolerance"])):
        afk_window_img = img.copy()[windows_pos[0][1]:windows_pos[1][1],
                                    windows_pos[0][0]:windows_pos[1][0]]
        things_afk = yolo_detect(afk_det_model, afk_window_img)
        if get_config()["advanced"]["saveYOLOImage"]:
            for obj in things_afk[0]:
                cv2.rectangle(afk_window_img, (obj['x_1'], obj['y_1']),
                              (obj['x_2'], obj['y_2']), color=(0, 255, 0), thickness=2)
                cv2.putText(afk_window_img, obj['name'], (obj['x_1'], obj['y_1'] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                if caller == "main":
                    save_image(afk_window_img, "det", "yolo")
                elif caller == "test":
                    save_test_image(afk_window_img, "det", test_time)
        start_pos = end_pos = None
        start_max_confidence = end_max_confidence = 0
        start_size = 0
        for thing in things_afk[0]:
            if thing['name'] == 'Start' and thing['confidence'] > start_max_confidence:
                start_pos = (thing['x_avg'], thing['y_avg'])
                start_max_confidence = thing['confidence']
                start_size = (abs(thing['x_2'] - thing['x_1']) +
                              abs(thing['y_2'] - thing['y_1'])) / 2
            if thing['name'] == 'End' and thing['confidence'] > end_max_confidence:
                end_pos = (thing['x_avg'], thing['y_avg'])
                end_max_confidence = thing['confidence']
        return start_pos, end_pos, windows_pos, start_size
    return None


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


def exposure_image(left_top_bound, right_bottom_bound, duration, hwnd=None, capture_method=None):
    debugger("Exposure image", left_top_bound,
             right_bottom_bound, duration, hwnd, capture_method)
    start_time = time()
    frames = []
    region = (left_top_bound[0], left_top_bound[1],
              right_bottom_bound[0]-left_top_bound[0], right_bottom_bound[1]-left_top_bound[1])
    while time() - start_time < duration:
        if hwnd is None:
            screenshot = pyautogui.screenshot(region=region)
            screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        else:
            if capture_method == "Windows Graphics Capture":
                screenshot = wgc.wgc_capture(hwnd)
            elif capture_method == "BitBlt":
                screenshot = bitblt.bitblt_capture(hwnd)
            screenshot = crop_image(
                left_top_bound, right_bottom_bound, screenshot)
            screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGBA2RGB)
        frames.append(screenshot)
    average_frame = np.mean(frames, axis=0).astype(np.uint8)
    return average_frame


def calculate_offset(path, hwnd):
    rect = wgc.get_fixed_window_rect(hwnd)
    return [(p[0] + rect[0], p[1] + rect[1]) for p in path]


def get_masks_by_iou(image, afk_seg_model: YOLO, lower_iou=0.3, upper_iou=0.7, stepping=0.1):
    debugger("Getting masks by iou", lower_iou, upper_iou, stepping)
    for iou in np.arange(upper_iou, lower_iou, -stepping):
        results = afk_seg_model.predict(
            image, retina_masks=True, verbose=False, iou=iou)
        if results[0].masks is None:
            return None
        debugger("Found masks", iou, len(results[0].masks.data))
        if len(results[0].masks.data) == 1:
            break
    if len(results[0].masks.data) != 1:
        results.sort(key=lambda x: x.boxes.conf[0], reverse=True)
    mask = results[0].masks.data[0]
    return mask, results


def export_result_to_dataset(results, image, epsilon=1):
    if results[0].masks is None:
        return
    now = int(time())
    image_height, image_width, _ = image.shape
    debugger("Exporting result to dataset", now, image_height, image_width)
    labelme_data = {
        "version": "5.6.0",
        "flags": {},
        "shapes": [],
        "imagePath": f"../images/{now}.png",
        "imageData": None,
        "imageHeight": image_height,
        "imageWidth": image_width
    }
    if results and results[0].masks:
        for i in range(len(results[0].masks)):
            original_polygon_points = results[0].masks.xy[i]
            simplified_polygon_np = rdp(
                original_polygon_points, epsilon=epsilon)
            simplified_polygon_points = simplified_polygon_np.tolist()
            shape_entry = {
                "label": "path",
                "points": simplified_polygon_points,
                "group_id": None,
                "description": "",
                "shape_type": "polygon",
                "flags": {},
                "mask": None
            }
            labelme_data["shapes"].append(shape_entry)
    if not path.exists("./train"):
        mkdir("./train")
    if not path.exists("./train/images"):
        mkdir("./train/images")
    if not path.exists("./train/split"):
        mkdir("./train/split")

    cv2.imwrite(f"./train/images/{now}.png", image)
    with open(f"./train/split/{now}.json", "w") as f:
        dump(labelme_data, f, indent=4)
    with tar_open("./train/train.tar.gz", "w:gz") as tar:
        tar.add("./train/images", arcname="images")
        tar.add("./train/split", arcname="split")
    return labelme_data


def check_config(shared_logger):
    config = get_config()
    if config["runs"]["autoTakeOverWhenIdle"]:
        log_ret("Idle Detection is currently enabled, set `autoTakeOverWhenIdle` to `false` if you just want to test the AFK Bypass ability",
                "WARNING", shared_logger, save=False)
    if config["runs"]["idleDetInterval"] > config["runs"]["idleDetIntervalMax"]:
        log_ret("Idle Detection Interval is greater than the maximum value, set `idleDetInterval` to a value less than `idleDetIntervalMax`",
                "WARNING", shared_logger, save=False)
    if config["exposure"]["duration"] > 10:
        log_ret("Too long exposure duration",
                "WARNING", shared_logger, save=False)
    if config["exposure"]["moveInterval"]*4 > config["exposure"]["duration"]:
        log_ret("`moveInterval` is grater than exposure `duration`, it may cause unnecessary move lead to death",
                "WARNING", shared_logger, save=False)
    if not config["advanced"]["saveTrainData"]:
        log_ret("If you want to improve the AFK model and contribute to the project, suggest turning on `Save Trainable Dataset` in Settings > Advanced",
                "Notice", shared_logger, save=False)
    if pyautogui.size().height < 1080:
        log_ret("You have a low screen resolution, it may cause detection failure",
                "WARNING", shared_logger, save=False)
    memory_gb = virtual_memory().total / (1024 ** 3)
    cpu_freq_ghz = cpu_freq().current / 1000
    if memory_gb < 6:
        log_ret(f"Running on {memory_gb:.2f}GB memory, OutOfMemory Warning",
                "WARNING", shared_logger, save=False)
    if cpu_freq_ghz < 2.5:
        log_ret(f"Running on {cpu_freq_ghz:.2f}GHz CPU, slow exection Warning",
                "WARNING", shared_logger, save=False)


def draw_annotated_image(ori_image, line, start_p, end_p, window_pos, start_color, path_width, path_length) -> cv2.Mat:
    image = ori_image.copy()
    for point in line:
        cv2.circle(image, point, 3, (255, 0, 0), -1)
    for i in range(len(line) - 1):
        cv2.arrowedLine(
            image,
            line[i],
            line[i + 1],
            (255, 255, 0),
            2,
            tipLength=0.1
        )
    if start_p is not None:
        cv2.circle(image, (int(
            start_p[0]+window_pos[0][0]), int(start_p[1]+window_pos[0][1])), 5, start_color, -1)
    if end_p is not None:
        cv2.circle(image, (int(
            end_p[0]+window_pos[0][0]), int(end_p[1]+window_pos[0][1])), 5, (0, 0, 255), -1)

    p1 = window_pos[0]
    p2 = window_pos[1]
    corner_len = 20

    cv2.line(image, p1, (p1[0]+corner_len, p1[1]), (0, 0, 255), 2)
    cv2.line(image, p1, (p1[0], p1[1]+corner_len), (0, 0, 255), 2)
    cv2.line(image, (p2[0], p1[1]), (p2[0]-corner_len, p1[1]), (0, 0, 255), 2)
    cv2.line(image, (p2[0], p1[1]), (p2[0], p1[1]+corner_len), (0, 0, 255), 2)
    cv2.line(image, (p1[0], p2[1]), (p1[0]+corner_len, p2[1]), (0, 0, 255), 2)
    cv2.line(image, (p1[0], p2[1]), (p1[0], p2[1]-corner_len), (0, 0, 255), 2)
    cv2.line(image, p2, (p2[0]-corner_len, p2[1]), (0, 0, 255), 2)
    cv2.line(image, p2, (p2[0], p2[1]-corner_len), (0, 0, 255), 2)

    cv2.putText(image, f'P_w: {round(path_width*100)/100}, P_l: {round(path_length*100)/100}', (window_pos[0][0]+10, window_pos[1][1]-10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255))
    return image


def gh_upload_file(repo, target_path, content, message):
    try:
        contents = repo.get_contents(target_path)
        repo.update_file(
            target_path,
            message,
            content,
            contents.sha
        )
    except Exception:
        repo.create_file(
            target_path,
            message,
            content
        )


def gh_upload_dataset(image, label):
    g = Github(constants.GITHUB_TOKEN)
    repo = g.get_repo(constants.DATASET_REPO)
    alias = str(uuid.uuid4()).replace("-", "")
    date = datetime.now().strftime("%Y-%m-%dT%H_%M_%SZ")
    _, img_encoded = cv2.imencode('.png', image)
    img_bytes = img_encoded.tobytes()
    label_bytes = json.dumps(label, ensure_ascii=False,
                             indent=2).encode('utf-8')
    gh_upload_file(repo, f"./{alias}/{alias}.png",
                   img_bytes, f"{date} {alias[:5]} Image")
    gh_upload_file(repo, f"./{alias}/{alias}.json",
                   label_bytes, f"{date} {alias[:5]} Label")
    log(f"Uploaded dataset to GitHub with hash {alias}", "INFO")
    return alias
