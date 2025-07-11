import requests
from tqdm import tqdm
from matplotlib import use
import cv2
import pyautogui
from psutil import Process, virtual_memory, cpu_freq
from json import load, dump, dumps, loads
from sys import _getframe, getwindowsversion
from os import path, mkdir, remove, listdir, startfile
import numpy as np
from datetime import datetime
from time import sleep, time
from rich.console import Console
from scipy.spatial import distance, KDTree
from rdp import rdp
import constants
import torch
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize as skimage_skeletonize
from capture import gdi, bitblt, wgc
from ultralytics import YOLO
from tarfile import open as tar_open
from github import Github, GithubIntegration
from uuid import uuid4
from heapq import heappop, heappush
from random import randint, uniform, choice, shuffle
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
    elif type.lower() == "chat":
        style = "#d46183"
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
    response = request_github_api(f"repos/{repo}/releases/latest")
    if response is None:
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
        if not download_github_asset(asset["browser_download_url"],
                                     "./models/"+asset["name"]):
            log(f"Failed to download {asset['name']}", "ERROR", save=False)
            return
    with open("./models/version", "w") as f:
        f.write(release_date)
    log("Models updated", "INFO")


def request_github_api(endpoint):
    mirrors = [
        "https://api.github.com",
        "https://gh.llkk.cc/https://api.github.com",
        "https://j.1win.ggff.net/https://api.github.com",
        "https://j.1win.ip-ddns.com/https://api.github.com",
        "https://j.1win.ddns-ip.net/https://api.github.com",
        "https://ghfile.geekertao.top/https://api.github.com"
    ]
    for mirror in mirrors:
        url = f"{mirror}/{endpoint}"
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return response
            else:
                log(f"Failed to get {endpoint} from {mirror}, code: {response.status_code}",
                    "WARNING", save=False)
        except requests.RequestException as e:
            continue
    return None


def download_github_asset(url, filename):
    mirrors = [
        "",  # base
        "https://github.moeyy.xyz/",
        "https://ghproxy.net/",
        "https://ghfast.top/",
    ]
    for mirror in mirrors:
        full_url = f"{mirror}{url}"
        try:
            download_file(full_url, filename)
            return True
        except requests.RequestException as e:
            log(
                f"Failed to download {filename} from {full_url}, error: {e}", "WARNING", save=False)
            continue
    return None


def check_update():
    def parse_version(version):
        version = version.removeprefix("v")
        version_parts = version.split('.')
        return tuple(map(int, version_parts))

    def compare_versions(version1, version2):
        return version1 < version2

    repo = constants.PROJECT_REPO
    response = request_github_api(f"repos/{repo}/releases/latest")
    if response is None:
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
    def __init__(self, raw_points: list[tuple[int, int]], start_p=None, end_p=None, width=None) -> None:
        self.raw_points: list[tuple[int, int]] = raw_points
        self.start_p: tuple[int, int] = start_p
        self.end_p: tuple[int, int] = end_p
        self.rdp_ed: bool = False
        self.rdp_points: list[tuple[int, int]] = None
        self.extend_point: tuple[int, int] = None
        self.sorted: bool = False
        self.sorted_points: list[tuple[int, int]] = None
        self.length: float = None
        self.difficulty: float = None
        self.width: float = width
        self.sort_method = None

    def sort(self) -> None:
        if self.width is None or self.width <= 0:
            self.width = 20
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
                    if not ((dist > last_dist) and (dist < self.width/2)):
                        truncated_points.append(point)
                        last_dist = dist
                    else:
                        truncated_points.append(self.start_p)
                        break
            else:
                truncated_points = sorted_points
            self.sorted_points = truncated_points[::-1]
            self.sorted_points.append(self.end_p)
            self.sort_method = "skeleton"
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
                self.sorted_points = sorted_points
                self.sort_method = "skeleton"
                self.sorted = True
            else:
                unused = set(self.raw_points)
                sorted_points = []
                current_point = self.raw_points[0]
                while unused:
                    next_point = min(
                        unused, key=lambda p: distance.euclidean(current_point, p))
                    sorted_points.append(next_point)
                    unused.remove(next_point)
                    current_point = next_point
                self.sorted_points = sorted_points
                self.sort_method = "skeleton"
                self.sorted = True

    def dijkstra(self, mask: torch.Tensor) -> bool:
        if self.start_p is None or self.end_p is None:
            return False
        area_mask = mask.cpu().numpy().astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            area_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area_mask = np.zeros_like(area_mask)
        cv2.drawContours(area_mask, contours, -1, 255, -1)
        dt = cv2.distanceTransform(area_mask, cv2.DIST_L2, 5)
        h, w = dt.shape
        sx, sy = round(self.start_p[0]), round(self.start_p[1])
        ex, ey = round(self.end_p[0]), round(self.end_p[1])
        if area_mask[sy, sx] == 0 or area_mask[ey, ex] == 0:
            return False
        max_w = -np.ones((h, w), dtype=np.float32)
        prev = {}
        max_w[sy, sx] = dt[sy, sx]
        pq = [(-dt[sy, sx], (sx, sy))]
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        while pq:
            neg_dist, (x, y) = heappop(pq)
            curr = -neg_dist
            if (x, y) == (ex, ey):
                break
            if curr < max_w[y, x]:
                continue
            for dx, dy in dirs:
                nx, ny = x+dx, y+dy
                if not (0 <= nx < w and 0 <= ny < h):
                    continue
                if area_mask[ny, nx] == 0:
                    continue
                new_w = min(curr, dt[ny, nx])
                if new_w > max_w[ny, nx]:
                    max_w[ny, nx] = new_w
                    prev[(nx, ny)] = (x, y)
                    heappush(pq, (-new_w, (nx, ny)))
        path = []
        cur = (ex, ey)
        if cur in prev or cur == (sx, sy):
            while cur != (sx, sy):
                path.append(cur)
                cur = prev[cur]
            path.append((sx, sy))
            path.reverse()
        else:
            return False
        self.sorted_points = path
        self.sort_method = "dijkstra"
        self.sorted = True
        return True

    def rdp(self, epsilon=1) -> None:
        if self.sorted:
            if not self.rdp_points:
                self.rdp_points = rdp(
                    self.sorted_points, epsilon=epsilon)
                self.rdp_ed = True
        else:
            if not self.rdp_points:
                self.rdp_points = rdp(self.raw_points, epsilon=epsilon)
                self.rdp_ed = True

    def extend(self, length) -> None:
        if self.rdp_ed:
            if len(self.rdp_points) < 2:
                self.extend_point = self.rdp_points[0]
                return
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

    def get_difficulty(self, density_r=50) -> float:
        if self.difficulty is not None:
            return self.difficulty
        if self.rdp_points:
            points = np.array(self.rdp_points)
            tree = KDTree(points)
            neighbors = tree.query_ball_point(points, r=density_r)
            counts = np.array([len(n)-1 for n in neighbors])
            density = np.mean(counts)
            lwr = self.get_length()/self.width
            self.difficulty = (max(1, density*3)*lwr *
                               (len(self.rdp_points)**0.5))**0.5
            return self.difficulty


class AFK_Segment:
    def __init__(self, afk_window_image: cv2.Mat, mask: torch.Tensor, start: tuple[int, int], end: tuple[int, int], start_size: int) -> None:
        self.image = afk_window_image
        self.mask = mask
        self.start = start
        self.end = end
        self.width = None
        self.start_size = start_size
        self.segmented_path = None
        if start is not None:
            self.start_color = tuple(int(i) for i in tuple(
                afk_window_image[int(start[1]), int(start[0])]))
            self.inverse_start_color = tuple(int(i) for i in tuple(
                255 - afk_window_image[int(start[1]), int(start[0])]))
        else:
            self.start_color = (0, 0, 0)
            self.inverse_start_color = (255, 255, 255)

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


class AFK_BW:
    def __init__(self, image):
        self.offset = [0, 0]
        self.raw_image = image.copy()
        self.image = None
        self.start_p = None
        self.end_p = None
        self.length: float = None
        self.difficulty: float = None
        self.width: float = None
        self.extend_point: tuple[int, int] = None
        self.sorted_points: list[tuple[int, int]] = None
        self.rdp_points: list[tuple[int, int]] = None
        self.extend_point: tuple[int, int] = None
        self.white_labels = None
        self.max_white_label = None
        self.difficulty = None
        self.start_color = (0, 0, 0)
        self.inverse_start_color = (255, 255, 255)

    def crop_nb_image(self):
        gray = cv2.cvtColor(self.raw_image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)

            cropped = self.raw_image[y:y+h, x:x+w]
            self.offset[0] += x
            self.offset[1] += y
            self.raw_image = cropped
        ratio = self.raw_image.shape[1] / self.raw_image.shape[0]
        if (0.787*(1-0.1) < ratio < 0.787*(1+0.1)):
            h = int(self.raw_image.shape[0] * 0.205)
            self.raw_image = self.raw_image[h:, :]
            self.offset[1] += h
        return self

    def rarity_colors(self):
        def hex2bgr(hex_color):
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i + 2], 16) for i in (4, 2, 0))

        return list({
            "common": hex2bgr("#7EEF6D"),
            "unusual": hex2bgr("#FFE65D"),
            "rare": hex2bgr("#4d52e3"),
            "epic": hex2bgr("#861FDE"),
            "legendary": hex2bgr("#DE1F1F"),
            "mythic": hex2bgr("#1fdbde"),
            "ultra": hex2bgr("#ff2b75"),
            "super": hex2bgr("#2bffa3")
        }.values())

    def get_mask(self):
        white_mask = cv2.inRange(
            self.image, (255, 255, 255), (255, 255, 255))
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            white_mask, connectivity=8)
        if num_labels <= 1:
            return 0, None, None
        max_idx = 1 + np.argmax(stats[1:, 4])
        contour_img = None
        mask = (labels == max_idx).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_img = self.image.copy()
        cv2.drawContours(contour_img, contours, -1, (0, 0, 255), 2)
        self.white_labels = labels
        self.max_white_label = max_idx

    def normalize(self, threshold=40):
        image = self.raw_image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        color_diff = np.abs(image[:, :, 0] - image[:, :, 1]) + np.abs(
            image[:, :, 1] - image[:, :, 2]) + np.abs(image[:, :, 0] - image[:, :, 2])
        gray_mask = color_diff == 0
        output = np.ones_like(image) * 255
        output[gray_mask & (gray > threshold)] = [255, 255, 255]
        output[gray_mask & (gray <= threshold)] = [0, 0, 0]
        output[~gray_mask] = [255, 255, 255]
        self.image = output
        return self

    def get_end(self):
        white_mask = (self.white_labels == self.max_white_label).astype(
            np.uint8) * 255
        contours, hierarchy = cv2.findContours(
            white_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            return 0, None
        hole_areas = []
        hole_centers = []
        for i, h in enumerate(hierarchy[0]):
            if h[3] != -1:
                area = cv2.contourArea(contours[i])
                M = cv2.moments(contours[i])
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = 0, 0
                hole_areas.append(area)
                hole_centers.append((cx, cy))
        if not hole_areas:
            return 0, None
        max_idx = int(np.argmax(hole_areas))
        self.end_p = hole_centers[max_idx]
        return self.end_p

    def get_start(self, color_tol=40):
        max_area = 0
        best_center = None
        max_color = (0, 0, 0)
        for color in self.rarity_colors():
            lower = np.array([max(0, c - color_tol)
                             for c in color], dtype=np.uint8)
            upper = np.array([min(255, c + color_tol)
                             for c in color], dtype=np.uint8)
            mask = cv2.inRange(self.raw_image, lower, upper)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                mask, connectivity=8)
            if num_labels <= 1:
                continue
            max_idx = 1 + np.argmax(stats[1:, 4])
            area = stats[max_idx, 4]
            center = tuple(np.round(centroids[max_idx]).astype(int))
            if area > max_area:
                max_area = area
                best_center = center
                max_color = color

        if best_center is not None:
            self.start_p = best_center
            self.start_color = max_color
            self.inverse_start_color = tuple(
                255 - c for c in max_color)
            return self.start_p
        return None

    def dijkstra(self):
        mask = (self.white_labels == self.max_white_label).astype(
            np.uint8) * 255
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area_mask = np.zeros_like(mask)
        cv2.drawContours(area_mask, contours, -1, 255, -1)

        dt = cv2.distanceTransform(area_mask, cv2.DIST_L2, 5)
        h, w = dt.shape
        sx, sy = int(round(self.start_p[0])), int(round(self.start_p[1]))
        ex, ey = int(round(self.end_p[0])), int(round(self.end_p[1]))

        if area_mask[sy, sx] == 0 or area_mask[ey, ex] == 0:
            return None

        dist_map = -np.ones((h, w), dtype=np.float32)
        dist_map[sy, sx] = dt[sy, sx]
        prev = {}
        pq = [(-dt[sy, sx], (sx, sy))]
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        while pq:
            neg_dist, (x, y) = heappop(pq)
            curr = -neg_dist
            if (x, y) == (ex, ey):
                break
            if curr < dist_map[y, x]:
                continue
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and area_mask[ny, nx] == 255:
                    new_w = min(curr, dt[ny, nx])
                    if new_w > dist_map[ny, nx]:
                        dist_map[ny, nx] = new_w
                        prev[(nx, ny)] = (x, y)
                        heappush(pq, (-new_w, (nx, ny)))

        path = []
        cur = (ex, ey)
        if cur in prev or cur == (sx, sy):
            while cur != (sx, sy):
                path.append(cur)
                cur = prev[cur]
            path.append((sx, sy))
            path.reverse()
            self.sorted_points = path
            return path
        return None

    def rdp(self, epsilon=1):
        if self.sorted_points is None:
            return None
        self.rdp_points = rdp(self.sorted_points, epsilon=epsilon)
        return self.rdp_points

    def extend(self, length) -> None:
        if self.rdp_points:
            if len(self.rdp_points) < 2:
                self.extend_point = self.rdp_points[0]
                return
            end = self.rdp_points[-1]
            last = self.rdp_points[-2]
            l_l2_dist = distance.euclidean(end, last)
            sine_theta = (end[1] - last[1]) / l_l2_dist
            cosine_theta = (end[0] - last[0]) / l_l2_dist
            self.extend_point = (
                end[0] + length * cosine_theta, end[1] + length * sine_theta)

    def get_final(self, top_left_bound: tuple[int, int] = (0, 0), precise=True) -> list[tuple[int, int]]:
        if self.rdp_points:
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
        if self.length:
            return self.length
        if self.rdp_points:
            self.length = sum(distance.euclidean(self.rdp_points[i], self.rdp_points[i + 1])
                              for i in range(len(self.rdp_points) - 1))
        return self.length

    def get_width(self):
        if self.width:
            return self.width
        mask = (self.white_labels == self.max_white_label).astype(np.uint8)
        skeleton = skimage_skeletonize(mask > 0).astype(np.uint8)
        dist_map = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        thickness_list = dist_map[skeleton > 0] * 2
        self.width = np.mean(thickness_list)
        return self.width

    def get_difficulty(self, density_r=50) -> float:
        if self.difficulty is not None:
            return self.difficulty
        if self.rdp_points:
            points = np.array(self.rdp_points)
            tree = KDTree(points)
            neighbors = tree.query_ball_point(points, r=density_r)
            counts = np.array([len(n)-1 for n in neighbors])
            density = np.mean(counts)
            lwr = self.get_length()/self.width
            self.difficulty = (max(1, density*3)*lwr *
                               (len(self.rdp_points)**0.5))**0.5
            return self.difficulty


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


def test_environment(afk_seg_model, afk_det_model, shared_logger):
    if get_config()["advanced"]["environment"]:
        return
    initiate()
    log("Testing environment", "INFO")
    image = cv2.imread("./imgs/test.png")
    results = {}
    log("Testing YOLO", "INFO")

    t_start = time()
    afk_window_pos = detect_afk_window(image, afk_det_model)
    t_det_win = time()
    cropped_image = crop_image(
        afk_window_pos[0], afk_window_pos[1], image)
    start_p, end_p, start_size, pack = detect_afk_things(
        cropped_image, afk_det_model, caller="")
    t_det_things = time()
    res = get_masks_by_iou(cropped_image, afk_seg_model)
    t_seg = time()
    mask, _ = res
    afk_mask = AFK_Segment(image, mask, start_p, end_p, start_size)
    t_init_mask = time()
    if start_p is not None:
        afk_mask.save_start()
    afk_path = AFK_Path(afk_mask.segment_path(), start_p,
                        end_p, afk_mask.get_width())
    t_init_path = time()
    afk_path.dijkstra(afk_mask.mask)
    t_sort = time()
    afk_path.rdp(round(eval(get_config()["advanced"]["rdpEpsilon"].replace(
        "width", str(afk_mask.get_width())))))
    t_rdp = time()
    afk_path.extend(get_config()["advanced"]["extendLength"])
    afk_path.get_final(afk_window_pos[0], precise=False)
    t_final = time()
    log(f"===== YOLO TEST =====", "INFO")
    log(f"Det Win: {t_det_win - t_start:.5f}s/{t_det_win - t_start:.5f}s", "INFO")
    log("↑ This should take a bit longer because YOLO need to initialize the model.", "INFO")
    log(f"Det Things: {t_det_things - t_det_win:.5f}s/{t_det_things - t_start:.5f}s", "INFO")
    log(f"Segment: {t_seg - t_det_things:.5f}s/{t_seg - t_start:.5f}s", "INFO")
    log(f"Init Mask: {t_init_mask - t_seg:.5f}s/{t_init_mask - t_start:.5f}s", "INFO")
    log(f"Init Path: {t_init_path - t_init_mask:.5f}s/{t_init_path - t_start:.5f}s", "INFO")
    log(f"Sort Path: {t_sort - t_init_path:.5f}s/{t_sort - t_start:.5f}s", "INFO")
    log(f"RDP Path: {t_rdp - t_sort:.5f}s/{t_rdp - t_start:.5f}s", "INFO")
    log(f"Final Path: {t_final - t_rdp:.5f}s/{t_final - t_start:.5f}s", "INFO")
    log(f"Total: {t_final - t_start:.5f}s", "INFO")
    results['yolo'] = True

    log("Testing PyAutoGUI", "INFO")
    log(f"===== PyAutoGUI TEST =====", "INFO")
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
    log_ret("YOLO Test Time results saved to ./latest.log",
            "INFO", shared_logger, save=False)


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


def detect_afk_window(img, afk_det_model):
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
    if (1*(1-0.1) < ratio < 1*(1+0.1)) or (0.787*(1-0.1) < ratio < 0.787*(1+0.1)):
        return windows_pos
    return None


def detect_afk_things(cropped_img, afk_det_model, caller="main", test_time=None):
    afk_window_img = cropped_img.copy()
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
    pack = [None, None]
    for thing in things_afk[0]:
        if thing['name'] == 'Start' and thing['confidence'] > start_max_confidence:
            start_pos = (thing['x_avg'], thing['y_avg'])
            start_max_confidence = thing['confidence']
            start_size = (abs(thing['x_2'] - thing['x_1']) +
                          abs(thing['y_2'] - thing['y_1'])) / 2
            pack[0] = [(thing['x_1'], thing['y_1']),
                       (thing['x_2'], thing['y_2'])]
        if thing['name'] == 'End' and thing['confidence'] > end_max_confidence:
            end_pos = (thing['x_avg'], thing['y_avg'])
            end_max_confidence = thing['confidence']
            pack[1] = [(thing['x_1'], thing['y_1']),
                       (thing['x_2'], thing['y_2'])]
    return start_pos, end_pos, start_size, pack


def move_a_bit(
    interval_min=0.1,
    interval_max=0.3,
    epochs=3,
    num_outgoing_moves_range=(1, 3),
    return_segments_per_axis_range=(1, 2),
    min_return_segment_duration=0.05,
    pause_between_moves_range=(0.05, 0.15)
):
    def _generate_segments(total_duration, num_segments_desired, min_len_segment):
        if total_duration < 1e-9:
            return []

        num_segments = max(1, num_segments_desired)

        if total_duration < min_len_segment * num_segments:
            num_segments = max(1, int(total_duration / min_len_segment))

        if num_segments == 1:
            return [total_duration] if total_duration > 1e-9 else []
        base_allocations = [min_len_segment] * num_segments
        remaining_to_distribute = total_duration - sum(base_allocations)

        if remaining_to_distribute < 0:
            return [total_duration] if total_duration > 1e-9 else []

        durations = list(base_allocations)

        weights = [uniform(0.0, 1.0) for _ in range(num_segments)]
        sum_weights = sum(weights)

        if sum_weights < 1e-9:
            avg_add = remaining_to_distribute / num_segments if num_segments > 0 else 0
            for i in range(num_segments):
                durations[i] += avg_add
        else:
            for i in range(num_segments):
                durations[i] += (weights[i] / sum_weights) * \
                    remaining_to_distribute
        return [d for d in durations if d > 1e-9]

    directions = ["w", "d", "s", "a"]
    for epoch_num in range(epochs):
        net_displacement_time = {'w': 0.0, 's': 0.0, 'a': 0.0, 'd': 0.0}
        num_moves = randint(
            num_outgoing_moves_range[0], num_outgoing_moves_range[1])
        for _ in range(num_moves):
            dir_key = choice(directions)
            duration = uniform(interval_min, interval_max)

            pyautogui.keyDown(dir_key)
            sleep(duration)
            pyautogui.keyUp(dir_key)

            net_displacement_time[dir_key] += duration
            sleep(
                uniform(pause_between_moves_range[0], pause_between_moves_range[1]))
        delta_ws = net_displacement_time['w'] - net_displacement_time['s']
        delta_ad = net_displacement_time['d'] - net_displacement_time['a']
        return_actions_needed = []
        if delta_ws > 1e-9:
            return_actions_needed.append(('s', delta_ws))
        elif delta_ws < -1e-9:
            return_actions_needed.append(('w', -delta_ws))
        if delta_ad > 1e-9:
            return_actions_needed.append(('a', delta_ad))
        elif delta_ad < -1e-9:
            return_actions_needed.append(('d', -delta_ad))

        if not return_actions_needed:
            continue

        return_path_segments = []
        for return_dir, total_duration_needed in return_actions_needed:
            if total_duration_needed <= 1e-9:
                continue
            num_segments = randint(
                return_segments_per_axis_range[0], return_segments_per_axis_range[1])

            segments_for_this_dir = _generate_segments(
                total_duration_needed, num_segments, min_return_segment_duration)

            for seg_dur in segments_for_this_dir:
                if seg_dur > 1e-9:
                    return_path_segments.append((return_dir, seg_dur))

        shuffle(return_path_segments)
        for dir_key, duration in return_path_segments:
            pyautogui.keyDown(dir_key)
            sleep(duration)
            pyautogui.keyUp(dir_key)
            sleep(
                uniform(pause_between_moves_range[0], pause_between_moves_range[1]))


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
            if capture_method == "GDI":
                screenshot = gdi.gdi_capture(hwnd)
            elif capture_method == "BitBlt":
                screenshot = bitblt.bitblt_capture(hwnd)
            elif capture_method == "WGC":
                screenshot = wgc.wgc_capture(hwnd)
            screenshot = crop_image(
                left_top_bound, right_bottom_bound, screenshot)
            screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGBA2RGB)
        frames.append(screenshot)
    average_frame = np.mean(frames, axis=0).astype(np.uint8)
    return average_frame


def calculate_offset(path, hwnd):
    rect = gdi.get_fixed_window_rect(hwnd)
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


def export_segmentation_to_label(results, image, now, epsilon=1):
    if results[0].masks is None:
        return
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
    return labelme_data


def export_annotation_to_label(image, difficulty, now):
    image_resolution = f"{image.shape[1]}x{image.shape[0]}"
    client_resolution = f"{pyautogui.size().width}x{pyautogui.size().height}"
    label = {
        "timestamp": now,
        "difficulty": difficulty,
        "imageResolution": image_resolution,
        "clientResolution": client_resolution,
        "isBlockingAlpha": get_config()['extensions']['bgRemove'],
        "version": f"v{constants.VERSION_INFO}"
    }
    return label


def export_detection_to_label(packs, image, now):
    """
    classes = ["Window", "Start", "End"]
    """
    h, w = image.shape[:2]
    result = []
    for cls_idx, pack in enumerate(packs):
        if pack is None:
            continue
        (x1, y1), (x2, y2) = pack
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        cx = (x1 + x2) / 2 / w
        cy = (y1 + y2) / 2 / h
        result.append({
            "class": cls_idx,
            "position": [cx, cy, bw, bh],
        })
    return result


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
    try:
        eval(config["advanced"]["rdpEpsilon"].replace("width", str(1)))
    except (SyntaxError, NameError, TypeError, ValueError) as e:
        log_ret(f"Invalid RDP Epsilon Expression: {config['advanced']['rdpEpsilon']}",
                "CRITICAL", shared_logger, save=False)
        raise e
    if config['extensions']['autoChat']['selfUsername'] == "enter <username> here or chat will respond to your own messages" and config['extensions']['autoChat']:
        log_ret("You have not set your username in the config, chat will not work",
                "ERROR", shared_logger, save=False)


def draw_annotated_image(ori_image, line, start_p, end_p, window_pos, start_color, path_width, path_length, difficulty, sort_method) -> cv2.Mat:
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

    cv2.putText(image, f'Method: {sort_method}', (window_pos[0][0]+10, window_pos[1][1]-46), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255))
    cv2.putText(image, f'Difficulty: {difficulty:.2f}', (window_pos[0][0]+10, window_pos[1][1]-28), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255))
    cv2.putText(image, f'P_w: {path_width:.2f}, P_l: {path_length:.2f}', (window_pos[0][0]+10, window_pos[1][1]-10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255))
    return image


def locate_ready(image):
    ready = cv2.imread("./models/assets/Ready.PNG")
    screenshot = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    found = None
    for scale in np.linspace(0.5, 1.5, 20)[::-1]:
        resized = cv2.resize(ready, (0, 0), fx=scale, fy=scale)
        if resized.shape[0] > screenshot.shape[0] or resized.shape[1] > screenshot.shape[1]:
            continue
        result = cv2.matchTemplate(screenshot, resized, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        if found is None or max_val > found[0]:
            found = (max_val, max_loc, resized.shape[:2])

    if found and found[0] >= 0.7:
        max_val, max_loc, (h, w) = found
        center_x = max_loc[0] + w // 2
        center_y = max_loc[1] + h // 2
        return center_x, center_y, found[0]


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


def gh_upload_dataset(now):
    try:
        app_id = 1375071
        private_key = constants.GITHUB_TOKEN
        integration = GithubIntegration(app_id, private_key)
        install = integration.get_repo_installation(
            constants.DATASET_REPO.split('/')[0], constants.DATASET_REPO.split('/')[1])
        token = integration.get_access_token(install.id).token
    except Exception as e:
        log(f"Private key expired, {e}", "ERROR")
        return None
    g = Github(token)
    repo = g.get_repo(constants.DATASET_REPO)
    alias = str(uuid4()).replace("-", "")
    date = datetime.now().strftime("%Y-%m-%dT%H_%M_%SZ")
    with open(f"./train/images/{now}.png", "rb") as f:
        img_bytes = f.read()
    with open(f"./train/split/{now}.json", "r", encoding='utf-8') as f:
        seg_label_bytes = f.read().encode('utf-8')
    with open(f"./train/detection/{now}.txt", "r", encoding='utf-8') as f:
        det_label_bytes = f.read().encode('utf-8')
    with open(f"./train/annotation/{now}.json", "r") as f:
        anno_label_bytes = f.read().encode('utf-8')

    gh_upload_file(repo, f"./{alias}/{alias}.png",
                   img_bytes, f"{date} {alias[:5]} Image")
    gh_upload_file(repo, f"./{alias}/{alias}_seg.json",
                   seg_label_bytes, f"{date} {alias[:5]} Seg")
    gh_upload_file(repo, f"./{alias}/{alias}_det.txt",
                   det_label_bytes, f"{date} {alias[:5]} Det")
    gh_upload_file(repo, f"./{alias}/{alias}_anno.json",
                   anno_label_bytes, f"{date} {alias[:5]} Anno")
    log(f"Uploaded dataset to GitHub with hash {alias}", "INFO")
    return alias
