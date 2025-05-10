import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from tktooltip import ToolTip
import pygetwindow as gw
import pywinstyles
from PIL import Image, ImageTk, ImageDraw, ImageFont, ImageFilter
from tkinter import ttk
from darkdetect import theme as detect_theme
from sys import argv
from win32process import GetWindowThreadProcessId
from random import choices
from capture import bitblt, wgc
from segment_utils import *
from win11toast import toast
from playsound import playsound
import sv_ttk
import json
import ctypes
import torch
import tkinter as tk


config = get_config()
capture_windows = []


theme = detect_theme() if get_config(
)["gui"]["theme"] == "auto" else ("Dark" if get_config()["gui"]["theme"].lower() == "dark" else "Light")


def apply_theme_to_titlebar(root):
    version = getwindowsversion()
    theme = detect_theme() if get_config(
    )["gui"]["theme"] == "auto" else ("Dark" if get_config()["gui"]["theme"].lower() == "dark" else "Light")

    if version.major == 10 and version.build >= 22000:
        pywinstyles.change_header_color(
            root, "#1c1c1c" if theme == "Dark" else "#fafafa")
    elif version.major == 10:
        pywinstyles.apply_style(
            root, "dark" if theme == "Dark" else "normal")
        root.wm_attributes("-alpha", 0.99)
        root.wm_attributes("-alpha", 1)


def create_scrollable_frame(parent):
    canvas = tk.Canvas(parent, highlightthickness=0)
    scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True, pady=10, padx=10)
    scrollbar.pack(side="right", fill="y")

    def _on_mouse_wheel(event):
        canvas.yview_scroll(-1 * (event.delta // 120), "units")

    canvas.bind("<Enter>", lambda e: canvas.bind_all(
        "<MouseWheel>", _on_mouse_wheel))
    canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

    return scrollable_frame


def save_config_to_file(config_data, file_path="./config.json"):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=4, ensure_ascii=False)


def load_translations(lang="en-us"):
    translations = {}
    with open("./gui/i18n/"+lang+".txt", "r", encoding="utf-8") as f:
        for line in f:
            if "=" in line:
                key, value = line.strip().split("=", 1)
                translations[key] = value
    return translations


def translate(key, translations):
    return translations.get(key, key)


def create_settings_widgets(parent, config_data, parent_key=""):
    global theme, translations, config
    translations = load_translations(get_config()["gui"]["language"])
    for key, value in config_data.items():
        full_key = f"{parent_key}.{key}" if parent_key else key
        display_key = translate(full_key, translations)
        tips = get_settings_tips(full_key)
        if isinstance(value, dict):
            label = ttk.Label(
                parent, text=f"> {display_key}:", font=("Microsoft Yahei", 14, "bold"))
            label.pack(anchor="w", pady=5)
            ToolTip(label, msg=tips, delay=0.2)
            create_settings_widgets(parent, value, full_key)
        elif isinstance(value, list):
            label = ttk.Label(
                parent, text=f"{display_key}:", font=("Microsoft Yahei", 10))
            label.pack(anchor="w", pady=5)
            ToolTip(label, msg=tips, delay=0.2)
            entry = ttk.Entry(parent)
            entry.insert(0, json.dumps(value))
            entry.pack(fill="x", pady=2)

            def update_list(event, key=key):
                try:
                    config_data[key] = json.loads(entry.get())
                    parent_key = None
                    for k, v in config.items():
                        if list(v.keys()) == list(config_data.keys()):
                            parent_key = k
                            break
                    if parent_key:
                        config[parent_key] = config_data
                    save_config_to_file(config)
                except json.JSONDecodeError:
                    pass
            entry.bind("<Return>", update_list)
            entry.bind("<FocusOut>", update_list)
        elif isinstance(value, bool):
            var = tk.BooleanVar(value=value)
            checkbutton = ttk.Checkbutton(
                parent, text=display_key, variable=var)
            checkbutton.pack(anchor="w", pady=5)
            ToolTip(checkbutton, msg=tips, delay=0.2)
            checkbutton_font = ("Microsoft Yahei", 10)
            checkbutton.configure(style="Custom.TCheckbutton")
            style = ttk.Style()
            style.configure("Custom.TCheckbutton", font=checkbutton_font)

            def update_bool(*args, key=key, var=var):
                global config
                config_data[key] = var.get()
                parent_key = None
                for k, v in config.items():
                    if list(v.keys()) == list(config_data.keys()):
                        parent_key = k
                        break
                if parent_key:
                    config[parent_key] = config_data
                save_config_to_file(config)
            var.trace_add("write", update_bool)

        elif isinstance(value, (int, float)):
            label = ttk.Label(
                parent, text=f"{display_key}:", font=("Microsoft Yahei", 10))
            label.pack(anchor="w", pady=5)
            ToolTip(label, msg=tips, delay=0.2)
            entry = ttk.Entry(parent)
            entry.insert(0, str(value))
            entry.pack(fill="x", pady=2)

            def validate_entry_input(P):
                if P == "" or P == "-":
                    return True
                try:
                    float(P)
                    return True
                except ValueError:
                    return False

            validate_cmd = parent.register(validate_entry_input)
            entry.config(validate="key", validatecommand=(validate_cmd, "%P"))

            def update_number(event=None, key=key, entry=entry):
                try:
                    current_value = entry.get()
                    if current_value == "" or current_value == ".":
                        return
                    new_value = int(current_value) if "." not in current_value else float(
                        current_value)
                    config_data[key] = new_value
                    parent_key = None
                    for k, v in config.items():
                        if list(v.keys()) == list(config_data.keys()):
                            parent_key = k
                            break
                    if parent_key:
                        config[parent_key] = config_data
                    save_config_to_file(config)
                except ValueError:
                    pass
            entry.bind("<FocusOut>", update_number)
            entry.bind("<Return>", update_number)
        else:
            label = ttk.Label(
                parent, text=f"{display_key}:", font=("Microsoft Yahei", 10))
            label.pack(anchor="w", pady=5)
            ToolTip(label, msg=tips, delay=0.2)
            entry = ttk.Entry(parent)
            entry.insert(0, str(value))
            entry.pack(fill="x", pady=2)

            def update_other(event, key=key, entry=entry):
                config_data[key] = entry.get()
                parent_key = None
                for k, v in config.items():
                    if list(v.keys()) == list(config_data.keys()):
                        parent_key = k
                        break
                if parent_key:
                    config[parent_key] = config_data
                save_config_to_file(config)

            entry.bind("<FocusOut>", update_other)
            entry.bind("<Return>", update_other)


def create_fill_image(image, width, height):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGBA))
    tiled_image = Image.new("RGBA", (width, height))
    img_width, img_height = image.size
    for x in range(0, width, img_width):
        for y in range(0, height, img_height):
            tiled_image.paste(image, (x, y))

    return cv2.cvtColor(np.array(tiled_image), cv2.COLOR_RGBA2BGRA)


def create_rounded_image(image, width, height, radius, theme):
    if width <= 0 or height <= 0:
        raise ValueError(
            f"Invalid width ({width}) or height ({height}) for resizing.")
    image = cv2.resize(image, (width, height))
    if image.shape[2] == 3:  # If the image is RGB (3 channels)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    bg_color = (28, 28, 28, 255) if theme == "Dark" else (250, 250, 250, 255)
    background = np.full((height, width, 4), bg_color, dtype=np.uint8)
    mask = np.zeros((height, width, 4), dtype=np.uint8)
    mask[:, :, 3] = 255  # Set alpha channel to fully opaque
    cv2.rectangle(mask, (radius, 0), (width - radius, height),
                  (255, 255, 255, 255), -1)
    cv2.rectangle(mask, (0, radius), (width, height - radius),
                  (255, 255, 255, 255), -1)
    cv2.circle(mask, (radius, radius), radius, (255, 255, 255, 255), -1)
    cv2.circle(mask, (width - radius, radius),
               radius, (255, 255, 255, 255), -1)
    cv2.circle(mask, (radius, height - radius),
               radius, (255, 255, 255, 255), -1)
    cv2.circle(mask, (width - radius, height - radius),
               radius, (255, 255, 255, 255), -1)

    rounded_image = cv2.bitwise_and(image, mask)
    inverted_mask = cv2.bitwise_not(mask)
    rounded_image = cv2.add(
        rounded_image, cv2.bitwise_and(background, inverted_mask))
    rounded_image = draw_version(rounded_image, width, height)
    rounded_image = cv2.cvtColor(rounded_image, cv2.COLOR_BGRA2RGBA)
    return Image.fromarray(rounded_image)


def draw_version(image, width, height):
    rounded_image = draw_text_pil(
        image,
        "florr-auto-afk",
        (width // 2, height // 2 - 5),
        "./gui/Ubuntu-R.ttf",
        50,
        (255, 255, 255),
        align="center",
        outline_color=(0, 0, 0),
        outline_width=2,
    )
    text_width, text_height = ImageDraw.Draw(
        Image.fromarray(rounded_image)).textsize("florr-auto-afk", font=ImageFont.truetype("./gui/Ubuntu-R.ttf", 50))

    rounded_image = draw_text_pil(
        rounded_image,
        f"v{constants.VERSION_INFO}",
        ((width+text_width) // 2+5, (height+text_height) // 2-28),
        "./gui/Ubuntu-R.ttf",
        25,
        (255, 255, 255),
        align="left",
        outline_color=(0, 0, 0),
        outline_width=1,
    )

    rounded_image = draw_text_pil(
        rounded_image,
        f"GitHub: https://github.com/{constants.PROJECT_REPO}",
        (width-15, height-30),
        "./gui/Ubuntu-R.ttf",
        15,
        (255, 255, 255),
        align="right",
        outline_color=(0, 0, 0),
        outline_width=1,
    )
    return rounded_image


def draw_text_pil(img, text, position, font_path, font_size,
                  color=(255, 255, 255), align='left',
                  outline_color=(0, 0, 0), outline_width=2,
                  shadow_color=(50, 50, 50), shadow_offset=(3, 3),
                  shadow_blur_radius=5):
    pil_img = Image.fromarray(img).convert("RGBA")
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.truetype(font_path, font_size)

    text_width, text_height = draw.textsize(text, font=font)

    x, y = position
    if align == 'center':
        x -= text_width // 2
        y -= text_height // 2
    elif align == 'right':
        x -= text_width
    shadow_img = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow_img)
    shadow_x = x + shadow_offset[0]
    shadow_y = y + shadow_offset[1]
    shadow_draw.text((shadow_x, shadow_y), text, font=font,
                     fill=shadow_color + (255,))
    if shadow_blur_radius > 0:
        shadow_img = shadow_img.filter(
            ImageFilter.GaussianBlur(radius=shadow_blur_radius))
    pil_img = Image.alpha_composite(pil_img, shadow_img)
    draw = ImageDraw.Draw(pil_img)
    if outline_width > 0:
        for dx in range(-outline_width, outline_width + 1):
            for dy in range(-outline_width, outline_width + 1):
                if dx == 0 and dy == 0:
                    continue
                draw.text((x + dx, y + dy), text,
                          font=font, fill=outline_color)
    draw.text((x, y), text, font=font, fill=color)
    return np.array(pil_img.convert("RGB"))


def create_fit_image(image, canvas_width, canvas_height, alignment="center", max_image_width=None, max_image_height=None):
    if isinstance(image, np.ndarray):
        if image.shape[2] == 4:
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA))
        else:
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGBA))

    original_img_width, original_img_height = image.size

    primary_scale_factor = float(canvas_width) / original_img_width

    if max_image_width is not None:
        if (original_img_width * primary_scale_factor) > max_image_width:
            primary_scale_factor = float(max_image_width) / original_img_width

    if max_image_height is not None:
        if (original_img_height * primary_scale_factor) > max_image_height:
            primary_scale_factor = float(
                max_image_height) / original_img_height

    scaled_img_width = int(original_img_width * primary_scale_factor)
    scaled_img_height = int(original_img_height * primary_scale_factor)

    scaled_img_width = max(1, scaled_img_width)
    scaled_img_height = max(1, scaled_img_height)

    if scaled_img_width != original_img_width or scaled_img_height != original_img_height:

        image = image.resize(
            (scaled_img_width, scaled_img_height), Image.Resampling.LANCZOS)

    canvas = Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 0))

    offset_x = 0
    offset_y = 0

    if alignment == "center":
        offset_x = (canvas_width - scaled_img_width) // 2
        offset_y = (canvas_height - scaled_img_height) // 2
    elif alignment == "left":
        offset_x = 0
        offset_y = (canvas_height - scaled_img_height) // 2
    elif alignment == "right":
        offset_x = canvas_width - scaled_img_width
        offset_y = (canvas_height - scaled_img_height) // 2
    elif alignment == "top":
        offset_x = (canvas_width - scaled_img_width) // 2
        offset_y = 0
    elif alignment == "bottom":
        offset_x = (canvas_width - scaled_img_width) // 2
        offset_y = canvas_height - scaled_img_height
    elif alignment == "top_left":
        offset_x = 0
        offset_y = 0
    elif alignment == "top_right":
        offset_x = canvas_width - scaled_img_width
        offset_y = 0
    elif alignment == "bottom_left":
        offset_x = 0
        offset_y = canvas_height - scaled_img_height
    elif alignment == "bottom_right":
        offset_x = canvas_width - scaled_img_width
        offset_y = canvas_height - scaled_img_height

    offset_x = max(0, offset_x)
    offset_y = max(0, offset_y)

    canvas.paste(image, (offset_x, offset_y))

    return cv2.cvtColor(np.array(canvas), cv2.COLOR_RGBA2BGRA)


def add_rounded_image_to_canvas(main_content, image, theme, height=200, radius=20):
    def update_image(event=None):
        if not canvas.winfo_exists():
            return
        available_width = main_content.winfo_width()
        if available_width <= 0:
            return
        width = int(available_width - 20)
        if width <= 0 or height <= 0:
            return
        image_np = cv2.imread(image["path"])
        if image["type"] == "copy":
            image_np = cv2.resize(image_np, (300, 300))
            tiled_image = create_fill_image(image_np, width, height)
        elif image["type"] == "tile":
            tiled_image = create_fit_image(
                image_np, width, height)
        rounded_image = create_rounded_image(
            tiled_image, width, height, radius, theme)
        rounded_image_tk = ImageTk.PhotoImage(rounded_image)
        canvas.delete("all")
        canvas.config(width=width, height=height)
        canvas.create_image(0, 0, anchor="nw", image=rounded_image_tk)
        canvas.image = rounded_image_tk
    canvas = tk.Canvas(main_content, bg="white", highlightthickness=0)
    canvas.place(anchor="center", relx=0.5, y=height/2+10)
    main_content.bind("<Configure>", update_image)
    update_image()
    return canvas


def generate_announcement(skip_update=False):
    if not skip_update:
        upd = check_update()
    else:
        upd = None
    if upd == None:
        update_msg = "Failed to check for updates."
    else:
        if upd[0] and not skip_update:
            new_update, remote_version = upd
            remote_version = '.'.join([str(i) for i in remote_version])
            update_msg = f"New version v{remote_version} is available!"
        elif not upd[0] and not skip_update:
            update_msg = "No new updates available."
        elif skip_update:
            update_msg = "Update check skipped."
    changelog_msg = "Changelog:\n"
    latest = constants.CHANGELOG.get(constants.VERSION_INFO, [])
    if latest or skip_update:
        changelog_msg += f"Version {constants.VERSION_INFO}:\n"
        for change in latest:
            changelog_msg += f"- {change}\n"
    else:
        if not skip_update:
            latest_v = get_changelog()
            if latest_v:
                latest = latest_v[list(latest_v.items())[0][0]]
                changelog_msg += f"Version {list(latest_v.items())[0][0]}:\n"
                for change in latest:
                    changelog_msg += f"- {change}\n"
            else:
                changelog_msg += "No changelog available.\n"
        else:
            changelog_msg += "No changelog available.\n"
    return "\n\n".join([update_msg, changelog_msg])


def get_changelog():
    url = f"https://raw.githubusercontent.com/{constants.ASSET_REPO}/refs/heads/main/CHANGELOG.json"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            changelog = response.json()
            return changelog
        else:
            try:
                url = "https://github.moeyy.xyz/"+url
                response = requests.get(url)
                if response.status_code == 200:
                    changelog = response.json()
                    return changelog
                else:
                    return None
            except:
                return None
    except:
        return None


def open_capture_window(root, main_content):
    def update_preview():
        selected_title = choose_window.get()
        if selected_title == "Select a window":
            create_window.after(1000, update_preview)
            return
        available_windows = gw.getWindowsWithTitle("")
        names = [
            f"[{Process(GetWindowThreadProcessId(w._hWnd)[1]).name()}]: {w.title}" for w in available_windows]
        w = available_windows[names.index(selected_title)]
        hwnd = w._hWnd
        if str(w.title).strip().endswith("Microsoft​ Edge") or str(w.title).strip().endswith("Microsoft​ Edge Beta"):
            notice_label.config(
                text="Note: canvas won't auto refresh with Release Microsoft Edge or Edge Beta, use Edge Dev/Canary instead")
        elif str(w.title).strip().endswith("Google Chrome"):
            notice_label.config(
                text="Note: make sure you disable `CalculateNativeWinOcclusion`")
        else:
            notice_label.config(text="")
        try:
            if capture_method.get() == "BitBlt":
                frame = bitblt.bitblt_capture(hwnd)
            elif capture_method.get() == "Windows Graphics Capture":
                frame = wgc.wgc_capture(hwnd)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA)
            h, w = frame.shape[:2]
            scale = min(800 / w, 400 / h)
            new_w, new_h = int(w * scale), int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h))
            img = Image.fromarray(frame)
            img_tk = ImageTk.PhotoImage(img)
            preview_label.config(image=img_tk)
            preview_label.image = img_tk
        except Exception as e:
            print(f"Error capturing window: {e}")

        create_window.after(1000, update_preview)
    create_window = tk.Toplevel(root)
    create_window.title(get_ui_translation("capture_winname"))
    create_window.iconbitmap("./gui/icon.ico")
    create_window.geometry("1000x600")
    create_window.resizable(False, False)
    preview_label = ttk.Label(
        create_window, text="Preview will appear here", anchor="center")
    preview_label.pack(pady=10, padx=10, fill="both", expand=True)
    bottom_frame = ttk.Frame(create_window)
    bottom_frame.pack(pady=10, padx=20, fill="x")
    notice_label = ttk.Label(
        bottom_frame, text="", font=("Microsoft Yahei", 9), foreground="#ef5350")
    notice_label.pack(padx=10, pady=10)

    available_windows_label = ttk.Label(
        bottom_frame, text=get_ui_translation("capture_windows"), font=("Microsoft Yahei", 12))
    available_windows_label.pack(side="left", padx=10)

    available_windows = gw.getWindowsWithTitle("")
    available_windows = [w for w in available_windows if w.title != ""]
    available_windows = sorted(
        available_windows, key=lambda w: w.title.lower())
    choose_window = ttk.Combobox(
        bottom_frame,
        name="window_selector",
        values=[
            f"[{Process(GetWindowThreadProcessId(w._hWnd)[1]).name()}]: {w.title}" for w in available_windows],
        state="readonly",
        font=("Microsoft Yahei", 12),
    )
    choose_window.set("Select a window")
    choose_window.pack(side="left", fill="x", expand=True, padx=10)
    capture_method_frame = ttk.Frame(create_window)
    capture_method_frame.pack(pady=10, padx=20, fill="x")
    capture_method_label = ttk.Label(
        capture_method_frame, text=get_ui_translation("capture_method"), font=("Microsoft Yahei", 12))
    capture_method_label.pack(side="left", padx=10)
    capture_method = tk.StringVar(value="Windows Graphics Capture")
    capture_method_selector = ttk.Combobox(
        capture_method_frame,
        textvariable=capture_method,
        values=["Windows Graphics Capture", "BitBlt"],
        state="readonly",
        font=("Microsoft Yahei", 12),
    )
    available_windows = gw.getWindowsWithTitle("")
    available_windows = [w for w in available_windows if w.title != ""]
    available_windows = sorted(
        available_windows, key=lambda w: w.title.lower())
    choose_window["values"] = [
        f"[{Process(GetWindowThreadProcessId(w._hWnd)[1]).name()}]: {w.title}" for w in available_windows]
    capture_method_selector.pack(side="left", fill="x", expand=True, padx=10)
    choose_window.bind("<<ComboboxSelected>>", lambda e: update_preview())
    add_button = ttk.Button(
        bottom_frame,
        text=get_ui_translation("capture_add"),
        command=lambda: add_window_hook(
            choose_window.get(), capture_method.get(), create_window, main_content),
    )
    add_button.pack(side="left", padx=10)
    apply_theme_to_titlebar(create_window)


def add_window_hook(title, capture_method, create_window, main_content):
    global capture_windows
    if title == "Select a window":
        return
    available_windows = gw.getWindowsWithTitle("")
    names = [
        f"[{Process(GetWindowThreadProcessId(w._hWnd)[1]).name()}]: {w.title}" for w in available_windows]
    available_windows = available_windows[names.index(title)]
    hwnd = available_windows._hWnd
    for w in capture_windows:
        if w["hwnd"] == hwnd:
            create_window.destroy()
            return
    if capture_method == "BitBlt":
        thumbnail = bitblt.bitblt_capture(hwnd)
    elif capture_method == "Windows Graphics Capture":
        thumbnail = wgc.wgc_capture(hwnd)
    window = {
        "title": title,
        "hwnd": hwnd,
        "capture_method": capture_method,
        "thumbnail": thumbnail,
    }
    capture_windows.append(window)
    create_window.destroy()
    update_capture_menu(main_content, window)


def update_capture_menu(main_content, window):
    global capture_windows
    frame = ttk.Frame(main_content, relief="solid", borderwidth=1, padding=5)
    frame.pack(fill="x", padx=10, pady=5)
    thumbnail_image = Image.fromarray(cv2.cvtColor(
        window['thumbnail'], cv2.COLOR_BGRA2RGBA))
    thumbnail_image.thumbnail((100, 100))
    thumbnail_image_tk = ImageTk.PhotoImage(thumbnail_image)
    thumbnail_label = ttk.Label(frame, image=thumbnail_image_tk)
    thumbnail_label.image = thumbnail_image_tk
    thumbnail_label.pack(side="left", padx=10, pady=5)
    details_frame = ttk.Frame(frame)
    details_frame.pack(side="left", fill="x", expand=True, padx=10)
    title_label = ttk.Label(
        details_frame, text=window['title'], font=("Microsoft Yahei", 12, "bold"))
    title_label.pack(anchor="w")
    method_label = ttk.Label(
        details_frame, text=window['capture_method'], font=("Microsoft Yahei", 10))
    method_label.pack(anchor="w")
    hwnd_label = ttk.Label(
        details_frame, text=window['hwnd'], font=("Microsoft Yahei", 10))
    hwnd_label.pack(anchor="w")

    def delete_window():
        global capture_windows
        capture_windows = [
            w for w in capture_windows if w["hwnd"] != window["hwnd"]]
        frame.destroy()

    delete_button = ttk.Button(
        frame, text="Delete", command=delete_window)
    delete_button.pack(side="right", padx=10, pady=5)


def get_ui_translation(key_, lang=get_config()["gui"]["language"]):
    translations = {}
    with open("./gui/i18n/ui_"+lang+".txt", "r", encoding="utf-8") as f:
        for line in f:
            if "=" in line:
                key, value = line.strip().split("=", 1)
                translations[key] = value
    return translations.get(key_, key_)


def get_settings_tips(key_, lang=get_config()["gui"]["language"]):
    translations = {}
    with open("./gui/i18n/tip_"+lang+".txt", "r", encoding="utf-8") as f:
        for line in f:
            if "=" in line:
                key, value = line.strip().split("=", 1)
                translations[key] = value
    return translations.get(key_, key_)


def send_notification(message: str):
    if get_config()["runs"]["notify"]:
        icon_path = path.join(path.dirname(
            path.abspath(argv[0])), 'gui', 'icon.ico')
        toast('florr-auto-afk', message,
              icon=icon_path)


def send_sound_notification(epoch: int = 3):
    if get_config()["runs"]["sound"]:
        for _ in range(epoch):
            playsound(get_config()["runs"]["soundPath"])


def get_random_background():
    with open("./gui/backgrounds/structure.json", "r") as f:
        structure = json.load(f)
    weights = [bg["weight"] for bg in structure]
    selected_bg = choices(structure, weights=weights, k=1)[0]
    return selected_bg
