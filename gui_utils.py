import pywinstyles
from PIL import Image, ImageTk, ImageDraw, ImageFont
import ctypes
import tkinter as tk
from tkinter import ttk
import darkdetect
import sv_ttk
import json
from win32mica import ApplyMica, MICAMODE
from win32gui import GetParent
from segment_utils import *

config = get_config()

theme = darkdetect.theme() if get_config(
)["gui"]["theme"] == "auto" else ("Dark" if get_config()["gui"]["theme"].lower() == "dark" else "Light")


def apply_theme_to_titlebar(root):
    version = getwindowsversion()
    theme = darkdetect.theme() if get_config(
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
    """Create a scrollable frame with a canvas and scrollbar."""
    canvas = tk.Canvas(parent, highlightthickness=0)  # Remove blue border
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
    """Save the updated configuration to a file."""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=4, ensure_ascii=False)


def load_translations(lang="en-us"):
    """Load translations from a file into a dictionary."""
    translations = {}
    with open("./gui/i18n/"+lang+".txt", "r", encoding="utf-8") as f:
        for line in f:
            if "=" in line:
                key, value = line.strip().split("=", 1)
                translations[key] = value
    return translations


def translate(key, translations):
    """Translate a key using the loaded translations."""
    return translations.get(key, key)


def create_settings_widgets(parent, config_data, parent_key=""):
    global theme, translations, config
    translations = load_translations(get_config()["gui"]["language"])
    for key, value in config_data.items():
        full_key = f"{parent_key}.{key}" if parent_key else key
        display_key = translate(full_key, translations)
        if isinstance(value, dict):
            label = ttk.Label(
                parent, text=f"> {display_key}:", font=("Microsoft Yahei", 14, "bold"))
            label.pack(anchor="w", pady=5)
            create_settings_widgets(parent, value, full_key)
        elif isinstance(value, list):
            label = ttk.Label(
                parent, text=f"{display_key}:", font=("Microsoft Yahei", 10))
            label.pack(anchor="w", pady=5)
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
                        config[parent_key] = config_data  # 同步修改到全局 config

                    save_config_to_file(config)
                except json.JSONDecodeError:
                    pass  # 忽略无效的 JSON
            entry.bind("<Return>", update_list)
            entry.bind("<FocusOut>", update_list)
        elif isinstance(value, bool):
            var = tk.BooleanVar(value=value)
            checkbutton = ttk.Checkbutton(
                parent, text=display_key, variable=var)
            checkbutton.pack(anchor="w", pady=5)
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
                        config[parent_key] = config_data  # 同步修改到全局 config
                    save_config_to_file(config)
                except ValueError:
                    pass
            entry.bind("<FocusOut>", update_number)
            entry.bind("<Return>", update_number)
        else:
            label = ttk.Label(
                parent, text=f"{display_key}:", font=("Microsoft Yahei", 10))
            label.pack(anchor="w", pady=5)
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


def create_rounded_image(image_path, width, height, radius, theme):
    if width <= 0 or height <= 0:
        raise ValueError(
            f"Invalid width ({width}) or height ({height}) for resizing.")
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
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

    rounded_image = draw_text_pil(
        rounded_image,
        f"florr-auto-afk v{constants.VERSION_INFO}",
        (width // 2, height // 2 - 5),
        "./gui/Ubuntu-R.ttf",
        50,
        (255, 255, 255),
        align="center",
        outline_color=(0, 0, 0),
        outline_width=2,
    )

    rounded_image = cv2.cvtColor(rounded_image, cv2.COLOR_BGRA2RGBA)
    return Image.fromarray(rounded_image)


def draw_text_pil(img, text, position, font_path, font_size, color=(255, 255, 255), align='left', outline_color=(0, 0, 0), outline_width=2):
    """Draw text on an image with specified alignment and an optional outline."""
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.truetype(font_path, font_size)

    text_width, text_height = draw.textsize(text, font=font)

    x, y = position
    if align == 'center':
        x -= text_width // 2
        y -= text_height // 2
    elif align == 'right':
        x -= text_width

    if outline_width > 0:
        for dx in range(-outline_width, outline_width + 1):
            for dy in range(-outline_width, outline_width + 1):

                if dx == 0 and dy == 0:
                    continue
                draw.text((x + dx, y + dy), text,
                          font=font, fill=outline_color)

    draw.text((x, y), text, font=font, fill=color)

    return np.array(pil_img)


def add_rounded_image_to_canvas(main_content, image_path, theme, height=200, radius=20):
    """Add a rounded rectangle image to the Tkinter canvas with dynamic width."""
    def update_image(event=None):
        if not canvas.winfo_exists():
            return  # 如果 canvas 已被销毁，停止更新
        available_width = main_content.winfo_width()

        if available_width <= 0:
            return

        width = int(available_width - 20)

        if width <= 0 or height <= 0:
            return

        rounded_image = create_rounded_image(
            image_path, width, height, radius, theme)

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


def Get_hWnd(root, hWnd=None):
    hWnd = GetParent(root.winfo_id())
    ApplyMica(hWnd, MICAMODE.DARK)


def generate_announcement():
    new_update, remote_version = check_update()
    remote_version = '.'.join([str(i) for i in remote_version])
    if new_update == None:
        return "Failed to check for updates."
    if new_update:
        message = f"New version v{remote_version} is available!"
        return message
    else:
        return "No new updates available."
