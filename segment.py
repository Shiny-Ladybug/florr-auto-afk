from gui_utils import *
import multiprocessing


def test_idle_thread(idled_flag, suppress_idle_detection, shared_logger):
    last_pos = None
    idle_iter = 0
    sleep_time = get_config()["runs"]["idleDetInterval"]
    while True:
        with suppress_idle_detection.get_lock():
            if suppress_idle_detection.value:
                sleep(1)
                continue
        pos = pyautogui.position()
        if last_pos != pos:
            idle_iter = 0
            sleep_time = get_config()["runs"]["idleDetInterval"]
            if idled_flag.value == True:
                with idled_flag.get_lock():
                    idled_flag.value = False
                    log_ret("No longer idle", "EVENT",
                            shared_logger, save=False)
            last_pos = pos
        else:
            idle_iter += 1
        if idle_iter > (get_config()["runs"]["idleTimeThreshold"]//get_config()["runs"]["idleDetInterval"]):
            if idled_flag.value == False:
                with idled_flag.get_lock():
                    idled_flag.value = True
                    log_ret("Idle detected", "EVENT",
                            shared_logger, save=False)
            sleep_time = get_config()["runs"]["idleDetIntervalMax"]
        sleep(sleep_time)


def afk_thread(idled_flag, suppress_idle_detection, shared_logger, capture_windows):
    try:
        debugger("Loading YOLO models")
        afk_seg_model = YOLO(get_config()["yoloConfig"]["segModel"])
        afk_det_model = YOLO(get_config()["yoloConfig"]["detModel"])
        debugger("Loaded YOLO models")
    except:
        log_ret("YOLO models are corrupted, trying to restore files",
                "ERROR", shared_logger)
        remove(get_config()["yoloConfig"]["segModel"])
        remove(get_config()["yoloConfig"]["detModel"])
        remove("./models/version")
        return
    debugger("Loaded configs", get_config())
    test_environment(afk_seg_model)
    log_ret("恭喜你，你成功把代码跑起来了，你是这个👍", "INFO", shared_logger, save=False)
    countdown = get_config()["runs"]["runningCountDown"]
    if countdown == -1:
        log_ret("Running indefinitely", "INFO", shared_logger)
        eta_timestamp = -1
    else:
        log_ret(f"Running for {countdown} minutes", "INFO")
        eta_timestamp = int(time()) + countdown * 60
        eta_time = datetime.strftime(
            datetime.fromtimestamp(eta_timestamp), '%Y-%m-%d %H:%M:%S')
        log_ret(f"ETA: {eta_time} / {eta_timestamp}", "INFO")
    if capture_windows == []:
        log_ret("Fullscreen detection enabled", "INFO", shared_logger)
    else:
        log_ret(
            f"Window detection for {[w['title'] for w in capture_windows]} enabled", "INFO", shared_logger)
    while True:
        with idled_flag.get_lock():
            if not idled_flag.value:
                sleep(1)
                continue
        if capture_windows == []:
            image = pyautogui.screenshot()
            image = cv2.cvtColor(
                np.array(image), cv2.COLOR_RGB2BGR)
            ori_image = image.copy()
            position = detect_afk(image, afk_det_model)
            if position == None:
                debugger("No AFK window found")
                if get_config()["advanced"]["verbose"]:
                    log("No AFK window found", "EVENT", save=False)
                if countdown != -1 and time() > eta_timestamp:
                    log_ret("Countdown Ends, program exiting", "EVENT")
                    break
                sleep(get_config()["advanced"]["epochInterval"])
                continue
            log_ret("Found AFK window", "EVENT", shared_logger)
            debugger("Found AFK window")
            multiprocessing.Process(
                target=send_notification, args=("AFK Detected",)).start()
            multiprocessing.Process(
                target=send_sound_notification, args=(3,)).start()

            if get_config()["executeBinary"]["runBeforeAFK"] != "":
                try:
                    system("start "+get_config()
                           ["executeBinary"]["runBeforeAFK"])
                except:
                    log_ret("Cannot execute extra binary", "ERROR")

            save_image(image, "afk", "afk")
            left_top_bound = position[2][0]
            right_bottom_bound = position[2][1]
            debugger("Bounds", left_top_bound, right_bottom_bound)
            if get_config()["exposure"]["enable"]:
                if get_config()["exposure"]["moveInterval"] > 0:
                    multiprocessing.Process(
                        target=move_a_bit, args=(get_config()["exposure"]["moveInterval"],)).start()
                image = exposure_image(
                    left_top_bound, right_bottom_bound, get_config()["exposure"]["duration"])
                save_image(image, "exposure", "afk")
            else:
                image = crop_image(
                    left_top_bound, right_bottom_bound, ori_image)
            execute_afk(position, ori_image, image, afk_seg_model,
                        left_top_bound, right_bottom_bound, suppress_idle_detection, shared_logger, type="fullscreen")
        else:
            results = []
            for window in capture_windows:
                if window["capture_method"] == "BitBlt":
                    image = bitblt.bitblt_capture(window["hwnd"])
                elif window["capture_method"] == "Windows Graphics Capture":
                    image = wgc.wgc_capture(window["hwnd"])
                image = cv2.cvtColor(
                    np.array(image), cv2.COLOR_RGBA2RGB)
                ori_image = image.copy()
                position = detect_afk(image, afk_det_model)
                if position == None:
                    results.append(False)
                    continue
                log_ret(
                    f"Found AFK window in {window['title']}", "EVENT", shared_logger)

                multiprocessing.Process(
                    target=send_notification, args=("AFK Detected",)).start()
                multiprocessing.Process(
                    target=send_sound_notification, args=(3,)).start()

                if get_config()["executeBinary"]["runBeforeAFK"] != "":
                    try:
                        system("start "+get_config()
                               ["executeBinary"]["runBeforeAFK"])
                    except:
                        log_ret("Cannot execute extra binary", "ERROR")
                save_image(image, "afk", "afk")
                left_top_bound = position[2][0]
                right_bottom_bound = position[2][1]
                if get_config()["exposure"]["enable"]:
                    image = exposure_image(
                        left_top_bound, right_bottom_bound, get_config()["exposure"]["duration"], window["hwnd"], window["capture_method"])
                    save_image(image, "exposure", "afk")
                else:
                    image = crop_image(
                        left_top_bound, right_bottom_bound, ori_image)
                execute_afk(position, ori_image, image, afk_seg_model,
                            left_top_bound, right_bottom_bound, suppress_idle_detection, shared_logger, type="window", hwnd=window['hwnd'])
                results.append(True)
            if not any(results):
                if get_config()["advanced"]["verbose"]:
                    log("No AFK window found", "EVENT", save=False)
                if countdown != -1 and time() > eta_timestamp:
                    log_ret("Countdown Ends, program exiting", "EVENT")
                    break
                sleep(get_config()["advanced"]["epochInterval"])
                continue


def execute_afk(position, ori_image, image, afk_seg_model, left_top_bound, right_bottom_bound, suppress_idle_detection, shared_logger, type="fullscreen", hwnd=None):
    mask = get_masks_by_iou(image, afk_seg_model)
    '''
    results = afk_seg_model.predict(image, retina_masks=True, verbose=False)
    masks = results[0].masks
    masks = pred.segmentation.onnx_seg_afk(afk_seg_model, image)
    '''
    start = position[0]
    end = position[1]
    if start != None:
        start = (round(position[0][0]), round(position[0][1]))
        start = (start[0] + left_top_bound[0],
                 start[1] + left_top_bound[1])
    else:
        log_ret("No start found", "WARNING", shared_logger)
    if end != None:
        end = (round(position[1][0]), round(position[1][1]))
        end = (end[0] + left_top_bound[0], end[1] + left_top_bound[1])
    else:
        log_ret("No end found, going for linear prediction",
                "WARNING", shared_logger)
    if mask == None:
        log_ret("No masks found", "ERROR", shared_logger)
        save_image(image, "mask", "error")
        sleep(1)
        return
    if get_config()["advanced"]["saveYOLOImage"]:
        mask_image = image.copy()
        mask_ = mask.cpu().numpy()
        mask_ = cv2.resize(mask_, (mask_image.shape[1], mask_image.shape[0]))
        mask_colored = np.stack(
            [mask_ * 255, mask_ * 0, mask_ * 0], axis=-1).astype(np.uint8)
        overlay = cv2.addWeighted(mask_image, 0.7, mask_colored, 0.3, 0)
        save_image(overlay, "seg", "yolo")
    log_ret("Using yolo to bypass AFK", "EVENT", shared_logger)
    line_ = segment_path(mask, start, end, left_top_bound)
    line = [line_[i]
            for i in range(0, len(line_), get_config()["advanced"]["optimizeQuantization"])]
    line = rdp(line, get_config()["advanced"]["rdpEpsilon"])
    try:
        line = extend_line(line)
    except:
        pass
    final_np = np.array(line, np.int32)
    final_np = final_np.reshape((-1, 1, 2))
    if start != None:
        cv2.circle(ori_image, start, 5, (0, 255, 0), -1)
    if end != None:
        cv2.circle(ori_image, end, 5, (0, 0, 255), -1)
    cv2.polylines(ori_image, [final_np], False, (0, 255, 0), 2)
    cv2.rectangle(ori_image, left_top_bound,
                  right_bottom_bound, (0, 0, 255), 2)
    for point in line:
        cv2.circle(ori_image, point, 3, (255, 0, 0), -1)
    save_image(ori_image, "afk_solution", "afk")
    if get_config()["advanced"]["showLogger"]:
        cv2.imshow("image", ori_image)
        cv2.waitKey(0)
    if get_config()["advanced"]["useOBS"]:
        obs("start")
        sleep(1)
    if type == "fullscreen":
        debugger("Executing AFK with fullscreen")
        if get_config()["advanced"]["moveMouse"]:
            with suppress_idle_detection.get_lock():
                suppress_idle_detection.value = True
            ori_pos = pyautogui.position()
            apply_mouse_movement(line, active=True)
            pyautogui.moveTo(ori_pos[0], ori_pos[1], duration=0.1)
            with suppress_idle_detection.get_lock():
                suppress_idle_detection.value = False
        if get_config()["runs"]["moveAfterAFK"]:
            move_a_bit()
    else:
        debugger("Executing AFK with spec window", hwnd)
        ori_pos = pyautogui.position()
        now_windows: list[gw.Win32Window] = gw.getWindowsWithTitle("")
        for w in now_windows:
            if w._hWnd == hwnd:
                debugger("Activating window", hwnd, w.title)
                w.activate()
                break
        if get_config()["advanced"]["moveMouse"]:
            with suppress_idle_detection.get_lock():
                suppress_idle_detection.value = True
            apply_mouse_movement(calculate_offset(line, hwnd), active=False)
            pyautogui.moveTo(ori_pos[0], ori_pos[1], duration=0.1)
            with suppress_idle_detection.get_lock():
                suppress_idle_detection.value = False
        if get_config()["runs"]["moveAfterAFK"]:
            move_a_bit()
        sleep(1)
        pyautogui.hotkey('alt', 'tab')
    if get_config()["executeBinary"]["runAfterAFK"] != "":
        try:
            system("start "+get_config()["executeBinary"]["runAfterAFK"])
        except:
            log_ret("Cannot execute extra binary", "ERROR", shared_logger)
    if get_config()["advanced"]["useOBS"]:
        sleep(1)
        obs("stop")
    debugger("`execute_afk` finished")


def run_segment(idled_flag, suppress_idle_detection, shared_logger, capture_windows):
    global segment_process, idle_thread, afk_thread_process

    idle_thread = multiprocessing.Process(
        target=test_idle_thread, args=(idled_flag, suppress_idle_detection, shared_logger))
    afk_thread_process = multiprocessing.Process(
        target=afk_thread, args=(idled_flag, suppress_idle_detection, shared_logger, capture_windows))

    if get_config()["runs"]["autoTakeOverWhenIdle"]:
        idle_thread.start()
    afk_thread_process.start()

    if get_config()["runs"]["autoTakeOverWhenIdle"]:
        idle_thread.join()
    afk_thread_process.join()


def start_segment_process(segment_running, shared_logger, capture_windows):
    global segment_process, idled_flag, suppress_idle_detection, idle_thread, afk_thread_process
    debugger("Starting segment process")
    check_config(shared_logger)
    segment_running.value = True
    if segment_process is not None and segment_process.is_alive():
        log_ret("Segment process is already running.",
                "WARNING", shared_logger)
        return

    idled_flag = multiprocessing.Value('b', False)
    suppress_idle_detection = multiprocessing.Value('b', False)

    if not get_config()["runs"]["autoTakeOverWhenIdle"]:
        with idled_flag.get_lock():
            idled_flag.value = True

    segment_process = multiprocessing.Process(
        target=run_segment, args=(idled_flag, suppress_idle_detection, shared_logger, capture_windows))
    segment_process.start()
    log_ret("Segment process started", "INFO", shared_logger)


def toggle_segment_process(capture_windows):
    global segment_process, segment_running, shared_logger, idle_thread, afk_thread_process
    debugger("Toggling segment process")
    if segment_running.value:
        segment_running.value = False

        if idle_thread is not None and idle_thread.is_alive():
            idle_thread.terminate()
            idle_thread.join()
            idle_thread = None

        if afk_thread_process is not None and afk_thread_process.is_alive():
            afk_thread_process.terminate()
            afk_thread_process.join()
            afk_thread_process = None

        if segment_process is not None and segment_process.is_alive():
            segment_process.terminate()
            segment_process.join()
        segment_process = None

        log_ret("Segment process and associated threads terminated.",
                "INFO", shared_logger)
        update_page("launch")
    else:
        start_segment_process(segment_running, shared_logger, capture_windows)
        log_ret("Segment process started.", "INFO", shared_logger)
        update_page("console")


def update_page(new_page_stat):
    global page_stat, console_text, capture_windows
    debugger(f"Updating page to {new_page_stat}")
    theme = detect_theme() if get_config(
    )["gui"]["theme"] == "auto" else ("Dark" if get_config()["gui"]["theme"].lower() == "dark" else "Light")
    page_stat = new_page_stat
    for widget in main_content.winfo_children():
        widget.destroy()
    if page_stat == "launch":
        canvas = add_rounded_image_to_canvas(
            main_content, f"./gui/backgrounds/{choice(listdir('./gui/backgrounds'))}", theme)
        announcement_title = ttk.Label(
            main_content,
            text=f"{get_ui_translation('launch_announcements')}:",
            font=("Microsoft Yahei", 15),
        )
        announcement_label = ttk.Label(
            main_content,
            text=announcement,
            font=("Microsoft Yahei", 12),
        )
        canvas.pack(anchor="center", pady=20)
        announcement_title.pack(anchor="w", pady=0)
        announcement_label.pack(anchor="w", pady=0)
        launch_button = ttk.Button(
            main_content,
            text=get_ui_translation(
                "launch_run") if not segment_running.value else "Terminate",
            width=30,
            style="Large.TButton",
            command=lambda: toggle_segment_process([{"title": w["title"],
                                                     "hwnd": w["hwnd"], "capture_method": w["capture_method"]} for w in capture_windows])
        )
        launch_button.pack(side="bottom", anchor="se", padx=20, pady=20)
    elif page_stat == "console":
        if theme == "Dark":
            console_bg = "#1c1c1c"
            console_fg = "#ffffff"
        else:
            console_bg = "#fafafa"
            console_fg = "#1f2328"
        console_frame = ttk.Frame(main_content)
        console_frame.pack(fill="both", expand=True, padx=10, pady=10)
        scrollbar = ttk.Scrollbar(console_frame, orient="vertical")
        scrollbar.pack(side="right", fill="y")
        console_text = tk.Text(console_frame, wrap="word",
                               bg=console_bg, fg=console_fg, font=(
                                   "Consolas", 10),
                               bd=0, highlightthickness=0, yscrollcommand=scrollbar.set)
        console_text.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=console_text.yview)
        console_text.config(state="disabled")
        console_text.tag_config("info", foreground="#2ad058")
        console_text.tag_config("error", foreground="#ff0000")
        console_text.tag_config("warning", foreground="#808000")
        console_text.tag_config("critical", foreground="#BA3537")
        console_text.tag_config("event", foreground="#ffab70")
        console_text.tag_config("default", foreground=console_fg)

        def refresh_logger():
            if not console_text.winfo_exists():
                return
            console_text.config(state="normal")
            console_text.delete(1.0, tk.END)
            for log_entry in shared_logger:
                console_text.insert(
                    tk.END, log_entry["logger"] + "\n", log_entry["type"])
            console_text.config(state="disabled")
            console_text.see(tk.END)
            main_content.after(1000, refresh_logger)

        clear_console_button = ttk.Button(
            main_content,
            text=get_ui_translation("console_clear"),
            command=lambda: clear_console()
        )
        clear_console_button.pack(side="bottom", anchor="se", padx=10, pady=5)

        def clear_console():
            shared_logger[:] = []
            console_text.config(state="normal")
            console_text.delete(1.0, tk.END)
            console_text.config(state="disabled")

        refresh_logger()
    elif page_stat == "settings":
        scrollable_frame = create_scrollable_frame(main_content)
        create_settings_widgets(scrollable_frame, get_config())
    elif page_stat == "history":
        scrollable_frame = ttk.Frame(main_content)
        canvas = tk.Canvas(scrollable_frame)
        scrollbar = ttk.Scrollbar(
            scrollable_frame, orient="vertical", command=canvas.yview)
        scrollable_content = ttk.Frame(canvas)
        scrollable_frame.pack(fill="both", expand=True)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        canvas.create_window((0, 0), window=scrollable_content, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        def on_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        scrollable_content.bind("<Configure>", on_configure)

        def _on_mouse_wheel(event):
            canvas.yview_scroll(-1 * int(event.delta / 120), "units")

        canvas.bind("<Enter>", lambda e: canvas.bind_all(
            "<MouseWheel>", _on_mouse_wheel))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))
        if not path.exists("./images"):
            mkdir("./images")
        if not path.exists("./images/afk"):
            mkdir("./images/afk")
        images = listdir("./images/afk")
        names = [name.removeprefix("afk_solution_").removesuffix(".png") for name in images if name.startswith(
            "afk_solution_") and name.endswith(".png")]
        for name in sorted(names):
            name_label = ttk.Label(
                scrollable_content, text=name, font=("Arial", 14, "bold"))
            name_label.pack(anchor="w", pady=5)
            frame = ttk.Frame(scrollable_content)
            frame.pack(fill="x", pady=5)
            solution_path = f"./images/afk/afk_solution_{name}.png"
            if f"afk_solution_{name}.png" in images:
                solution_image = Image.open(solution_path)
                solution_image.thumbnail((350, 350))
                solution_image_tk = ImageTk.PhotoImage(solution_image)
                solution_label = ttk.Label(
                    frame, image=solution_image_tk, text=f"afk_solution_{name}.png", compound="top")
                solution_label.image = solution_image_tk
                solution_label.pack(side="left", padx=5)
            exposure_path = f"./images/afk/exposure_{name}.png"
            if f"exposure_{name}.png" in images:
                exposure_image = Image.open(exposure_path)
                exposure_image.thumbnail((200, 200))
                exposure_image_tk = ImageTk.PhotoImage(exposure_image)
                exposure_label = ttk.Label(
                    frame, image=exposure_image_tk, text=f"exposure_{name}.png", compound="top")
                exposure_label.image = exposure_image_tk
                exposure_label.pack(side="left", padx=5)
    elif page_stat == "hooks":
        button_frame = ttk.Frame(main_content)
        button_frame.pack(fill="x", pady=5)
        plus_button = ttk.Button(
            button_frame, text=get_ui_translation("hooks_capture"), command=lambda: open_capture_window(root, main_content))
        plus_button.pack(side="right", padx=10, pady=10)
        for window in capture_windows:
            update_capture_menu(main_content, window)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    debugger("Main thread started")
    page_stat = "launch"
    segment_process = idle_thread = afk_thread_process = None
    segment_running = multiprocessing.Value(ctypes.c_bool, False)
    announcement = generate_announcement(
        get_config()["advanced"]["skipUpdate"])
    if not get_config()["advanced"]["skipUpdate"]:
        update_models()
    else:
        log("Skipping model update", "WARNING")
    with multiprocessing.Manager() as manager:
        shared_logger = manager.list()
        root = tk.Tk()
        root.title(f"florr-auto-afk (v{constants.VERSION_INFO})")
        root.iconbitmap("./gui/icon.ico")
        root.geometry("1000x600")
        root.minsize(1000, 600)
        sidebar = ttk.Frame(root, width=150)
        sidebar.pack(side="left", fill="y", padx=10)
        launch_button = ttk.Button(sidebar, text=get_ui_translation("page_launch"),
                                   width=20, command=lambda: update_page("launch"))
        launch_button.pack(pady=10)
        console_button = ttk.Button(sidebar, text=get_ui_translation("page_console"),
                                    width=20, command=lambda: update_page("console"))
        console_button.pack(pady=10)
        hook_button = ttk.Button(
            sidebar, text=get_ui_translation("page_hooks"), width=20, command=lambda: update_page("hooks"))
        hook_button.pack(pady=10)
        history_button = ttk.Button(
            sidebar, text=get_ui_translation("page_history"), width=20, command=lambda: update_page("history"))
        history_button.pack(pady=10)
        settings_button = ttk.Button(
            sidebar, text=get_ui_translation("page_settings"), width=20, command=lambda: update_page("settings"))
        settings_button.pack(side="bottom", pady=10)
        main_content = ttk.Frame(root)
        main_content.pack(side="left", fill="both", expand=True)

        update_page(page_stat)
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
        ScaleFactor = ctypes.windll.shcore.GetScaleFactorForDevice(0)
        root.tk.call('tk', 'scaling', ScaleFactor/75)
        sv_ttk.set_theme(theme)
        apply_theme_to_titlebar(root)

        root.mainloop()

        if segment_process is not None and segment_process.is_alive():
            segment_process.terminate()
