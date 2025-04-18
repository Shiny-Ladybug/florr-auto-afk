from multiprocessing import Process, Value, Manager
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


def afk_thread(idled_flag, suppress_idle_detection, afk_det_model, afk_seg_model, shared_logger):
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
    while True:
        with idled_flag.get_lock():
            if not idled_flag.value:
                sleep(1)
                continue
        image = pyautogui.screenshot()
        ori_image = image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        position = detect_afk(image, afk_det_model)
        if position == None:
            if get_config()["advanced"]["verbose"]:
                log_ret("No AFK window found", "EVENT",
                        shared_logger, save=False)
            if countdown != -1 and time() > eta_timestamp:
                log_ret("Countdown Ends, program exiting", "EVENT")
                break
            sleep(get_config()["advanced"]["epochInterval"])
            continue
        log_ret("Found AFK window", "EVENT", shared_logger)
        if get_config()["executeBinary"]["runBeforeAFK"] != "":
            try:
                system("start "+get_config()["executeBinary"]["runBeforeAFK"])
            except:
                log_ret("Cannot execute extra binary", "ERROR")
        save_image(image, "afk", "afk")
        left_top_bound = position[2][0]
        right_bottom_bound = position[2][1]
        if get_config()["exposure"]["enable"]:
            image = exposure_image(
                left_top_bound, right_bottom_bound, get_config()["exposure"]["duration"])
            save_image(image, "exposure", "afk")
        else:
            image = crop_image(left_top_bound, right_bottom_bound, ori_image)
        results = afk_seg_model.predict(
            image, retina_masks=True, verbose=False)
        masks = results[0].masks
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
        if masks == None:
            log_ret("No masks found", "ERROR", shared_logger)
            save_image(image, "mask", "error")
            sleep(1)
            continue
        log_ret("Using yolo to bypass AFK", "EVENT", shared_logger)
        line_ = segment_path(masks, start, end, left_top_bound)
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
        if get_config()["advanced"]["moveMouse"]:
            with suppress_idle_detection.get_lock():
                suppress_idle_detection.value = True
            ori_pos = pyautogui.position()
            apply_mouse_movement(line)
            pyautogui.moveTo(ori_pos[0], ori_pos[1], duration=0.1)
            with suppress_idle_detection.get_lock():
                suppress_idle_detection.value = False
        if get_config()["runs"]["moveAfterAFK"]:
            move_a_bit()
        if get_config()["executeBinary"]["runAfterAFK"] != "":
            try:
                system("start "+get_config()["executeBinary"]["runAfterAFK"])
            except:
                log_ret("Cannot execute extra binary", "ERROR", shared_logger)
        if get_config()["advanced"]["useOBS"]:
            sleep(1)
            obs("stop")
        sleep(1)
        position = detect_afk(cv2.cvtColor(
            np.array(pyautogui.screenshot()), cv2.COLOR_RGB2BGR), afk_det_model)
        if position != None:
            log_ret("Cannot bypass AFK", "ERROR", shared_logger)
        else:
            log_ret("Bypassed AFK", "EVENT", shared_logger)


def run_segment(idled_flag, suppress_idle_detection, shared_logger):
    global idle_thread, afk_thread_process
    try:
        afk_seg_model = YOLO(get_config()["yoloConfig"]["segModel"])
        afk_det_model = YOLO(get_config()["yoloConfig"]["detModel"])
    except:
        log_ret("YOLO models are corrupted, trying to restore files",
                "ERROR", shared_logger)
        remove(get_config()["yoloConfig"]["segModel"])
        remove(get_config()["yoloConfig"]["detModel"])
        remove("./models/version")
        return

    idle_thread = multiprocessing.Process(
        target=test_idle_thread, args=(idled_flag, suppress_idle_detection, shared_logger))
    afk_thread_process = multiprocessing.Process(
        target=afk_thread, args=(idled_flag, suppress_idle_detection, afk_det_model, afk_seg_model, shared_logger))

    if get_config()["runs"]["autoTakeOverWhenIdle"]:
        idle_thread.start()
    afk_thread_process.start()

    if get_config()["runs"]["autoTakeOverWhenIdle"]:
        idle_thread.join()
    afk_thread_process.join()


def start_segment_process(segment_running, shared_logger):
    global segment_process, idled_flag, suppress_idle_detection, idle_thread, afk_thread_process
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
    else:
        log_ret("Idle Detection is currently enabled, set `autoTakeOverWhenIdle` to `false` if you just want to test the AFK Bypass ability",
                "WARNING", shared_logger, save=False)
    segment_process = multiprocessing.Process(
        target=run_segment, args=(idled_flag, suppress_idle_detection, shared_logger))
    segment_process.start()
    log_ret("Segment process started", "INFO", shared_logger)


def toggle_segment_process():
    global segment_process, segment_running, shared_logger, idle_thread, afk_thread_process

    if segment_running.value:
        segment_running.value = False
        if segment_process is not None:
            segment_process.terminate()
            segment_process.join()
        segment_process = None
        log_ret("Segment process terminated.", "INFO", shared_logger)
        update_page("launch")
    else:
        start_segment_process(segment_running, shared_logger)
        log_ret("Segment process started.", "INFO", shared_logger)
        update_page("console")


def update_page(new_page_stat):
    global page_stat, console_text
    theme = darkdetect.theme() if get_config(
    )["gui"]["theme"] == "auto" else ("Dark" if get_config()["gui"]["theme"].lower() == "dark" else "Light")
    page_stat = new_page_stat
    for widget in main_content.winfo_children():
        widget.destroy()
    if page_stat == "launch":
        canvas = add_rounded_image_to_canvas(
            main_content, "./gui/bg.png", theme)
        announcement_label = ttk.Label(
            main_content,
            text=announcement,
            font=("Microsoft Yahei", 15),
        )
        canvas.pack(anchor="center", pady=20)
        announcement_label.pack(anchor="w", pady=0)
        launch_button = ttk.Button(
            main_content,
            text="Run" if not segment_running.value else "Terminate",
            width=30,
            style="Large.TButton",
            command=toggle_segment_process
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
            text="Clear Console",
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


if __name__ == "__main__":
    multiprocessing.freeze_support()
    page_stat = "launch"
    segment_process = idle_thread = afk_thread_process = None
    segment_running = Value(ctypes.c_bool, False)
    announcement = generate_announcement()
    update_models()
    with Manager() as manager:
        shared_logger = manager.list()
        root = tk.Tk()
        root.title(f"florr-auto-afk (v{constants.VERSION_INFO})")
        root.iconbitmap("./gui/icon.ico")
        root.geometry("1000x600")
        # if get_config()["gui"]["mica"] and theme == "Dark":
        #     root.after(1, Get_hWnd, root)
        root.resizable(False, False)
        sidebar = ttk.Frame(root, width=150)
        sidebar.pack(side="left", fill="y", padx=10)
        launch_button = ttk.Button(sidebar, text="Launch",
                                   width=20, command=lambda: update_page("launch"))
        launch_button.pack(pady=10)
        console_button = ttk.Button(sidebar, text="Console",
                                    width=20, command=lambda: update_page("console"))
        console_button.pack(pady=10)
        settings_button = ttk.Button(
            sidebar, text="Settings", width=20, command=lambda: update_page("settings"))
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
