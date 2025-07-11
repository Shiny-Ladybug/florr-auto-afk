from gui_utils import *
import multiprocessing
import experimental
import server


def test_idle_thread(idled_flag, suppress_idle_detection, shared_logger, not_stop_event):
    last_pos = None
    idle_iter = 0
    sleep_time = get_config()["runs"]["idleDetInterval"]
    while not not_stop_event.is_set():
        with suppress_idle_detection.get_lock():
            if suppress_idle_detection.value:
                sleep(1)
                continue
        pos = pyautogui.position()
        if last_pos != pos:
            idle_iter = 0
            sleep_time = get_config()["runs"]["idleDetInterval"]
            if idled_flag.value:
                with idled_flag.get_lock():
                    idled_flag.value = False
                    log_ret("No longer idle", "EVENT",
                            shared_logger, save=False)
            last_pos = pos
        else:
            idle_iter += 1
        if idle_iter > (get_config()["runs"]["idleTimeThreshold"]//get_config()["runs"]["idleDetInterval"]):
            if not idled_flag.value:
                with idled_flag.get_lock():
                    idled_flag.value = True
                    log_ret("Idle detected", "EVENT",
                            shared_logger, save=False)
            sleep_time = get_config()["runs"]["idleDetIntervalMax"]
        sleep(sleep_time)


def afk_thread(idled_flag, suppress_idle_detection, shared_logger, capture_windows, stop_run_event):
    try:
        debugger("Loading YOLO models")
        afk_seg_model = YOLO(get_config()["yoloConfig"]["segModel"])
        afk_det_model = YOLO(get_config()["yoloConfig"]["detModel"])
        debugger("Loaded YOLO models")
    except:
        log("YOLO models are corrupted, trying to restore files", "ERROR")
        remove(get_config()["yoloConfig"]["segModel"])
        remove(get_config()["yoloConfig"]["detModel"])
        remove("./models/version")
        return
    debugger("Loaded configs", get_config())
    test_environment(afk_seg_model, afk_det_model, shared_logger)
    log_ret("恭喜你，你成功把代码跑起来了，你是这个👍", "EVENT", shared_logger, save=False)
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
    if not capture_windows:
        log_ret("Fullscreen detection enabled", "INFO", shared_logger)
    else:
        log_ret(
            f"Window detection for {[w['title'] for w in capture_windows]} enabled", "INFO", shared_logger)
    while not stop_run_event.is_set():
        start_time = time()
        with idled_flag.get_lock():
            if not idled_flag.value:
                sleep(1)
                continue
        if not capture_windows:
            image = pyautogui.screenshot()
            image = cv2.cvtColor(
                np.array(image), cv2.COLOR_RGB2BGR)
            ori_image = image.copy()

            if get_config()["runs"]["rejoin"]:
                ready_button = locate_ready(image)
                if ready_button is not None:
                    multiprocessing.Process(
                        target=send_notification, args=("AFK Detection Failed",)).start()
                    multiprocessing.Process(
                        target=send_sound_notification, args=(3,)).start()
                    with suppress_idle_detection.get_lock():
                        suppress_idle_detection.value = True
                    pyautogui.click(
                        ready_button[0], ready_button[1], button='left')
                    with suppress_idle_detection.get_lock():
                        suppress_idle_detection.value = False
                    log_ret("AFK detection failed", "ERROR", shared_logger)
                    debugger(
                        "AFK detection failed, ready button found", ready_button)
                    sleep(1)
                    continue

            if (get_config()["extensions"]["enable"] and experimental.get_connected()) and not get_config()["advanced"]["forceYOLO"]["det"]:
                if not experimental.get_aspac():
                    debugger("No AFK window found [EXP]")
                    if experimental.get_block_alpha():
                        experimental.switch_block_alpha(False)
                    if get_config()["advanced"]["verbose"]:
                        log("No AFK window found [EXP]", "EVENT", save=False)
                    if countdown != -1 and time() > eta_timestamp:
                        log_ret("Countdown Ends, program exiting", "EVENT")
                        break
                    sleep(
                        max(0, get_config()["advanced"]["epochInterval"] - (time() - start_time)))
                    continue

            afk_window_pos = detect_afk_window(image, afk_det_model)
            if afk_window_pos is None:
                debugger("No AFK window found")
                experimental.set_aspac(False)
                if experimental.get_block_alpha():
                    experimental.switch_block_alpha(False)
                if get_config()["advanced"]["verbose"]:
                    log("No AFK window found", "EVENT", save=False)
                if countdown != -1 and time() > eta_timestamp:
                    log_ret("Countdown Ends, program exiting", "EVENT")
                    break
                sleep(
                    max(0, get_config()["advanced"]["epochInterval"] - (time() - start_time)))
                continue
            log_ret("Found AFK window", "EVENT", shared_logger)
            debugger("Found AFK window")

            if (get_config()["extensions"]["enable"] and experimental.get_connected()):
                if get_config()["extensions"]["bgRemove"]:
                    log_ret("Blocking alpha channel", "EVENT", shared_logger)
                    experimental.switch_block_alpha(True)
                if get_config()["extensions"]["uploadPath"]:
                    experimental.switch_send(True)
                sleep(get_config()["extensions"]["swapInterval"])

            multiprocessing.Process(
                target=send_notification, args=("AFK Detected",)).start()
            multiprocessing.Process(
                target=send_sound_notification, args=(3,)).start()

            save_image(image, "afk", "afk")
            left_top_bound, right_bottom_bound = afk_window_pos
            debugger("Bounds", afk_window_pos)
            if get_config()["extensions"]["enable"] and get_config()["extensions"]["bgRemove"]:
                image = cv2.cvtColor(
                    np.array(pyautogui.screenshot()), cv2.COLOR_RGB2BGR)
                save_image(image, "full", "extension")
                image = crop_image(
                    left_top_bound, right_bottom_bound, image)
                save_image(image, "extension", "afk")
            elif get_config()["exposure"]["enable"]:
                image = exposure_image(
                    left_top_bound, right_bottom_bound, get_config()["exposure"]["duration"])
                save_image(image, "exposure", "afk")
            else:
                image = crop_image(
                    left_top_bound, right_bottom_bound, ori_image)
            execute_afk(afk_window_pos, ori_image, image, afk_seg_model, afk_det_model,
                        suppress_idle_detection, shared_logger, type="fullscreen")
        else:
            results = []
            for window in capture_windows:
                if window["capture_method"] == "BitBlt":
                    image = bitblt.bitblt_capture(window["hwnd"])
                elif window["capture_method"] == "GDI":
                    image = gdi.gdi_capture(window["hwnd"])
                elif window["capture_method"] == "WGC":
                    image = wgc.wgc_capture(window["hwnd"])
                image = cv2.cvtColor(
                    np.array(image), cv2.COLOR_RGBA2RGB)
                ori_image = image.copy()

                if get_config()["runs"]["rejoin"]:
                    ready_button = locate_ready(image)
                    if ready_button is not None:
                        multiprocessing.Process(
                            target=send_notification, args=("AFK Detection Failed",)).start()
                        multiprocessing.Process(
                            target=send_sound_notification, args=(3,)).start()

                        now_windows: list[gw.Win32Window] = gw.getWindowsWithTitle(
                            "")
                        for w in now_windows:
                            if w._hWnd == window['hwnd']:
                                debugger("Activating window",
                                         window['hwnd'], w.title)
                                w.activate()
                                break
                        with suppress_idle_detection.get_lock():
                            suppress_idle_detection.value = True
                        pyautogui.click(
                            ready_button[0], ready_button[1], button='left')
                        with suppress_idle_detection.get_lock():
                            suppress_idle_detection.value = False
                        pyautogui.hotkey('alt', 'tab')

                        log_ret("AFK detection failed", "EVENT", shared_logger)
                        debugger(
                            "AFK detection failed, ready button found", ready_button)
                        sleep(1)
                        continue

                afk_window_pos = detect_afk_window(image, afk_det_model)
                if afk_window_pos is None:
                    results.append(False)
                    continue
                log_ret(
                    f"Found AFK window in {window['title']}", "EVENT", shared_logger)

                if get_config()["extensions"]["enable"]:
                    if get_config()["extensions"]["bgRemove"]:
                        log_ret("Blocking alpha channel",
                                "EVENT", shared_logger)
                        experimental.switch_block_alpha(True)
                    if get_config()["extensions"]["uploadPath"]:
                        experimental.switch_send(True)
                    sleep(get_config()["extensions"]["swapInterval"])

                multiprocessing.Process(
                    target=send_notification, args=("AFK Detected",)).start()
                multiprocessing.Process(
                    target=send_sound_notification, args=(3,)).start()

                save_image(image, "afk", "afk")
                debugger("Bounds", afk_window_pos)
                left_top_bound, right_bottom_bound = afk_window_pos
                if get_config()["extensions"]["enable"] and get_config()["extensions"]["bgRemove"]:
                    image = cv2.cvtColor(
                        np.array(pyautogui.screenshot()), cv2.COLOR_RGB2BGR)
                    image = crop_image(
                        left_top_bound, right_bottom_bound, image)
                    save_image(image, "extension", "afk")
                elif get_config()["exposure"]["enable"]:
                    image = exposure_image(
                        left_top_bound, right_bottom_bound, get_config()["exposure"]["duration"], window["hwnd"], window["capture_method"])
                    save_image(image, "exposure", "afk")
                else:
                    image = crop_image(
                        left_top_bound, right_bottom_bound, ori_image)
                execute_afk(afk_window_pos, ori_image, image, afk_seg_model, afk_det_model,
                            suppress_idle_detection, shared_logger, type="window", hwnd=window['hwnd'])
                results.append(True)
            if not any(results):
                if get_config()["advanced"]["verbose"]:
                    log("No AFK window found", "EVENT", save=False)
                if countdown != -1 and time() > eta_timestamp:
                    log_ret("Countdown Ends, program exiting", "EVENT")
                    break
                sleep(
                    max(0, get_config()["advanced"]["epochInterval"] - (time() - start_time)))
                continue


def execute_afk(afk_window_pos, ori_image, image, afk_seg_model, afk_det_model, suppress_idle_detection, shared_logger, type="fullscreen", hwnd=None):
    need_yolo = False
    if (get_config()["extensions"]["enable"] and experimental.get_connected()) and not get_config()["advanced"]["forceYOLO"]["seg"]:
        log_ret("Using BW OpenCV", "EVENT", shared_logger)
        afk_bw = AFK_BW(image)
        afk_bw.crop_nb_image().normalize().get_mask()
        start_p, end_p = afk_bw.get_start(), afk_bw.get_end()
        if start_p is None or end_p is None:
            log_ret("No start or end point found, fallback to use YOLO",
                    "WARNING", shared_logger)
            need_yolo = True
        else:
            afk_bw.dijkstra()
            afk_bw.rdp(round(eval(get_config()["advanced"]["rdpEpsilon"].replace(
                "width", str(afk_bw.get_width())))))
            afk_bw.extend(get_config()["advanced"]["extendLength"])
            line = afk_bw.get_final(afk_window_pos[0], precise=False)
            start_p = (start_p[0] + afk_bw.offset[0],
                       start_p[1] + afk_bw.offset[1])
            end_p = (end_p[0] + afk_bw.offset[0],
                     end_p[1] + afk_bw.offset[1])
            line = [(p[0] + afk_bw.offset[0], p[1] + afk_bw.offset[1])
                    for p in line]
            ori_image = draw_annotated_image(
                ori_image, line, start_p, end_p, afk_window_pos, afk_bw.inverse_start_color, afk_bw.get_width(), afk_bw.get_length(), afk_bw.get_difficulty(), "OpenCV Pro")

    if need_yolo or not (get_config()["extensions"]["enable"] and experimental.get_connected()) or get_config()["advanced"]["forceYOLO"]["seg"]:
        res = get_masks_by_iou(image, afk_seg_model)
        start_p, end_p, start_size, pack = detect_afk_things(
            image, afk_det_model, caller="main")
        if res is None:
            log_ret("No masks found", "ERROR", shared_logger)
            save_image(image, "mask", "error")
            experimental.switch_block_alpha(False)
            return

        position = start_p, end_p, afk_window_pos, start_size
        mask, results = res

        results_ = deepcopy(results)
        image_ = deepcopy(image)
        if start_p is None:
            log_ret("No start point found, going for AUTO prediction",
                    "WARNING", shared_logger)
        if end_p is None:
            log_ret("No end found, going for LINEAR prediction",
                    "WARNING", shared_logger)

        afk_mask = AFK_Segment(image, mask, start_p, end_p, start_size)
        if start_p is not None:
            afk_mask.save_start()

        if get_config()["advanced"]["saveYOLOImage"]:
            mask_image = image.copy()
            mask_ = mask.cpu().numpy()
            mask_ = cv2.resize(
                mask_, (mask_image.shape[1], mask_image.shape[0]))
            mask_colored = np.stack(
                [mask_ * 255, mask_ * 0, mask_ * 0], axis=-1).astype(np.uint8)
            overlay = cv2.addWeighted(mask_image, 0.7, mask_colored, 0.3, 0)
            save_image(overlay, "seg", "yolo")

        log_ret("Using yolo to bypass AFK", "EVENT", shared_logger)

        afk_path = AFK_Path(afk_mask.segment_path(), start_p,
                            end_p, afk_mask.get_width())
        dijkstra_stat = afk_path.dijkstra(afk_mask.mask)
        if not dijkstra_stat:
            afk_path.sort()
        afk_path.rdp(round(eval(get_config()["advanced"]["rdpEpsilon"].replace(
            "width", str(afk_mask.get_width())))))
        afk_path.extend(get_config()["advanced"]["extendLength"])
        line = afk_path.get_final(afk_window_pos[0], precise=False)

        ori_image = draw_annotated_image(
            ori_image, line, start_p, end_p, afk_window_pos, afk_mask.inverse_start_color, afk_mask.get_width(), afk_path.get_length(), afk_path.get_difficulty(), afk_path.sort_method)
        packs = [
            [(0, 0), (image.shape[1], image.shape[0])], pack[0], pack[1]]
        threading_save(image_, results_, packs, afk_path.get_difficulty())

    save_image(ori_image, "afk_solution", "afk")

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
            apply_mouse_movement(
                calculate_offset(line, hwnd), active=False)
            pyautogui.moveTo(ori_pos[0], ori_pos[1], duration=0.1)
            with suppress_idle_detection.get_lock():
                suppress_idle_detection.value = False
        if get_config()["runs"]["moveAfterAFK"]:
            move_a_bit()
        sleep(1)
        pyautogui.hotkey('alt', 'tab')

    if get_config()["advanced"]["useOBS"]:
        sleep(1)
        obs("stop")
    debugger("`execute_afk` finished")

    if (get_config()["extensions"]["enable"] and experimental.get_connected()):
        if get_config()["extensions"]["bgRemove"]:
            log_ret("Restoring background", "EVENT", shared_logger)
            experimental.switch_block_alpha(False)
        experimental.set_aspac(False)


def run_segment(idled_flag, suppress_idle_detection, shared_logger, capture_windows, stop_segment_event):
    global segment_process, idle_thread, afk_thread_process
    debugger("`run_segment` started")
    stop_idle_event = multiprocessing.Event()
    stop_afk_event = multiprocessing.Event()
    if get_config()["runs"]["autoTakeOverWhenIdle"]:
        idle_thread = multiprocessing.Process(
            target=test_idle_thread, args=(idled_flag, suppress_idle_detection, shared_logger, stop_idle_event))
    debugger("calling `afk_thread`")
    afk_thread_process = multiprocessing.Process(
        target=afk_thread, args=(idled_flag, suppress_idle_detection, shared_logger, capture_windows, stop_afk_event))
    if get_config()["runs"]["autoTakeOverWhenIdle"]:
        idle_thread.start()
    afk_thread_process.start()
    stop_segment_event.wait()
    stop_idle_event.set()
    stop_afk_event.set()
    if get_config()["runs"]["autoTakeOverWhenIdle"]:
        idle_thread.join()
    afk_thread_process.join()


def start_segment_process(segment_running, shared_logger, capture_windows):
    global segment_process, idled_flag, suppress_idle_detection, idle_thread, stop_event
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
    log_ret("Segment process started", "INFO", shared_logger)
    debugger("Calling `run_segment`")
    segment_process = multiprocessing.Process(
        target=run_segment, args=(idled_flag, suppress_idle_detection, shared_logger, capture_windows, stop_event))
    segment_process.start()


def toggle_segment_process(capture_windows):
    global segment_process, segment_running, shared_logger, stop_event
    debugger("Toggling segment process")
    if segment_running.value:
        segment_running.value = False
        stop_event.set()
        if segment_process is not None and segment_process.is_alive():
            segment_process.terminate()
            segment_process.join()
        segment_process = None
        stop_event.clear()
        log_ret("Segment process and associated threads terminated.",
                "INFO", shared_logger)
        update_page("launch")
    else:
        start_segment_process(segment_running, shared_logger, capture_windows)
        update_page("console")


def start_test(frame, file_path: str):
    test_time = int(time())
    if not file_path or file_path == "\n":
        show_label(frame, "You haven't chosen a file yet", "red")
        return
    try:
        if not (file_path.lower().endswith(".png") or file_path.lower().endswith(".jpg") or file_path.lower().endswith(".jpeg")):
            raise FileNotFoundError
        image = cv2.imread(file_path)
    except FileNotFoundError:
        show_label(
            frame, "Cannot find specified path or is not PNG or JPG type", "red")
        return
    try:
        debugger("Testing: Loading YOLO models")
        afk_seg_model = YOLO(get_config()["yoloConfig"]["segModel"])
        afk_det_model = YOLO(get_config()["yoloConfig"]["detModel"])
        debugger("Testing: Loaded YOLO models")
    except:
        remove(get_config()["yoloConfig"]["segModel"])
        remove(get_config()["yoloConfig"]["detModel"])
        remove("./models/version")
        show_label(frame, "Failed to load YOLO models", "red")
        return

    ori_image = image.copy()
    afk_window_pos = detect_afk_window(image, afk_det_model)
    if afk_window_pos is None:
        debugger("Test result: No AFK window found")
        show_label(frame, "Test result: No AFK window found")
        return
    debugger("Test result: Found AFK window")
    show_label(
        frame, f"Test result: Found AFK window, results saved to ./test/{test_time}")
    cropped_image = crop_image(
        afk_window_pos[0], afk_window_pos[1], image)
    start_p, end_p, start_size, pack = detect_afk_things(
        cropped_image, afk_det_model, caller="test", test_time=test_time)
    position = start_p, end_p, afk_window_pos, start_size
    res = get_masks_by_iou(cropped_image, afk_seg_model)
    if res is None:
        save_test_image(image, "mask_err", test_time)
        return
    mask, results = res

    afk_mask = AFK_Segment(image, mask, start_p, end_p, start_size)
    if start_p is not None:
        afk_mask.save_start()

    if get_config()["advanced"]["saveYOLOImage"]:
        mask_image = cropped_image.copy()
        mask_ = mask.cpu().numpy()
        mask_ = cv2.resize(mask_, (mask_image.shape[1], mask_image.shape[0]))
        mask_colored = np.stack(
            [mask_ * 255, mask_ * 0, mask_ * 0], axis=-1).astype(np.uint8)
        overlay = cv2.addWeighted(mask_image, 0.7, mask_colored, 0.3, 0)
        save_test_image(overlay, "yolo_seg", test_time)

    afk_path = AFK_Path(afk_mask.segment_path(), start_p,
                        end_p, afk_mask.get_width())
    dijkstra_stat = afk_path.dijkstra(afk_mask.mask)
    if not dijkstra_stat:
        afk_path.sort()
    afk_path.rdp(round(eval(get_config()["advanced"]["rdpEpsilon"].replace(
        "width", str(afk_mask.get_width())))))
    afk_path.extend(get_config()["advanced"]["extendLength"])
    line = afk_path.get_final(afk_window_pos[0], precise=False)

    ori_image = draw_annotated_image(
        ori_image, line, start_p, end_p, afk_window_pos, afk_mask.inverse_start_color, afk_mask.get_width(), afk_path.get_length(), afk_path.get_difficulty(), afk_path.sort_method)

    save_test_image(ori_image, "afk_solution", test_time)


def destroy_label():
    global error_label
    if error_label is not None:
        if error_label.winfo_exists():
            error_label.destroy()


def show_label(frame, text, color="base"):
    global error_label
    destroy_label()
    if color == "base":
        if get_theme() == "Dark":
            color = "#ffffff"
        else:
            color = "#000000"
    error_label = ttk.Label(
        frame, text=text, foreground=color, font=("Microsoft Yahei", 12))
    error_label.pack(side="bottom", anchor="w")


def choose_file(frame):
    global filepath, filepath_label
    if filepath_label is not None:
        if filepath_label.winfo_exists():
            filepath_label.destroy()
    filepath = filedialog.askopenfilename(filetypes=(
        ("Image files", ["*.png", "*.jpg", "*.jpeg"]),))
    filepath_label = ttk.Label(
        frame, text=filepath, font=("Microsoft Yahei", 10))
    filepath_label.pack(side="left", padx=10)
    destroy_label()


def update_page(new_page_stat):
    global page_stat, console_text, capture_windows
    debugger(f"Updating page to {new_page_stat}")
    theme = get_theme()
    page_stat = new_page_stat
    for widget in main_content.winfo_children():
        widget.destroy()
    if page_stat == "launch":
        canvas = add_rounded_image_to_canvas(
            main_content, get_random_background(), theme)
        canvas.pack(anchor="center", pady=20)
        announcement_title = ttk.Label(
            main_content,
            text=f"{get_ui_translation('launch_announcements')}:",
            font=("Microsoft Yahei", 15),
        )
        announcement_title.pack(anchor="w", pady=0)

        announcement_frame = ttk.Frame(main_content)
        announcement_frame.pack(fill="x", pady=(0, 10))
        announcement_scrollbar = ttk.Scrollbar(
            announcement_frame, orient="vertical")
        announcement_scrollbar.pack(side="right", fill="y")
        announcement_text = tk.Text(
            announcement_frame,
            height=9,
            wrap="word",
            font=("Microsoft Yahei", 12),
            yscrollcommand=announcement_scrollbar.set,
            bd=0,
            highlightthickness=0,
            state="normal"
        )
        announcement_text.insert("1.0", announcement)
        announcement_text.config(state="disabled")
        announcement_text.pack(side="left", fill="x", expand=True)
        announcement_scrollbar.config(command=announcement_text.yview)

        button_link_frame = ttk.Frame(main_content)
        button_link_frame.pack(side="bottom", anchor="se",
                               fill="x", padx=10, pady=10)

        launch_button = ttk.Button(
            button_link_frame,
            text=get_ui_translation(
                "launch_run") if not segment_running.value else get_ui_translation("launch_terminate"),
            width=30,
            style="Large.TButton",
            command=lambda: toggle_segment_process([{"title": w["title"],
                                                     "hwnd": w["hwnd"], "capture_method": w["capture_method"]} for w in capture_windows])
        )
        launch_button.config(style="Accent.TButton")
        launch_button.pack(side="right")

        def open_link(event):
            open_url("https://shiny-ladybug.github.io/")

        link_label = ttk.Label(
            button_link_frame,
            text="Pages: https://shiny-ladybug.github.io",
            foreground="#0969da",
            cursor="hand2",
            font=("Microsoft Yahei", 10, "underline")
        )
        link_label.pack(side="left")
        link_label.bind("<Button-1>", open_link)
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
        console_text.tag_config("notice", foreground="#0969da")
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

        button_frame = ttk.Frame(main_content)
        button_frame.pack(side="bottom", anchor="se",
                          padx=10, pady=5, fill="x")
        if segment_running.value:
            terminate_button = ttk.Button(
                button_frame,
                text=get_ui_translation("launch_terminate"),
                width=13,
                command=lambda: toggle_segment_process([{"title": w["title"],
                                                         "hwnd": w["hwnd"], "capture_method": w["capture_method"]} for w in capture_windows])
            )
            terminate_button.config(style="Accent.TButton")
            terminate_button.pack(side="right", padx=(10, 0))
        clear_console_button = ttk.Button(
            button_frame,
            text=get_ui_translation("console_clear"),
            command=lambda: clear_console()
        )
        clear_console_button.pack(side="right")

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
        for name in sorted(names, reverse=True):
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
        plus_button.config(style="Accent.TButton")
        plus_button.pack(side="right", padx=10, pady=10)
        for window in capture_windows:
            update_capture_menu(main_content, window)
    elif page_stat == "test":
        scrollable_frame = create_scrollable_frame(main_content)
        label = ttk.Label(scrollable_frame, text="Choose image to test YOLO model:", font=(
            "Microsoft Yahei", 10))
        button_frame = ttk.Frame(scrollable_frame)
        choose_button = ttk.Button(
            button_frame, width=13, text=get_ui_translation("image_choose"), command=lambda: choose_file(button_frame))
        start_test_button = ttk.Button(
            main_content, width=13, text=get_ui_translation("page_test"), command=lambda: start_test(scrollable_frame, filepath))
        label.pack(anchor="w", pady=5)
        button_frame.pack(anchor="w", pady=5)
        choose_button.pack(side="left", anchor="w")
        start_test_button.config(style="Accent.TButton")
        start_test_button.pack(side="bottom", pady=10)
    elif page_stat == "extension":
        def open_extensions_folder():
            folder = path.abspath("./extensions")
            startfile(folder)

        top_button_frame = ttk.Frame(main_content)
        top_button_frame.pack(fill="x", pady=(10, 0), padx=10)
        show_explorer_btn = ttk.Button(
            top_button_frame,
            text="Show in explorer",
            command=open_extensions_folder
        )
        show_explorer_btn.pack(side="right")
        show_explorer_btn.config(style="Accent.TButton")

        def make_toggle_callback(name: str, var: tk.BooleanVar):
            def _callback():
                on_extension_toggle(name, var.get())
            return _callback
        scrollable_frame = create_scrollable_frame(main_content)
        installed_extensions = get_installed_extensions()
        for ext in installed_extensions:
            ext_frame = ttk.Frame(scrollable_frame)
            ext_frame.pack(fill="x", pady=(5, 0), padx=10)

            left_frame = ttk.Frame(ext_frame)
            left_frame.pack(side="left", fill="x", expand=True)

            name_label = ttk.Label(
                left_frame,
                text=ext['name'],
                font=("Microsoft Yahei", 14, "bold"),
                foreground="#ffffff" if get_theme() == "Dark" else "#000000"
            )
            name_label.pack(anchor="w")

            description_label = ttk.Label(
                left_frame,
                text=f"> {ext['description']}",
                font=("Microsoft Yahei", 10),
                foreground="#B0B0B0" if get_theme() == "Dark" else "#333333"
            )
            description_label.pack(anchor="w", pady=(0, 2))

            meta_label = ttk.Label(
                left_frame,
                text=f"Author: {ext['author']}    Version: {ext['version']}",
                font=("Microsoft Yahei", 10),
                foreground="#B0B0B0" if get_theme() == "Dark" else "#333333"
            )
            meta_label.pack(anchor="w", pady=(0, 2))

            enabled_var = tk.BooleanVar(value=ext['enabled'])
            toggle_btn = ttk.Checkbutton(
                ext_frame,
                text="Enabled",
                variable=enabled_var,
                command=make_toggle_callback(ext['name'], enabled_var)
            )
            toggle_btn.pack(side="right", padx=20)

            separator = ttk.Separator(scrollable_frame, orient="horizontal")
            separator.pack(fill="x", pady=5, padx=10)


def threading_save(image, results, packs, difficulty):
    if get_config()["advanced"]["saveTrainData"]:
        now = int(time())
        seg_label = export_segmentation_to_label(results, image, now)
        det_label = export_detection_to_label(packs, image, now)
        anno_label = export_annotation_to_label(image, difficulty, now)

        if not path.exists("./train"):
            mkdir("./train")
        if not path.exists("./train/images"):
            mkdir("./train/images")
        if not path.exists("./train/split"):
            mkdir("./train/split")
        if not path.exists("./train/detection"):
            mkdir("./train/detection")
        if not path.exists("./train/annotation"):
            mkdir("./train/annotation")
        if not path.exists("./train/detection/classes.txt"):
            classes = ["Window", "Start", "End"]
            with open("./train/detection/classes.txt", "w") as f:
                for cls in classes:
                    f.write(cls + "\n")
        cv2.imwrite(f"./train/images/{now}.png", image)
        with open(f"./train/split/{now}.json", "w") as f:
            dump(seg_label, f, indent=4)
        with open(f"./train/detection/{now}.txt", "w") as f:
            for item in det_label:
                f.write(
                    f"{item['class']} "
                    f"{item['position'][0]:.6f} {item['position'][1]:.6f} "
                    f"{item['position'][2]:.6f} {item['position'][3]:.6f}\n"
                )
        with open(f"./train/annotation/{now}.json", "w") as f:
            dump(anno_label, f, indent=4)
        with tar_open("./train/train.tar.gz", "w:gz") as tar:
            tar.add("./train/images", arcname="images")
            tar.add("./train/split", arcname="split")

        if check_eula():
            if difficulty > 15 or get_config()["extensions"]["bgRemove"]:
                try:
                    multiprocessing.Process(
                        target=gh_upload_dataset, args=(now,)).start()
                except:
                    log("Failed to upload dataset to GitHub", "ERROR")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    print(
        f"florr-auto-afk v{constants.VERSION_INFO}({constants.VERSION_TYPE}.{constants.SUB_VERSION}) {constants.RELEASE_DATE}")
    error_label, filepath_label = None, None
    filepath = ""
    stop_event = multiprocessing.Event()
    debugger("Main thread started")
    page_stat = "launch"
    segment_process = None
    segment_running = multiprocessing.Value(ctypes.c_bool, False)
    announcement = generate_announcement(
        get_config()["advanced"]["skipUpdate"])

    if not get_config()["advanced"]["skipUpdate"]:
        try:
            update_models()
        except Exception as e:
            log("Failed to update models: " + str(e), "ERROR")
            log("Failed to fetch update", "WARNING")
            pass
    else:
        log("Skipping update", "WARNING")

    if constants.VERSION_TYPE == "Release":
        version = constants.VERSION_INFO
    elif constants.VERSION_TYPE == "Pre-Release":
        version = f"{constants.VERSION_INFO} Preview.{constants.SUB_VERSION}"
    else:
        version = f"{constants.VERSION_INFO} {constants.VERSION_TYPE}.{constants.SUB_VERSION}"

    if get_config()["extensions"]["enable"]:
        log("Attempting to start extension server", "INFO")
        multiprocessing.Process(
            target=server.start_extension_server, args=()).start()

    with multiprocessing.Manager() as manager:
        shared_logger = manager.list()
        root = tk.Tk()
        root.title(f"florr-auto-afk (v{version})")
        if get_theme() == "Dark":
            root.iconbitmap("./gui/dark.ico")
        else:
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
        test_button = ttk.Button(
            sidebar, text=get_ui_translation("page_test"), width=20, command=lambda: update_page("test"))
        test_button.pack(pady=10)
        extension_button = ttk.Button(
            sidebar, text=get_ui_translation("page_extensions"), width=20, command=lambda: update_page("extension"))
        extension_button.pack(pady=10)

        status_frame = ttk.Frame(sidebar)
        status_canvas = tk.Canvas(
            status_frame, width=10, height=10, bd=0, highlightthickness=0)
        status_label = ttk.Label(
            status_frame, text="", font=("Microsoft Yahei", 9), foreground="#ffffff" if get_theme() == "Dark" else "#000000")

        def update_connection_status():
            if not status_frame.winfo_exists():
                return
            if get_config()['extensions']['enable']:
                if experimental.get_connected():
                    color = "green"
                    text = get_ui_translation("extension_yes")
                else:
                    color = "red"
                    text = get_ui_translation("extension_no")
            else:
                color = "grey"
                text = get_ui_translation("extension_disabled")

            status_canvas.delete("all")
            status_canvas.create_oval(0, 0, 10, 10, fill=color, outline=color)
            status_label.config(text=text)
            root.after(get_config()['extensions']
                       ['swapInterval']*1000, update_connection_status)

        status_frame.pack(side="bottom", pady=(5, 10), anchor="w", fill="x")
        status_canvas.pack(side="left", padx=(0, 10))
        status_label.pack(side="left")

        settings_button = ttk.Button(
            sidebar, text=get_ui_translation("page_settings"), width=20, command=lambda: update_page("settings"))
        settings_button.pack(side="bottom", pady=0)
        main_content = ttk.Frame(root)
        main_content.pack(side="left", fill="both", expand=True)

        update_page(page_stat)
        update_connection_status()
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
        ScaleFactor = ctypes.windll.shcore.GetScaleFactorForDevice(0)
        root.tk.call('tk', 'scaling', ScaleFactor/75)
        sv_ttk.set_theme(theme)
        apply_theme_to_titlebar(root)
        show_share_warn()

        root.mainloop()

        if segment_process is not None and segment_process.is_alive():
            segment_process.terminate()
