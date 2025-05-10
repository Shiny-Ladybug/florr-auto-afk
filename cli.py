from segment_utils import *
import multiprocessing


def test_idle_thread(idled_flag, suppress_idle_detection):
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
            with idled_flag.get_lock():
                idled_flag.value = False
            log("No longer idle", "EVENT", save=False)
            last_pos = pos
        else:
            idle_iter += 1
        if idle_iter > (get_config()["runs"]["idleTimeThreshold"]//get_config()["runs"]["idleDetInterval"]):
            with idled_flag.get_lock():
                idled_flag.value = True
            log("Idle detected", "EVENT", save=False)
            sleep_time = get_config()["runs"]["idleDetIntervalMax"]
        sleep(sleep_time)


def afk_thread(idled_flag, suppress_idle_detection, afk_det_model, afk_seg_model):
    test_environment(afk_seg_model)
    log("恭喜你，你成功把代码跑起来了，你是这个👍", "INFO", save=False)
    countdown = get_config()["runs"]["runningCountDown"]
    if countdown == -1:
        log("Running indefinitely", "INFO")
        eta_timestamp = -1
    else:
        log(f"Running for {countdown} minutes", "INFO")
        eta_timestamp = int(time()) + countdown * 60
        eta_time = datetime.strftime(
            datetime.fromtimestamp(eta_timestamp), '%Y-%m-%d %H:%M:%S')
        log(f"ETA: {eta_time} / {eta_timestamp}", "INFO")
    while True:
        with idled_flag.get_lock():
            if not idled_flag.value:
                sleep(1)
                continue
        image = pyautogui.screenshot()
        ori_image = image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        position = detect_afk(image, afk_det_model)
        if position is None:
            if get_config()["advanced"]["verbose"]:
                log("No AFK window found", "EVENT", save=False)
            if countdown != -1 and time() > eta_timestamp:
                log("Countdown Ends, program exiting", "EVENT")
                break
            sleep(get_config()["advanced"]["epochInterval"])
            continue
        log("Found AFK window", "EVENT")
        if get_config()["executeBinary"]["runBeforeAFK"] != "":
            try:
                system("start "+get_config()["executeBinary"]["runBeforeAFK"])
            except:
                log("Cannot execute extra binary", "ERROR")
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
        if start is not None:
            start = (round(position[0][0]), round(position[0][1]))
            start = (start[0] + left_top_bound[0],
                     start[1] + left_top_bound[1])
        else:
            log("No start found", "WARNING")
        if end is not None:
            end = (round(position[1][0]), round(position[1][1]))
            end = (end[0] + left_top_bound[0], end[1] + left_top_bound[1])
        else:
            log("No end found, going for linear prediction", "WARNING")
        if masks is None:
            log("No masks found", "ERROR")
            save_image(image, "mask", "error")
            sleep(1)
            continue
        log("Using yolo to bypass AFK", "EVENT")
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
        if start is not None:
            cv2.circle(ori_image, start, 5, (0, 255, 0), -1)
        if end is not None:
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
                log("Cannot execute extra binary", "ERROR")
        if get_config()["advanced"]["useOBS"]:
            sleep(1)
            obs("stop")
        sleep(1)
        position = detect_afk(cv2.cvtColor(
            np.array(pyautogui.screenshot()), cv2.COLOR_RGB2BGR), afk_det_model)
        if position is not None:
            log("Cannot bypass AFK", "ERROR")
        else:
            log("Bypassed AFK", "EVENT")


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    multiprocessing.freeze_support()
    print(
        f"florr-auto-afk v{constants.VERSION_INFO}({constants.VERSION_TYPE}) {constants.RELEASE_DATE}")
    initiate()
    check_update()
    while True:
        if not get_config()["advanced"]["skipUpdate"]:
            try:
                update_models()
            except:
                print_exc()
                log("Cannot update models", "WARNING")
        try:
            afk_seg_model = YOLO(get_config()["yoloConfig"]["segModel"])
            afk_det_model = YOLO(get_config()["yoloConfig"]["detModel"])
            break
        except:
            log("YOLO models are corrupted, trying restore files", "ERROR")
            remove(get_config()["yoloConfig"]["segModel"])
            remove(get_config()["yoloConfig"]["detModel"])
            remove("./models/version")
    idled_flag = multiprocessing.Value('b', False)
    suppress_idle_detection = multiprocessing.Value('b', False)
    if not get_config()["runs"]["autoTakeOverWhenIdle"]:
        with idled_flag.get_lock():
            idled_flag.value = True
    else:
        log("Idle Detection is currently enabled, set `autoTakeOverWhenIdle` to `false` if you just want to test the AFK Bypass ability", "WARNING", save=False)
    idle_thread = multiprocessing.Process(
        target=test_idle_thread, args=(idled_flag, suppress_idle_detection))
    afk_thread_process = multiprocessing.Process(
        target=afk_thread, args=(idled_flag, suppress_idle_detection, afk_det_model, afk_seg_model))
    if get_config()["runs"]["autoTakeOverWhenIdle"]:
        idle_thread.start()
    afk_thread_process.start()
    if get_config()["runs"]["autoTakeOverWhenIdle"]:
        idle_thread.join()
    afk_thread_process.join()
