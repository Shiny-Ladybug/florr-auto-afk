from segment_utils import *

test_environment()
log("恭喜你，你成功把代码跑起来了，你是这个👍", "INFO", save=False)
countdown = get_config()["runningCountDown(min)"]
if countdown == -1:
    log("Running indefinitely", "INFO")
    eta_timestamp = -1
else:
    log(f"Running for {countdown} minutes", "INFO")
    eta_timestamp = int(time())+countdown*60
    eta_time = datetime.strftime(
        datetime.fromtimestamp(eta_timestamp), '%Y-%m-%d %H:%M:%S')
    log(f"ETA: {eta_time} / {eta_timestamp}", "INFO")

while True:
    image = pyautogui.screenshot()
    ori_image = image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    position = detect_afk(image)
    if position == None:
        if get_config()["verbose"]:
            log("No AFK window found", "EVENT", save=False)
        if countdown != -1 and time() > eta_timestamp:
            log("Countdown Ends, program exiting", "EVENT")
            break
        sleep(get_config()["epochInterval"])
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
    image = crop_image(left_top_bound, right_bottom_bound, ori_image)
    results = afk_seg_model.predict(image, retina_masks=True, verbose=False)
    masks = results[0].masks
    start = position[0]
    end = position[1]
    if start != None:
        start = (round(position[0][0]), round(position[0][1]))
        start = (start[0] + left_top_bound[0],
                 start[1] + left_top_bound[1])
    else:
        log("No start found", "WARNING")
    if end != None:
        end = (round(position[1][0]), round(position[1][1]))
        end = (end[0] + left_top_bound[0], end[1] + left_top_bound[1])
    else:
        log("No end found, going for linear prediction", "WARNING")
    if masks == None:
        log("No masks found", "ERROR")
        save_image(image, "mask", "error")
        sleep(1)
        continue
    log("Using yolo to bypass AFK", "EVENT")
    line_ = segment_path(masks, start, end, left_top_bound)
    line = [line_[i]
            for i in range(0, len(line_), get_config()["optimizeQuantization"])]
    line = rdp(line, get_config()["rdpEpsilon"])
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
    if get_config()["showLogger"]:
        cv2.imshow("image", ori_image)
        cv2.waitKey(0)
    if get_config()["useOBS"]:
        obs("start")
        sleep(1)
    if get_config()["moveMouse"]:
        apply_mouse_movement(line)
    if get_config()["moveAfterAFK"]:
        move_a_bit()
    if get_config()["executeBinary"]["runAfterAFK"] != "":
        try:
            system("start "+get_config()["executeBinary"]["runAfterAFK"])
        except:
            log("Cannot execute extra binary", "ERROR")
    if get_config()["useOBS"]:
        sleep(1)
        obs("stop")
    sleep(1)
    position = detect_afk(cv2.cvtColor(
        np.array(pyautogui.screenshot()), cv2.COLOR_RGB2BGR))
    if position != None:
        log("Cannot bypass AFK", "ERROR")
    else:
        log("Bypassed AFK", "EVENT")
