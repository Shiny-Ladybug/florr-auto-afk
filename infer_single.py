from segment_utils import *
import argparse
import os
import json
import cv2
import numpy as np
import math
from tqdm import tqdm

afk_det_model = YOLO("./models/afk-det.pt")
afk_seg_model = YOLO("./models/afk-seg.pt")


def inference(image):
    position = detect_afk(image, afk_det_model,
                          caller="inference")
    if position is None:
        return
    start_p, end_p, window_pos, start_size = position
    cropped_image = crop_image(
        window_pos[0], window_pos[1], image)
    res = get_masks_by_iou(cropped_image, afk_seg_model)
    if res is None:
        return
    mask, _ = res
    if mask is None:
        print("No mask found for the detected object.")
        return
    afk_mask = AFK_Segment(image, mask, start_p, end_p, start_size)
    if start_p is not None:
        afk_mask.save_start()
    afk_path = AFK_Path(afk_mask.segment_path(), start_p,
                        end_p, afk_mask.get_width())
    afk_path.sort()
    afk_path.rdp(round(eval(get_config()["advanced"]["rdpEpsilon"].replace(
        "width", str(afk_mask.get_width())))))
    afk_path.extend(get_config()["advanced"]["extendLength"])
    line = afk_path.get_final(precise=False)
    annotated_image = draw_annotated_image(
        cropped_image, line, start_p, end_p, ((0, 0), (window_pos[1][0]-window_pos[0][0], window_pos[1][1]-window_pos[0][1])), afk_mask.inverse_start_color, afk_mask.get_width(), afk_path.get_length(), afk_path.get_difficulty())
    return annotated_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--image_path", type=str, default="./imgs/test.png", help="Path to the image(s) to be processed")
    args = parser.parse_args()
    image = cv2.imread(args.image_path)
    if image is None:
        raise ValueError(f"Could not read image from {args.image_path}.")
    annotated_image = inference(image)
    if annotated_image is None:
        print("No annotation generated for the image.")
    else:
        cv2.imshow("Annotated Image", annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
