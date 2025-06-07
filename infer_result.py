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
    afk_window_pos = detect_afk_window(image, afk_det_model)
    if afk_window_pos is None:
        return
    cropped_image = crop_image(
        afk_window_pos[0], afk_window_pos[1], image)
    start_p, end_p, start_size, pack = detect_afk_things(
        cropped_image, afk_det_model, caller="")
    position = start_p, end_p, afk_window_pos, start_size
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
    dijkstra_stat = afk_path.dijkstra(afk_mask.mask)
    if not dijkstra_stat:
        afk_path.sort()
    afk_path.rdp(round(eval(get_config()["advanced"]["rdpEpsilon"].replace(
        "width", str(afk_mask.get_width())))))
    afk_path.extend(get_config()["advanced"]["extendLength"])
    line = afk_path.get_final(precise=False)
    annotated_image = draw_annotated_image(
        cropped_image, line, start_p, end_p, ((0, 0), (afk_window_pos[1][0]-afk_window_pos[0][0], afk_window_pos[1][1]-afk_window_pos[0][1])), afk_mask.inverse_start_color, afk_mask.get_width(), afk_path.get_length(), afk_path.get_difficulty(), afk_path.sort_method)
    return annotated_image


def inference_flow(images_path):
    images = os.listdir(images_path)
    images = [image for image in images if image.endswith(
        ('.jpg', '.png', '.jpeg'))]

    annotated_images_list = []

    for image_name in tqdm(images, desc="Processing images"):
        image_path = os.path.join(images_path, image_name)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image {image_path}. Skipping.")
            continue
        annotation = inference(image)
        if annotation is None:
            print(f"No annotation generated for {image_name}. Skipping.")
            continue
        annotated_images_list.append(annotation)

    if not annotated_images_list:
        print("No annotations were generated to create a gallery.")
        return

    resized_images = []
    max_side = 100
    for img in annotated_images_list:
        h, w = img.shape[:2]
        scale = max_side / max(h, w)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        resized = cv2.resize(
            img, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LANCZOS4)
        resized_images.append(resized)

    tile_h, tile_w = max_side, max_side

    num_annotations = len(resized_images)
    cols = int(round(math.sqrt(num_annotations)))
    if cols < 1:
        cols = 1
    rows = int(math.ceil(num_annotations / cols))

    gallery_height = rows * tile_h
    gallery_width = cols * tile_w
    gallery_image = np.full(
        (gallery_height, gallery_width, 3), 255, dtype=np.uint8)

    for i, ann_img in enumerate(resized_images):
        current_h, current_w = ann_img.shape[:2]
        row_idx = i // cols
        col_idx = i % cols
        y_offset = row_idx * tile_h + (tile_h - current_h) // 2
        x_offset = col_idx * tile_w + (tile_w - current_w) // 2
        gallery_image[y_offset: y_offset + current_h,
                      x_offset: x_offset + current_w] = ann_img

    gallery_output_path = os.path.join(images_path, "gallery_annotations.png")
    try:
        cv2.imwrite(gallery_output_path, gallery_image)
        print(f"Gallery image saved to: {gallery_output_path}")
    except Exception as e:
        print(f"Error saving gallery image: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--image_path", type=str, default="./raw", help="Path to the image(s) to be processed")
    args = parser.parse_args()
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(
            f"Image path {args.image_path} does not exist.")
    inference_flow(args.image_path)
