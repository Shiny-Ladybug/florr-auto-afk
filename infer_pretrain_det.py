from segment_utils import *
import argparse
import os
import json

afk_det_model = YOLO("./models/afk-det.pt")


def export_detection_to_label(packs, image):
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


def inference(image):
    afk_window_pos = detect_afk_window(image, afk_det_model)
    if afk_window_pos is None:
        return
    cropped_image = crop_image(
        afk_window_pos[0], afk_window_pos[1], image.copy())
    start_p, end_p, start_size, pack = detect_afk_things(
        cropped_image, afk_det_model, caller="")
    if pack[0] is not None:
        pack[0] = [(afk_window_pos[0][0]+pack[0][0][0], afk_window_pos[0][1]+pack[0][0][1]),
                   (afk_window_pos[0][0]+pack[0][1][0], afk_window_pos[0][1]+pack[0][1][1])]
    if pack[1] is not None:
        pack[1] = [(afk_window_pos[0][0]+pack[1][0][0], afk_window_pos[0][1]+pack[1][0][1]),
                   (afk_window_pos[0][0]+pack[1][1][0], afk_window_pos[0][1]+pack[1][1][1])]
    packs = [
        afk_window_pos, pack[0], pack[1]]
    return export_detection_to_label(packs, image), image


def inference_flow(images_path, outputs_path):
    classes = ["Window", "Start", "End"]
    if not os.path.exists(outputs_path):
        os.makedirs(outputs_path)
    if not os.path.exists(os.path.join(outputs_path, "classes.txt")):
        with open(os.path.join(outputs_path, "classes.txt"), "w") as f:
            for item in classes:
                f.write(item + "\n")
    images = os.listdir(images_path)
    images = [image for image in images if image.endswith(
        ('.jpg', '.png', '.jpeg'))]
    for image_name in images:
        image_path = os.path.join(images_path, image_name)
        image = cv2.imread(image_path)
        res = inference(image)
        if res is None:
            print(f"Image {image_name} has no AFK detected.")
            continue
        label, cropped_image = res
        if label is None:
            continue
        label_name = os.path.splitext(image_name)[0] + ".txt"
        label_path = os.path.join(outputs_path, label_name)
        with open(label_path, "w") as f:
            for item in label:
                f.write(
                    f"{item['class']} "
                    f"{item['position'][0]:.6f} {item['position'][1]:.6f} "
                    f"{item['position'][2]:.6f} {item['position'][3]:.6f}\n"
                )
        cv2.imwrite(os.path.join(outputs_path, image_name), cropped_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--image_path", type=str, default="./extension", help="Path to the image(s) to be processed")
    parser.add_argument(
        "-o", "--output_path", type=str, default="./outputs", help="Path to save the output image(s)")
    args = parser.parse_args()
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(
            f"Image path {args.image_path} does not exist.")
    inference_flow(args.image_path, args.output_path)
