from segment_utils import *
import argparse
import os
import json

afk_det_model = YOLO("./models/afk-det.pt")
afk_seg_model = YOLO("./models/afk-seg.pt")


def export_result_to_label(results, image, epsilon=1, path=None):
    if results[0].masks is None:
        return
    image_height, image_width, _ = image.shape
    labelme_data = {
        "version": "5.6.0",
        "flags": {},
        "shapes": [],
        "imagePath": path,
        "imageData": None,
        "imageHeight": image_height,
        "imageWidth": image_width
    }
    if results and results[0].masks:
        for i in range(len(results[0].masks)):
            original_polygon_points = results[0].masks.xy[i]
            simplified_polygon_np = rdp(
                original_polygon_points, epsilon=epsilon)
            simplified_polygon_points = simplified_polygon_np.tolist()
            shape_entry = {
                "label": "path",
                "points": simplified_polygon_points,
                "group_id": None,
                "description": "",
                "shape_type": "polygon",
                "flags": {},
                "mask": None
            }
            labelme_data["shapes"].append(shape_entry)
    return labelme_data


def inference(image, image_path):
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
    mask, results = res
    label = export_result_to_label(results, cropped_image, path=image_path)
    return label, cropped_image


def inference_flow(images_path, outputs_path):
    if not os.path.exists(outputs_path):
        os.makedirs(outputs_path)
    images = os.listdir(images_path)
    images = [image for image in images if image.endswith(
        ('.jpg', '.png', '.jpeg'))]
    for image_name in images:
        image_path = os.path.join(images_path, image_name)
        image = cv2.imread(image_path)
        res = inference(
            image, os.path.abspath(os.path.join(outputs_path, image_name)))
        if res is None:
            print(f"Image {image_name} has no AFK detected.")
            continue
        label, cropped_image = res
        if label is None:
            continue
        label_name = os.path.splitext(image_name)[0] + ".json"
        label_path = os.path.join(outputs_path, label_name)
        with open(label_path, "w") as f:
            json.dump(label, f, indent=4)
        cv2.imwrite(os.path.join(outputs_path, image_name), cropped_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--image_path", type=str, default="./saves", help="Path to the image(s) to be processed")
    parser.add_argument(
        "-o", "--output_path", type=str, default="./outputs", help="Path to save the output image(s)")
    args = parser.parse_args()
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(
            f"Image path {args.image_path} does not exist.")
    inference_flow(args.image_path, args.output_path)
