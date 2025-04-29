import cv2
import numpy as np
import onnxruntime

CONFIDENCE_THRESHOLD = 0.45
IOU_THRESHOLD = 0.50
MASK_THRESHOLD = 0.5
MODEL_INPUT_SHAPE = (640, 640)
CLASS_NAMES = ['path']


def preprocess_image(img, input_shape):
    img_height, img_width = img.shape[:2]
    input_width, input_height = input_shape
    scale = min(input_width / img_width, input_height / img_height)
    new_width = int(img_width * scale)
    new_height = int(img_height * scale)
    resized_img = cv2.resize(img, (new_width, new_height),
                             interpolation=cv2.INTER_LINEAR)
    top_pad = (input_height - new_height) // 2
    bottom_pad = input_height - new_height - top_pad
    left_pad = (input_width - new_width) // 2
    right_pad = input_width - new_width - left_pad
    padded_img = cv2.copyMakeBorder(resized_img, top_pad, bottom_pad, left_pad, right_pad,
                                    cv2.BORDER_CONSTANT, value=(114, 114, 114))
    blob = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
    blob = blob.transpose(2, 0, 1)
    blob = blob.astype(np.float32) / 255.0
    blob = np.expand_dims(blob, axis=0)
    return blob, img, (img_width, img_height), (scale, top_pad, left_pad)


def process_masks(mask_predictions, mask_prototypes, detection_info, original_shape, pad_info):
    num_masks, proto_h, proto_w = mask_prototypes.shape
    num_detections = len(detection_info['boxes'])
    final_masks = []
    scale, top_pad, left_pad = pad_info
    original_w, original_h = original_shape
    if num_detections == 0:
        return np.array([])
    mask_coeffs = np.array([det['mask_coeffs']
                           for det in detection_info['boxes']])
    generated_masks = sigmoid(
        mask_coeffs @ mask_prototypes.reshape(num_masks, -1)).reshape(-1, proto_h, proto_w)
    input_w, input_h = MODEL_INPUT_SHAPE
    padded_crop_boxes = np.array([det['padded_box']
                                 for det in detection_info['boxes']])
    for i in range(num_detections):
        upscaled_mask = cv2.resize(
            generated_masks[i], (input_w, input_h), interpolation=cv2.INTER_LINEAR)
        x1_pad, y1_pad, x2_pad, y2_pad = map(int, padded_crop_boxes[i])
        y1_pad = max(0, y1_pad)
        x1_pad = max(0, x1_pad)
        y2_pad = min(input_h, y2_pad)
        x2_pad = min(input_w, x2_pad)
        cropped_mask_padded = upscaled_mask[y1_pad:y2_pad, x1_pad:x2_pad]
        orig_y1 = max(0, round((y1_pad - top_pad) / scale))
        orig_x1 = max(0, round((x1_pad - left_pad) / scale))
        orig_y2 = min(original_h, round((y2_pad - top_pad) / scale))
        orig_x2 = min(original_w, round((x2_pad - left_pad) / scale))
        original_crop_h = orig_y2 - orig_y1
        original_crop_w = orig_x2 - orig_x1
        if original_crop_h > 0 and original_crop_w > 0 and cropped_mask_padded.shape[0] > 0 and cropped_mask_padded.shape[1] > 0:
            final_mask_segment = cv2.resize(
                cropped_mask_padded, (original_crop_w, original_crop_h), interpolation=cv2.INTER_LINEAR)
            final_mask_segment = (final_mask_segment > MASK_THRESHOLD).astype(
                np.uint8)
            full_mask = np.zeros((original_h, original_w), dtype=np.uint8)
            full_mask[orig_y1:orig_y2, orig_x1:orig_x2] = final_mask_segment
            final_masks.append(full_mask)
        else:
            final_masks.append(
                np.zeros((original_h, original_w), dtype=np.uint8))
    return np.array(final_masks)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def postprocess_results(outputs, original_shape, pad_info):
    detections = outputs[0][0]
    mask_prototypes = outputs[1][0]
    detections = detections.T
    num_classes = len(CLASS_NAMES)
    boxes_xywh = detections[:, :4]
    class_scores = detections[:, 4:4+num_classes]
    mask_coeffs_raw = detections[:, 4+num_classes:]
    max_scores = np.max(class_scores, axis=1)
    class_ids = np.argmax(class_scores, axis=1)
    keep_indices = max_scores >= CONFIDENCE_THRESHOLD
    if not np.any(keep_indices):
        return {'boxes': [], 'scores': [], 'class_ids': []}, np.array([]), np.array([])
    filtered_boxes_xywh = boxes_xywh[keep_indices]
    filtered_scores = max_scores[keep_indices]
    filtered_class_ids = class_ids[keep_indices]
    filtered_mask_coeffs = mask_coeffs_raw[keep_indices]
    cx, cy, w, h = filtered_boxes_xywh.T
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    filtered_boxes_xyxy = np.vstack((x1, y1, x2, y2)).T
    indices_nms = cv2.dnn.NMSBoxes(filtered_boxes_xyxy.tolist(), filtered_scores.tolist(),
                                   CONFIDENCE_THRESHOLD, IOU_THRESHOLD)
    if len(indices_nms) > 0 and isinstance(indices_nms[0], (list, np.ndarray)):
        indices_nms = indices_nms.flatten()
    if len(indices_nms) == 0:
        return {'boxes': [], 'scores': [], 'class_ids': []}, np.array([]), np.array([])
    final_boxes_xyxy_padded = filtered_boxes_xyxy[indices_nms]
    final_scores = filtered_scores[indices_nms]
    final_class_ids = filtered_class_ids[indices_nms]
    final_mask_coeffs = filtered_mask_coeffs[indices_nms]
    scale, top_pad, left_pad = pad_info
    original_w, original_h = original_shape
    final_boxes_orig = []
    for box in final_boxes_xyxy_padded:
        bx1, by1, bx2, by2 = box
        bx1_no_pad = bx1 - left_pad
        by1_no_pad = by1 - top_pad
        bx2_no_pad = bx2 - left_pad
        by2_no_pad = by2 - top_pad
        orig_x1 = round(bx1_no_pad / scale)
        orig_y1 = round(by1_no_pad / scale)
        orig_x2 = round(bx2_no_pad / scale)
        orig_y2 = round(by2_no_pad / scale)
        orig_x1 = max(0, orig_x1)
        orig_y1 = max(0, orig_y1)
        orig_x2 = min(original_w, orig_x2)
        orig_y2 = min(original_h, orig_y2)
        final_boxes_orig.append([orig_x1, orig_y1, orig_x2, orig_y2])
    final_boxes_orig = np.array(final_boxes_orig)
    detection_info = {
        'boxes': [],
        'scores': final_scores.tolist(),
        'class_ids': final_class_ids.tolist()
    }
    for i in range(len(final_boxes_orig)):
        detection_info['boxes'].append({
            'orig_box': final_boxes_orig[i].tolist(),
            'padded_box': final_boxes_xyxy_padded[i].tolist(),
            'mask_coeffs': final_mask_coeffs[i]
        })
    final_masks = process_masks(
        mask_predictions=None,
        mask_prototypes=mask_prototypes,
        detection_info=detection_info,
        original_shape=original_shape,
        pad_info=pad_info
    )
    return_boxes = [det['orig_box'] for det in detection_info['boxes']]
    return detection_info, return_boxes, final_masks


def onnx_seg_afk(model, img):
    input_name = model.get_inputs()[0].name
    output_names = [output.name for output in model.get_outputs()]
    input_blob, original_image, original_shape, pad_info = preprocess_image(
        img, MODEL_INPUT_SHAPE)
    outputs = model.run(output_names, {input_name: input_blob})
    detection_info, final_boxes, final_masks = postprocess_results(
        outputs, original_shape, pad_info)
    return final_masks


if __name__ == "__main__":
    model = onnxruntime.InferenceSession("./models/afk-seg.onnx", providers=[
        'CPUExecutionProvider'])
    masks = onnx_seg_afk(model, cv2.imread("./imgs/test.png"))
    for mask in masks:
        mask = (mask * 255).astype(np.uint8)
        skeleton = cv2.ximgproc.thinning(mask)
        contours, _ = cv2.findContours(
            skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[0]
    print(contour)
