import onnxruntime
import numpy as np
import cv2


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - \
        new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / \
            shape[0]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right,
                            cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)


class YOLOv11det():
    def __init__(self, onnx_model, img_size=(640, 640), conf_thres=0.60, iou_thres=0.45):
        self.onnx_model = onnx_model
        self.img_size = img_size
        self.session = onnx_model
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = ['Window', 'Start', 'End']
        self.color_palette = np.random.uniform(
            0, 255, size=(len(self.classes), 3))

    def preprocess(self, img0s):
        img = letterbox(img0s, new_shape=self.img_size, auto=True)[0]

        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)

        img = img.astype(dtype=np.float32)
        img /= 255
        if len(img.shape) == 3:
            img = img[None]
        if len(img.shape) == 3:
            img = img[None]

        return img

    def postprocess(self, input_image, img_data, output):
        outputs = np.transpose(np.squeeze(output[0]))
        rows = outputs.shape[0]
        boxes = []
        scores = []
        class_ids = []
        ori_hight, ori_width = input_image.shape[:2]
        new_height, new_width = img_data.shape[2:]
        x_factor = ori_width / new_width
        y_factor = ori_hight / new_height
        for i in range(rows):
            classes_scores = outputs[i][4:]
            max_score = np.amax(classes_scores)
            if max_score >= self.conf_thres:
                class_id = np.argmax(classes_scores)
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])
        indices = cv2.dnn.NMSBoxes(
            boxes, scores, self.conf_thres, self.iou_thres)
        final_class_ids = []
        final_scores = []
        final_boxes = []
        for i in indices:
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]
            final_class_ids.append(class_id)
            final_scores.append(score)
            final_boxes.append(box)
        return final_class_ids, final_scores, final_boxes, input_image

    def infer(self, img0s):
        img_data = self.preprocess(img0s)
        outputs = self.session.run([self.session.get_outputs()[0].name], {
                                   self.session.get_inputs()[0].name: img_data})
        return self.postprocess(img0s, img_data, outputs)


def onnx_detect_afk(model, img, classes=['Window', 'Start', 'End']):
    yolov11_det = YOLOv11det(model)
    class_ids, scores, boxes, output_image = yolov11_det.infer(img)
    detected = []
    for class_id, score, box in zip(class_ids, scores, boxes):
        x_1, y_1, width, height = box
        x_2 = x_1 + width
        y_2 = y_1 + height
        x_avg = (x_1 + x_2) / 2
        y_avg = (y_1 + y_2) / 2
        object_name = classes[class_id]
        detected.append({
            "name": object_name,
            "x_1": x_1,
            "y_1": y_1,
            "x_2": x_2,
            "y_2": y_2,
            "x_avg": x_avg,
            "y_avg": y_avg,
            "confidence": score
        })
    return detected, img


if __name__ == "__main__":
    img = cv2.imread("./imgs/test.png")
    model = onnxruntime.InferenceSession("./models/afk-det.onnx", providers=[
        'CPUExecutionProvider'])
    detected, _ = onnx_detect_afk(model, img)
    print(detected)
