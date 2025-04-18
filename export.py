from ultralytics import YOLO


seg_model = YOLO("./models/afk-seg.pt")
det_model = YOLO("./models/afk-det.pt")

seg_model.export(format="onnx", dynamic=True)
det_model.export(format="onnx", dynamic=True)
