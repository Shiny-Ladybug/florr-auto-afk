import onnxruntime as ort
import numpy as np
import cv2


def preprocess_image(image_path, input_size):
    """
    Preprocess the input image for the YOLOv11 ONNX model.
    Args:
        image_path (str): Path to the input image.
        input_size (tuple): Model input size (width, height).
    Returns:
        np.ndarray: Preprocessed image.
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, input_size)
    image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]
    image = np.transpose(image, (2, 0, 1))  # HWC to CHW
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image


def postprocess_output(output, original_image_shape, input_size):
    """
    Postprocess the model output to map it back to the original image size.
    Args:
        output (np.ndarray): Model output.
        original_image_shape (tuple): Original image shape (height, width).
        input_size (tuple): Model input size (width, height).
    Returns:
        np.ndarray: Segmentation mask resized to the original image size.
    """
    mask = output[0]  # Assuming the first output is the segmentation mask
    mask = np.squeeze(mask)  # Remove batch dimension
    mask = cv2.resize(mask, (original_image_shape[1], original_image_shape[0]))
    return mask


def run_segmentation(model_path, image_path, input_size=(640, 640)):
    """
    Run segmentation using the YOLOv11 ONNX model.
    Args:
        model_path (str): Path to the ONNX model.
        image_path (str): Path to the input image.
        input_size (tuple): Model input size (width, height).
    """
    # Load the ONNX model
    session = ort.InferenceSession(model_path)

    # Get model input and output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Preprocess the input image
    original_image = cv2.imread(image_path)
    original_image_shape = original_image.shape[:2]
    input_image = preprocess_image(image_path, input_size)

    # Run inference
    outputs = session.run([output_name], {input_name: input_image})

    # Postprocess the output
    segmentation_mask = postprocess_output(
        outputs[0], original_image_shape, input_size)

    # Display the segmentation result
    cv2.imshow("Original Image", original_image)
    cv2.imshow("Segmentation Mask", segmentation_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Example usage
if __name__ == "__main__":
    model_path = "path_to_yolov11.onnx"  # Replace with your ONNX model path
    image_path = "path_to_image.jpg"  # Replace with your image path
    run_segmentation(model_path, image_path)
