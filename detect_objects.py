import cv2
import numpy as np
import os

# Load YOLO model
def load_yolo_model():
    config_path = "yolov4/yolov4.cfg"
    weights_path = "yolov4/yolov4.weights"
    labels_path = "yolov4/coco.names"

    # Load YOLO model files
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # Load class labels
    with open(labels_path, "r") as f:
        classes = f.read().strip().split("\n")

    return net, classes

# Perform object detection
def detect_objects(image_path, output_path, net, classes):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image.")
        return

    (H, W) = image.shape[:2]

    # YOLO input setup
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Get output layer names
    ln = net.getUnconnectedOutLayersNames()

    # Forward pass
    layer_outputs = net.forward(ln)

    # Initialize lists for detection results
    boxes, confidences, class_ids = [], [], []

    # Process detections
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                # Scale bounding box back to image size
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maxima suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(indices) > 0:
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # Draw bounding box and label
            color = (0, 255, 0)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
            cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Save and display the output image
    output_image_path = os.path.join(output_path, "detected_objects.jpg")
    cv2.imwrite(output_image_path, image)
    print(f"Output saved at: {output_image_path}")

    cv2.imshow("Detected Objects", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Paths
    image_path = "input_images/sample.jpg"
    output_path = "output_images/"

    # Create output folder if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Load YOLO model and classes
    net, classes = load_yolo_model()

    # Perform object detection
    detect_objects(image_path, output_path, net, classes)

