# Object Detection with YOLOv4

This project demonstrates how to perform object detection using the pre-trained YOLOv4 model with OpenCV.

---

## Features
- Detects objects in images with bounding boxes and class labels.
- Utilizes pre-trained YOLOv4 on the COCO dataset.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/object_detection_yolov4.git
   cd object_detection_yolov4
2. Install dependencies:
   ``` bash
   pip install -r requirements.txt
3. Download the YOLOv4 model files:
   ``` bash
   https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov4.cfg
   https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
   https://github.com/pjreddie/darknet/blob/master/data/coco.names
4. Place the files in the yolov4/ folder.
5. Run the script:
   ``` bash
   python detect_objects.py
