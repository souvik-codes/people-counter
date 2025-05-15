import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

# Load video
cap = cv2.VideoCapture("people.mp4")

# Load YOLOv8 model
model = YOLO("yolo11n.pt")

# COCO class names
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Load mask and overlay images
original_mask = cv2.imread("mask.png")
imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)

# Tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Counting lines
limitsUp = [103, 161, 296, 161]
limitsDown = [527, 489, 735, 489]

# Counters
totalCountUp = []
totalCountDown = []

# Get frame dimensions and video properties
ret, frame = cap.read()
if not ret:
    print("Failed to read video.")
    exit()

frame_height, frame_width = frame.shape[:2]
fps = cap.get(cv2.CAP_PROP_FPS)

# Resize mask once
mask = cv2.resize(original_mask, (frame_width, frame_height))

# Define video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for MP4
out = cv2.VideoWriter("output.mp4", fourcc, fps, (frame_width, frame_height))

# Reset video to first frame
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

while True:
    success, img = cap.read()
    if not success:
        break

    imgRegion = cv2.bitwise_and(img, mask)

    if imgGraphics is not None:
        img = cvzone.overlayPNG(img, imgGraphics, (730, 15))

    results = model(imgRegion, stream=True)

    detections = np.empty((0, 5))

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == "person" and conf > 0.3:
                detections = np.vstack((detections, np.array([x1, y1, x2, y2, conf])))

    resultsTracker = tracker.update(detections)

    # Draw lines
    cv2.line(img, tuple(limitsUp[:2]), tuple(limitsUp[2:]), (0, 0, 255), 5)
    cv2.line(img, tuple(limitsDown[:2]), tuple(limitsDown[2:]), (0, 0, 255), 5)

    for result in resultsTracker:
        x1, y1, x2, y2, id = map(int, result)
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2

        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(0, 0, 255))
        cvzone.putTextRect(img, f' {int(id)}', (x1, max(y1, 35)), scale=2, thickness=3, offset=10, colorR=(0, 0, 255))
        cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)

        if limitsUp[0] < cx < limitsUp[2] and limitsUp[1] - 15 < cy < limitsUp[1] + 15:
            if id not in totalCountUp:
                totalCountUp.append(id)
                cv2.line(img, tuple(limitsUp[:2]), tuple(limitsUp[2:]), (0, 255, 0), 5)

        if limitsDown[0] < cx < limitsDown[2] and limitsDown[1] - 15 < cy < limitsDown[1] + 15:
            if id not in totalCountDown:
                totalCountDown.append(id)
                cv2.line(img, tuple(limitsDown[:2]), tuple(limitsDown[2:]), (0, 255, 0), 5)

    # Show counts
    cv2.putText(img, str(len(totalCountUp)), (929, 100), cv2.FONT_HERSHEY_PLAIN, 5, (139, 195, 75), 7)
    cv2.putText(img, str(len(totalCountDown)), (1191, 100), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 230), 7)

    # Write frame to output video
    out.write(img)

    # Show frame
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()
