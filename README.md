# People Counting and Tracking on Escalator

This project uses **YOLOv11**, **SORT**, and **OpenCV** to perform real-time people detection, tracking, and counting of individuals moving **up** and **down** an escalator in video footage.

![Demo](output.gif)

## Features

- **Person Detection**: Detects people in the escalator area.
- **Object Tracking**: Assigns unique IDs to each detected person.
- **Counting**: Counts people crossing defined virtual lines going **up** and **down**.
- **Custom Mask**: Focuses detection only on the escalator region using a mask image.
- **Video Output**: Saves processed video with overlays and counts.

## Requirements

- Python 3.x
- Install dependencies:
    ```bash
    pip install ultralytics opencv-python numpy cvzone
    ```

## Setup

1. Place the following files in the same directory:
   - `people.mp4` (Input video file)
   - `mask.png` (Binary mask image of escalator region)
   - `graphics.png` (Optional overlay image)
   - `yolo11n.pt` (YOLOv11 model file)
   
2. Run the script:
    ```bash
    python people-counter.py
    ```

3. The script will process the video, detect, track, and count people moving up and down the escalator, and save the annotated output as `output.avi`.

## Customization

- Adjust the **counting line coordinates** in the `limitsUp` and `limitsDown` variables.
- Modify the **confidence threshold** in the script to filter detections.
- Replace the mask image to suit different escalator or crowd areas.
- Use different YOLOv8 models if needed for accuracy or speed.

---

Feel free to open issues or contribute improvements!
