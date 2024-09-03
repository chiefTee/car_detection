
import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from IPython.display import display, clear_output
import ipywidgets as widgets
import keyboard

import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import ipywidgets as widgets
import numpy as np

# Load the YOLO model
model = YOLO("yolov8n.pt")

# Open the video file
cap = cv2.VideoCapture(r'/content/drive/MyDrive/4K Road traffic video for object detection and tracking - free download now!.mp4')

if not cap.isOpened():
    raise IOError("Unable to open video source")

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Initialize VideoWriter to save the video output as MP4
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

frame_count = 0
saved_videos_count = 0  # Counter for videos saved with detected cars

# To track unique car detections
all_detected_cars = {}  # This will store all cars ever detected
active_cars = set()  # This will store currently active car IDs
next_car_id = 1
new_car_count = 0  # This will store the count of new cars

fig, ax = plt.subplots(figsize=(12, 8))
plt.ion()

video_widget = widgets.Output()
display(video_widget)

def iou(box1, box2):
    # Calculate intersection over union
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = intersection / float(area1 + area2 - intersection)
    return iou

def add_text_with_background(img, text, position, font_scale=1, thickness=2, text_color=(255,255,255), bg_color=(0,0,0), padding=5):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size

    x, y = position
    cv2.rectangle(img, (x - padding, y - text_h - padding - 8), (x + text_w + padding, y + padding), bg_color, -1)
    cv2.putText(img, text, (x, y - 5), font, font_scale, text_color, thickness, cv2.LINE_AA)

    return text_h + 2 * padding  # Return the height of the text box

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    frame_count += 1

    # Perform detection
    results = model(frame)

    boxes = results[0].boxes
    current_frame_cars = set()

    for box in boxes:
        if int(box.cls[0]) == 2:  # Class 2 corresponds to 'car'
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]

            # Check if this car matches any previously detected car
            matched_id = None
            for car_id, (prev_box, last_seen) in all_detected_cars.items():
                if iou((x1, y1, x2, y2), prev_box) > 0.5:  # Adjust threshold as needed
                    matched_id = car_id
                    break

            if matched_id is None:
                # New car detected
                car_id = next_car_id
                next_car_id += 1
                all_detected_cars[car_id] = ((x1, y1, x2, y2), frame_count)
                label = f'New car {car_id} {conf:.2f}'
                color = (0, 255, 0)  # Green for new cars
                new_car_count += 1  # Increment the new car count
            else:
                # Existing car detected
                all_detected_cars[matched_id] = ((x1, y1, x2, y2), frame_count)
                label = f'Tracked car {matched_id} {conf:.2f}'
                color = (0, 255, 255)  # Yellow for tracked cars
                car_id = matched_id

            current_frame_cars.add(car_id)

            # Draw the bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            add_text_with_background(frame, label, (x1, y1-10), 0.7, 2, color, (0,0,0))

    # Update active cars
    active_cars = current_frame_cars

    # Add text with background on the right side
    text_x = frame_width - 300  # Adjust this value to move text left or right
    text_y = 50
    line_height = 0

    line_height += add_text_with_background(frame, f'Frame: {frame_count}', (text_x, text_y + line_height), 1, 2)
    line_height += add_text_with_background(frame, f'Active cars: {len(active_cars)}', (text_x, text_y + line_height + 10), 1, 2)
    add_text_with_background(frame, f'New cars: {new_car_count}', (text_x, text_y + line_height + 20), 1, 2)

    # Convert the frame to RGB for displaying
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Display the frame in the widget
    with video_widget:
        ax.clear()
        ax.imshow(frame_rgb)
        ax.axis('off')
        plt.tight_layout()
        plt.pause(0.01)
        video_widget.clear_output(wait=True)

    # Write the processed frame to the output video file only if cars are detected
    if current_frame_cars:
        out.write(frame)
        saved_videos_count += 1

    # Stop the loop after saving 150 frames with detected cars
    if saved_videos_count >= 150:
        print("Saved 150 videos with detected cars. Stopping...")
        break

    # Check for user input to break the loop
    if plt.waitforbuttonpress(timeout=0.1):
        break

# Release resources
cap.release()
out.release()  # Release the video writer
plt.close(fig)

print("Video processing complete. Output saved as 'output.mp4'")
print(f"Total new cars detected: {new_car_count}")

