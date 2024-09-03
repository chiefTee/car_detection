import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
cascade_path = os.path.join(current_dir, 'cars.xml')
car_cascade = cv2.CascadeClassifier(cascade_path)

if car_cascade.empty():
    raise IOError("Unable to load the car cascade classifier xml file")

cap = cv2.VideoCapture(r'C:\Users\Industry Expert\Desktop\Car_object_detection\4K Road traffic video for object detection and tracking - free download now!.mp4')

if not cap.isOpened():
    raise IOError("Unable to open video source")

frame_count = 0

plt.ion()
fig, ax = plt.subplots(figsize=(12, 8))

all_detected_cars = []

def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    
    intersection_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    box1_area = w1 * h1
    box2_area = w2 * h2
    
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    frame_count += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, 1.1, 5)
    
    for (x, y, w, h) in cars:
        new_car = True
        for detected_car in all_detected_cars:
            if calculate_iou((x, y, w, h), detected_car) > 0.3:
                new_car = False
                break
        
        if new_car:
            all_detected_cars.append((x, y, w, h))
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, 'New Car', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, 'Tracked Car', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    total_unique_cars = len(all_detected_cars)
    
    cv2.putText(frame, f'Frame: {frame_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f'Cars in frame: {len(cars)}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f'Total unique cars: {total_unique_cars}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    ax.clear()
    ax.imshow(frame_rgb)
    ax.axis('off')
    
    plt.tight_layout()
    plt.pause(0.01)
    
    if plt.waitforbuttonpress(timeout=0.1):
        break

cap.release()
plt.close(fig)