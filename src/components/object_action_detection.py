import torch
import cv2
import numpy as np
from datetime import datetime
from plyer import notification
import os
import json
import objc

yolo_path = 'models/yolov5s.pt'

model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolo_path, force_reload=True)
model.conf = 0.5

## special model actions
actions = {
    'bottle': {
        'action': 'notify',
        'title': 'hydration reminder',
        'message': 'drink some water!'
    },
    'cell phone': {
        'action': 'notify',
        'title': 'phone usage',
        'message': 'your phone is in use'
    },
    'laptop': {
        'action': 'log',
        'message': 'laptop is being used'
    },
    'trash can': {
        'action': 'notify',
        'title': 'trash Reminder',
        'message': 'please empty the trash can'
    }
    # can add more here
}

## create log
logfile = 'object_detection_log.json'
if not os.path.exists(logfile):
    with open(logfile, 'w') as f:
        json.dump([], f)

# log message
def log_event(obj, message):
    with open(logfile, 'r') as f:
        data = json.load(f)
    data.append({
        'object': obj,
        'message': message,
        'time': str(datetime.now())
    })
    with open(logfile, 'w') as f:
        json.dump(data, f, indent=4)

## notification
def send_notification(title, message):
    notification.notify(
        title=title,
        message=message,
        timeout=5  # seconds
    )

def perform_action(obj):
    if obj in actions:
        action_info = actions[obj]
        if action_info['action'] == 'notify':
            send_notification(action_info['title'], action_info['message'])
        elif action_info['action'] == 'log':
            log_event(obj, action_info['message'])

## camera main loop

cap = cv2.VideoCapture(1)  # camera index changed

while True:
    ret, frame = cap.read()
    if not ret:
        print("Couldn't get frame")
        break

    results = model(frame)
    detections = results.pandas().xyxy[0]

    objects_detected = set()

    for index, row in detections.iterrows():
        obj = row['name'].lower()
        confidence = row['confidence']
        if obj in actions and obj not in objects_detected:
            perform_action(obj)
            objects_detected.add(obj)
    
    annotated_frame = np.squeeze(results.render())
    
    # frame
    cv2.imshow('object detection', annotated_frame)

    # exit by hitting escape
    if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()