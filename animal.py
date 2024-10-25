import os
import numpy as np
import pandas as pd
import cv2
import time
from time import sleep
import tensorflow as tf
from twilio.rest import Client
import pyttsx3
from statistics import mode

# Twilio credentials (ensure these are correct)
account_sid = 'AC49465a059d3200de610707898d4620e9'
auth_token = 'dcbfedbb018316d52ce40373c780991c'
twilio_number = '+15052072734'  # Your Twilio number

# Initialize the voice engine
text_speech = pyttsx3.init()

def say(message):
    text_speech.say(message)
    text_speech.runAndWait()

def makecall(animal, recipient):
    client = Client(account_sid, auth_token)
    alert_message = f"Caution, {animal} is approaching\n" * 5
    try:
        call = client.calls.create(
            twiml=f'<Response><Say voice="male" language="en">{alert_message}</Say></Response>',
            to=recipient,
            from_=twilio_number
        )
        print(f"Call initiated to {recipient} for {animal}. Call SID: {call.sid}")
    except Exception as e:
        print(f"Error making call: {e}")

# Load YOLO model
weights_path = r'D:\Kama\ANI-SOUL\yolov3.weights'
configuration_path = r'D:\Kama\ANI-SOUL\yolov3.cfg'
probability_minimum = 0.5
network = cv2.dnn.readNetFromDarknet(configuration_path, weights_path)

layers_names_all = network.getLayerNames()
outputlayers = [layers_names_all[i - 1] for i in network.getUnconnectedOutLayers()]

labels = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat',
          'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
          'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
          'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
          'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
          'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
          'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
          'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop',
          'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
          'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# Animals with higher threat levels
high_threat_animals = ['dog', 'horse', 'elephant', 'cow', 'sheep']

# Animal detection function
def ImagePath(frame):
    bounding_boxes = []
    confidences = []
    class_numbers = []
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    network.setInput(blob)
    output_from_network = network.forward(outputlayers)
    h, w = frame.shape[:2]

    for result in output_from_network:
        for detection in result:
            scores = detection[5:]
            class_current = np.argmax(scores)
            confidence_current = scores[class_current]
            if confidence_current > probability_minimum:
                box_current = detection[0:4] * np.array([w, h, w, h])
                x_center, y_center, box_width, box_height = box_current.astype('int')
                x_min = int(x_center - (box_width / 2))
                y_min = int(y_center - (box_height / 2))
                bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                confidences.append(float(confidence_current))
                class_numbers.append(class_current)

    if class_numbers:
        detected_animal = labels[mode(class_numbers)]
        return detected_animal
    return None

# Start video capture
video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
sleep(5)  # Allow time for the camera to warm up

while True:
    try:
        ret, frame = video.read()
        if not ret:
            print("Failed to capture video")
            break
        
        frame = cv2.flip(frame, 1)
        animal = ImagePath(frame)
        cv2.imshow("ANIMAL DETECTION!!", frame)

        if animal:
            print(f"Detected animal: {animal}")
            if animal in high_threat_animals:
                print("Classifying the animal: more harming possibility")
                say(f"Caution, {animal} is approaching.")
                makecall(animal, '+916380994878')  # Emergency number
            elif animal in ['butterfly', 'spider', 'squirrel']:
                print("Classifying the animal: less harming possibility")
                say(f"Alert, a {animal} is present.")
                makecall(animal, '+919042162169')  # Forest department number
            
            # Break loop after handling animal
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f"Error: {e}")
        continue

# Release video capture and destroy windows
video.release()
cv2.destroyAllWindows()
