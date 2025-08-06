import cv2
import mediapipe as mp
import numpy as np
import time
import os
import csv
import pyttsx3
from datetime import datetime

# Voice engine setup
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech

# Create output folders
os.makedirs("output_faces", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Logging CSV
log_file = open("logs/predictions_log.csv", mode='w', newline='')
csv_writer = csv.writer(log_file)
csv_writer.writerow(["Timestamp", "Gender", "Gender_Conf", "Age", "Age_Conf", "Alert"])

# Class Labels
AGE_CLASSES = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_CLASSES = ['Male', 'Female']

# Load Age & Gender models
age_net = cv2.dnn.readNetFromCaffe('models/age_deploy.prototxt', 'models/age_net.caffemodel')
gender_net = cv2.dnn.readNetFromCaffe('models/gender_deploy.prototxt', 'models/gender_net.caffemodel')

# MediaPipe Face Detector
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)

last_spoken = ""  # Track last spoken identity

# Speak function
def speak(text):
    global last_spoken
    if text != last_spoken:
        engine.say(text)
        engine.runAndWait()
        last_spoken = text

# Detection function
def detector(frame):
    height, width, _ = frame.shape
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(img_rgb)

    count = 0

    try:
        for detection in results.detections:
            count += 1
            box = detection.location_data.relative_bounding_box

            x = int(box.xmin * width)
            y = int(box.ymin * height)
            w = int(box.width * width)
            h = int(box.height * height)

            margin = 20
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(width, x + w + margin)
            y2 = min(height, y + h + margin)

            face = frame[y1:y2, x1:x2]
            if face.shape[0] < 50 or face.shape[1] < 50:
                continue

            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                                         (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

            # Gender
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender_idx = gender_preds[0].argmax()
            gender = GENDER_CLASSES[gender_idx]
            gender_conf = round(gender_preds[0][gender_idx] * 100, 2)

            # Age
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age_idx = age_preds[0].argmax()
            age = AGE_CLASSES[age_idx]
            age_conf = round(age_preds[0][age_idx] * 100, 2)

            # Alert condition
            alert = ""
            if age in ['(0-2)', '(4-6)', '(60-100)']:
                alert = "👶👵 Age Alert"
                alert_color = (0, 0, 255)
            else:
                alert_color = (0, 255, 0)

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Save face
            filename = f"output_faces/{timestamp.replace(':', '-')}_{gender}_{age}.jpg"
            cv2.imwrite(filename, face)

            # Log to CSV
            csv_writer.writerow([timestamp, gender, gender_conf, age, age_conf, alert])

            # Display label
            label = f"{gender} ({gender_conf}%), {age} ({age_conf}%)"
            cv2.rectangle(frame, (x1, y1), (x2, y2), alert_color, 2)
            cv2.rectangle(frame, (x1, y1 - 25), (x2, y1), alert_color, -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

            if alert:
                cv2.putText(frame, alert, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Voice alert
            speak(f"{gender}, age group {age.replace('(', '').replace(')', '').replace('-', ' to ')}")

    except Exception as e:
        print("Detection Error:", e)

    return count, frame

# Webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

print("✅ Webcam started. Press 'q' to exit.")

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to read frame.")
        break

    count, output = detector(frame)

    # FPS
    curr_time = time.time()
    fps = round(1 / (curr_time - prev_time), 2)
    prev_time = curr_time

    cv2.putText(output, f"Faces: {count}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)
    cv2.putText(output, f"FPS: {fps}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 0), 2)

    cv2.imshow("🧠 ML - Age & Gender Detection with Voice", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

log_file.close()
cap.release()
cv2.destroyAllWindows()
