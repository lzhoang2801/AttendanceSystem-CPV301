import cv2
import os
import numpy as np
import pandas as pd
from face_detection import face_detection
from train_model import model_path, map_path, train_recognizer

if not os.path.exists(model_path) or not os.path.exists(map_path):
    face_recognizer = train_recognizer()
else:
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read(model_path)
    identity_map = pd.read_csv(map_path)

def recognize_face(frame):
    faces = face_detection(frame)

    for (x, y, w, h, _) in faces:
        cropped_face = frame[y:y+h, x:x+w]
        cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2GRAY)
        cropped_face = cv2.resize(cropped_face, (100, 100))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label, confidence = face_recognizer.predict(cropped_face)
        if confidence < 100:
            label_name = identity_map[identity_map['label_id'] == label]['label_name'].values[0]
            print(f"User: {label_name}, Confidence: {confidence}")
            cv2.putText(frame, f"User: {label_name}, Confidence: {confidence}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            print(f"Unknown Face")
            cv2.putText(frame, f"Unknown Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return frame

if __name__ == '__main__':
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        raise ValueError('Could not open video capture')
        
    capture.set(3, 640)
    capture.set(4, 480)

    while True:
        ret, frame = capture.read()
        if not ret:
            break
    
        recognize_face(frame)

        cv2.imshow('Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    capture.release()
    cv2.destroyAllWindows()