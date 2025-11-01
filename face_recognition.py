import cv2
import os
import pandas as pd
from datetime import datetime
from train_model import model_path, map_path, train_recognizer
from attendance import log_attendance
from face_detection import face_detection

def recognize_faces(frame, detected_faces, recognizer, identity_map, last_logged={}, threshold=80, attendance_mode=True):
    logged_new = False

    for (x, y, w, h, _) in detected_faces:
        is_recognized = False
        cropped_face = frame[y:y+h, x:x+w]
        cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2GRAY)
        cropped_face = cv2.resize(cropped_face, (100, 100))

        try:
            label, confidence = recognizer.predict(cropped_face)
        except:
            confidence = threshold * 2

        if confidence < threshold:
            try:
                label_name = identity_map.loc[identity_map['label_id'] == label, 'label_name'].values[0]
                is_recognized = True

                if attendance_mode and label_name not in last_logged:
                    now = datetime.now()
                    date = now.strftime('%Y-%m-%d')
                    time = now.strftime('%H:%M:%S')

                    log_attendance(label_name, date, time)
                    last_logged[label_name] = (date, time)
                    logged_new = True

                if not attendance_mode or logged_new:
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, label_name, (x, y - 10), cv2.FONT_ITALIC, 0.7, color, 2)
            except:
                pass

        if not is_recognized:
            color = (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_ITALIC, 0.7, color, 2)

    return frame, last_logged, logged_new

if __name__ == '__main__':
    if not os.path.exists(model_path) or not os.path.exists(map_path):
        face_recognizer = train_recognizer()
    else:
        face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        face_recognizer.read(model_path)
        identity_map = pd.read_csv(map_path)

    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        raise ValueError('Camera not found. Please check your camera connection and try again.')
        
    capture.set(3, 640)
    capture.set(4, 480)

    threshold = 80
    last_logged = {}

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        detected_faces = face_detection(frame)
        frame, last_logged, _ = recognize_faces(frame, detected_faces, face_recognizer, identity_map, last_logged, threshold, False)

        cv2.imshow('Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    capture.release()
    cv2.destroyAllWindows()