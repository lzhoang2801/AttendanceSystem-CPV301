import cv2
import time
import os
from face_detection import face_detection

def face_registration(username, user_id, frame, face):
    for (x, y, w, h, _) in face:
        cropped_face = frame[y:y+h, x:x+w]
        cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2GRAY)
        resized_face = cv2.resize(cropped_face, (100, 100))

        cv2.imwrite(f'dataset/face_registration/{user_id}_{username}_{time.time()}.jpg', resized_face)

    return True

if __name__ == '__main__':
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        raise ValueError('Could not open video capture')
        
    capture.set(3, 640)
    capture.set(4, 480)

    username = input('Enter your name: ').replace(' ', '').lower().strip()
    user_id = input('Enter your user ID: ').replace(' ', '').lower().strip()

    os.makedirs('dataset/face_registration', exist_ok=True)

    frame_count = 0
    saved_count = 0
    wait_frames = 10
    target_saves = 50

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        faces = []
        if frame_count % wait_frames == 0:
            faces = face_detection(frame)

            if len(faces) > 0:
                face_registration(username, user_id, frame, faces)
                saved_count += 1

        for (x, y, w, h, _) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Face Registration for ' + username, frame)

        frame_count += 1

        if saved_count >= target_saves:
            print(f'Registration complete. Saved {saved_count} images.')
            break
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()