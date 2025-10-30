import cv2
from face_detection import face_detection

color_map = {0: (0, 255, 0), 1: (0, 0, 255), 2: (255, 0, 0)}

def capture_images():
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        return
        
    capture.set(3, 640)
    capture.set(4, 480)

    while True:
        ret, frame = capture.read()
        if not ret:
            break
    
        faces = face_detection(frame)

        for (x, y, w, h, v) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color_map[v], 2)

        cv2.imshow('Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("'q' pressed, stopping.")
            break
            
    capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    capture_images()