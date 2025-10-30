import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

def rotate_and_equalize(frame):
    height, width = frame.shape[:2]

    midX = int(width/2)
    leftSide = frame[int(0):height, int(0):midX]
    rightSide = frame[int(0):height, midX:width]

    equL = cv2.equalizeHist(leftSide)
    equR = cv2.equalizeHist(rightSide)

    rotated_equalized = np.concatenate((equL, equR), axis=1)

    return rotated_equalized

def non_max_suppression(boxes, iou_threshold=0.3):
    if boxes is None or len(boxes) == 0:
        return boxes

    boxes = np.asarray(boxes)
    if boxes.ndim == 1:
        boxes = boxes.reshape(1, -1)

    x1 = boxes[:, 0].astype(float)
    y1 = boxes[:, 1].astype(float)
    w = boxes[:, 2].astype(float)
    h = boxes[:, 3].astype(float)

    x2 = x1 + w
    y2 = y1 + h
    areas = w * h

    order = areas.argsort()
    keep = []
    while order.size > 0:
        i = order[-1]
        keep.append(i)

        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[:-1]])
        yy1 = np.maximum(y1[i], y1[order[:-1]])
        xx2 = np.minimum(x2[i], x2[order[:-1]])
        yy2 = np.minimum(y2[i], y2[order[:-1]])

        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        inter = inter_w * inter_h

        union = areas[i] + areas[order[:-1]] - inter
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)

        order = order[np.where(iou <= iou_threshold)[0]]

    return boxes[keep].astype(int)

def face_detection(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray_frame, (0, 0), fx=1/3, fy=1/3)

    scaleFactor = 1.05
    minNeighbors = 5

    variants = [
        resized,
        cv2.equalizeHist(resized),
        rotate_and_equalize(resized),
    ]

    faces = []
    for index, variant in enumerate(variants):
        current_faces = face_cascade.detectMultiScale(variant, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
        if len(current_faces) > 0:
            for face in current_faces:
                faces.append((*face * 3, index))

    if len(faces) > 0:
        faces = non_max_suppression(faces, iou_threshold=0.2)

    return faces

if __name__ == '__main__':
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        raise ValueError('Could not open video capture')
        
    capture.set(3, 640)
    capture.set(4, 480)

    color_map = {0: (0, 255, 0), 1: (0, 0, 255), 2: (255, 0, 0)}

    while True:
        ret, frame = capture.read()
        if not ret:
            break
    
        faces = face_detection(frame)

        for (x, y, w, h, v) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color_map[v], 2)

        cv2.imshow('Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    capture.release()
    cv2.destroyAllWindows()