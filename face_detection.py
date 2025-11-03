import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

def equalize_sides(frame):
    height, width = frame.shape[:2]

    midX = int(width/2)
    leftSide = frame[int(0):height, int(0):midX]
    rightSide = frame[int(0):height, midX:width]

    equL = cv2.equalizeHist(leftSide)
    equR = cv2.equalizeHist(rightSide)

    equalized = np.concatenate((equL, equR), axis=1)

    return equalized

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

def find_best_face(faces, frame_width, frame_height):
    if len(faces) == 0:
        return np.array([])
    
    if len(faces) == 1:
        return faces

    frame_center_x = frame_width / 2
    frame_center_y = frame_height / 2

    areas = faces[:, 2] * faces[:, 3]
    face_centers_x = faces[:, 0] + faces[:, 2] / 2
    face_centers_y = faces[:, 1] + faces[:, 3] / 2

    dists_to_center = np.sqrt((face_centers_x - frame_center_x)**2 + (face_centers_y - frame_center_y)**2)

    max_area = np.max(areas)
    norm_areas = areas / max_area if max_area > 0 else np.zeros_like(areas)

    max_dist = np.max(dists_to_center)
    norm_dists = 1 - (dists_to_center / max_dist) if max_dist > 0 else np.ones_like(dists_to_center)

    scores = 0.7 * norm_areas + 0.3 * norm_dists

    best_face_index = np.argmax(scores)
    return np.array([faces[best_face_index]])

def face_detection(frame, registration_mode=False):
    scale_frame = 3
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray_frame, (0, 0), fx=1/scale_frame, fy=1/scale_frame)

    scaleFactor = 1.05
    minNeighbors = 4

    variants = [
        resized,
        cv2.equalizeHist(resized),
        equalize_sides(resized),
    ]

    detected_faces = []
    for index, variant in enumerate(variants):
        current_faces = face_cascade.detectMultiScale(variant, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
        for face in current_faces:
            detected_faces.append((*face * scale_frame, index))

    if len(detected_faces) > 0:
        detected_faces = non_max_suppression(np.array(detected_faces), iou_threshold=0.2)
        if registration_mode and len(detected_faces) > 0:
            frame_height, frame_width = frame.shape[:2]
            detected_faces = find_best_face(detected_faces, frame_width, frame_height)

    return detected_faces

if __name__ == '__main__':
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        raise ValueError('Camera not found. Please check your camera connection and try again.')
        
    capture.set(3, 640)
    capture.set(4, 480)

    while True:
        ret, frame = capture.read()
        if not ret:
            break
    
        faces = face_detection(frame, not True)

        for (x, y, w, h, v) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    capture.release()
    cv2.destroyAllWindows()