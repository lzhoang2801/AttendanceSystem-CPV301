import cv2
import os
import numpy as np
import pandas as pd

model_path = 'models/face_recognizer.yml'
map_path = 'models/identity_map.csv'

def train_recognizer():
    image_dir = 'dataset/face_registration'
    if not os.path.exists(image_dir) or not os.listdir(image_dir):
        print(f"Training directory '{image_dir}' is empty or does not exist. Please register faces first.")
        return None

    faces = []
    labels_map = {}
    label_ids = []
    for filename in os.listdir(image_dir):
        try:
            label_name = '_'.join(filename.split('_')[0:2])
            image_path = os.path.join(image_dir, filename)
            face = cv2.imread(image_path)
            if face is not None:
                faces.append(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY))
                labels_map[label_name] = len(labels_map)
            
            label_ids.append(labels_map[label_name])
        except:
            print(f"Failed to process: {filename}")
            continue
            
    if not faces:
        print("No valid training images found.")
        return None

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(label_ids)) 
    
    os.makedirs('models', exist_ok=True)
    
    recognizer.save(model_path)
    pd.DataFrame(labels_map.items(), columns=['label_name', 'label_id']).to_csv(map_path, index=False)
    
    print(f"Model trained and saved as '{model_path}'")
    print(f"Identity map saved as '{map_path}'")
    
    return recognizer

if __name__ == '__main__':
    train_recognizer()