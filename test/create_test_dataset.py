import os
import random
import shutil

source_dir = 'dataset/lfw_funneled'

dest_dir = 'dataset/test_face_detection'
num_images = 1000

os.makedirs(dest_dir, exist_ok=True)

image_files = []
for dirpath, _, filenames in os.walk(source_dir):
    for filename in filenames:
        if filename.endswith('.jpg'):
            image_files.append(os.path.join(dirpath, filename))

print(f"Found {len(image_files)} images in {source_dir}.")

if len(image_files) > num_images:
    selected_files = random.sample(image_files, num_images)
else:
    selected_files = image_files

count = 0
for file_path in selected_files:
    file_name = file_path.split('\\')[-1]
    shutil.copy(file_path, os.path.join(dest_dir, file_name))
    count += 1

print(f"Copied {len(selected_files)} images to {dest_dir}.")