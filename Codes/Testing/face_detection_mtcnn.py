# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 11:43:45 2023

@author: 60183
"""


from mtcnn import MTCNN
import cv2
import os
import shutil

# Function to detect if an image has a face
def has_face(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(image)
    return len(results) > 0

# Define the MTCNN detector
detector = MTCNN()

# Root directory where images are stored
root_dir = "C:\\Users\\Alex\\Desktop\\Msc APU\\Sem 3\\Capstone\\Dataset\\DeepFashion\\In-shop Clothes Retrieval Benchmark\\Img\\img\\WOMEN"

# Destination directory
dest_dir = "C:\\Users\\Alex\\Desktop\\Msc APU\\Sem 3\\Capstone\\Dataset - imgpool\\Female"
counter = 1

# Recursive function to traverse subdirectories
def process_directory(directory):
    global counter
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        if os.path.isdir(file_path):
            process_directory(file_path)
        else:
            if has_face(file_path):
                # Define destination file path
                dest_file_path = os.path.join(dest_dir, str(counter) + ".jpg")
                shutil.copy2(file_path, dest_file_path)
                counter += 1

# Call the function to start processing
process_directory(root_dir)

print("Face images copied successfully!")
