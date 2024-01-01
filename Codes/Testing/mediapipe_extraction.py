# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 16:37:20 2023

@author: 60183
"""

import os
import shutil
import cv2
import mediapipe as mp

# Initialize face detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)

# Initialize pose detection (for body keypoints)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

source_dir = "C:\\Users\\Alex\\Desktop\\Msc APU\\Sem 3\\Capstone\\Dataset - imgpool\\Female"
destination_dir = "C:\\Users\\Alex\\Desktop\\Msc APU\\Sem 3\\Capstone\\Dataset - imgpool fullbody\\Women2"

if not os.path.exists(destination_dir):
    os.mkdir(destination_dir)

image_counter = 1
for root, dirs, files in os.walk(source_dir):
    for file in files:
        if file.endswith(('jpg', 'jpeg', 'png')):
            image_path = os.path.join(root, file)
            image = cv2.imread(image_path)
            
            # Check for face
            results_face = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            if results_face.detections:
                # Check for full body keypoints
                results_pose = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                
                if results_pose.pose_landmarks:
                    # Extract the required keypoints
                    landmarks = results_pose.pose_landmarks.landmark
                    # Assuming left_ankle is landmark 15 and right_ankle is landmark 16 (can vary based on the model's output)
                    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
                    right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

                    # If both ankles (or any other keypoints is require) are detected, copy the image
                    if left_ankle and right_ankle:
                        dest_path = os.path.join(destination_dir, str(image_counter) + '.jpg')
                        shutil.copy(image_path, dest_path)
                        image_counter += 1

print("Processing complete.")

