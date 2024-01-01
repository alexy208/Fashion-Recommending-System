# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 10:53:59 2023

@author: 60183
"""

import os
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from full_body_extract import get_body_proportion

# Initialize mediapipe components
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)

def extract_face_from_image(image):
    """Extract the largest face from the image."""
    if image is not None:
        # Get the dimensions of the image
        h, w, _ = image.shape

        # Calculate the half of the height
        half_h = h // 3

        # Crop the image from top to half
        # The crop syntax is image[start_row:end_row, start_col:end_col]
        image = image[0:half_h, 50:w-50]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image_rgb)
    
    if results.detections:
        bboxC = results.detections[0].location_data.relative_bounding_box
        ih, iw, _ = image.shape
        x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
        return image[y:y+h, x:x+w].copy()
    return None

def get_face_landmarks(face_img):
    """Get the landmarks for a given face."""
    results = face_mesh.process(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
    if results.multi_face_landmarks:
        return results.multi_face_landmarks[0]
    return None

def remove_features_from_face(face_img, landmarks, regions):
    """Remove eyes and mouth features from the face."""
    mask = np.zeros(face_img.shape[:2], dtype=np.uint8)
    for region in regions:
        points = np.array([[int(landmark.x * face_img.shape[1]), int(landmark.y * face_img.shape[0])] for landmark in (landmarks.landmark[p] for p in region)])
        cv2.fillPoly(mask, [points], 255)
    face_img[mask == 255] = [0, 0, 0]
    return face_img

def extract_skin_tone(face_img):
    """Extract the average skin tone from the face."""
    # Convert the face to HSV color space
    hsv_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)
    skin_mask = cv2.inRange(hsv_face, np.array([0, 40, 40]), np.array([20, 255, 255]))
    skin = cv2.bitwise_and(hsv_face, hsv_face, mask=skin_mask)
    non_black_pixels_mask = np.all(skin != [0, 0, 0], axis=-1)
    return cv2.mean(skin, mask=non_black_pixels_mask.astype("uint8"))[:3]

def process_image(filepath):
    """Process a single image to extract features."""
    image = cv2.imread(filepath)
    if image is None:
        print(f"Failed to load image at {filepath}")
        return None
    body_proportion = get_body_proportion(filepath)
    if body_proportion is None:
        print("Full body not detected in the image:", filepath)
        return None
    face_img = extract_face_from_image(image)
    if face_img is None:
        print("No face found in the image:", filepath)
        return None
    try:
        landmarks = get_face_landmarks(face_img)
        if landmarks:
            # Define landmark regions
            left_eye = [247,30,29,27,28,56,190,243,112,26,22,23,24,110,25]
            right_eye = [463, 414,286,258,257,259,260,467,359,255,339,254,253,252,256,341]
            mouth = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 185,40,39,37,0,267,269,270,409]
            regions = [left_eye, right_eye, mouth]

            face_img = remove_features_from_face(face_img, landmarks, regions)
            skin_tone = extract_skin_tone(face_img)
            return skin_tone, body_proportion
    except Exception as e:
        print(f"Error in processing landmarks for the image {filepath}: {e}")
        return None

def main(input_directory):
    """Main processing loop."""
    skin_tones_and_body = {}

    for filename in os.listdir(input_directory):
        filepath = os.path.join(input_directory, filename)
        features = process_image(filepath)
        
        if features:
            skin_tones_and_body[filename] = features

    # Convert to DataFrame and save
    formatted_data = pd.DataFrame([{
        'image_name': key,
        'hue': value[0][0],
        'saturation': value[0][1],
        'value': value[0][2],
        'body_proportion_1': value[1][0],
        'body_proportion_2': value[1][1],
        'body_proportion_3': value[1][2]
    } for key, value in skin_tones_and_body.items()])
    
    # Save the DataFrame to a CSV file
    csv_file = "skin_tone_and_body_proportion_men.csv"
    formatted_data.to_csv(csv_file, index=False)
    print(f"Data saved to {csv_file}")
    
if __name__ == "__main__":
    input_directory = "../Dataset - imgpool fullbody/Men"
    main(input_directory)
