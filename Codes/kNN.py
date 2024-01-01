# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 23:55:11 2023

@author: 60183
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from skin_tone_and_body_extraction import process_image
import tkinter as tk
from tkinter import filedialog, Label, Button, Frame
from PIL import Image, ImageTk

BASE_DIR = "../Dataset - imgpool fullbody/men"
#BASE_DIR = "../Dataset - imgpool fullbody/Women"

# Load the dataset
df = pd.read_csv("../CSV/skin_tone_and_body_proportion_men.csv")
#df = pd.read_csv("../CSV/skin_tone_and_body_proportion_women.csv")

# Extract features
X = df[["hue", "saturation", "value", "body_proportion_1", "body_proportion_2", "body_proportion_3"]].values
X_normalized = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))


# Train KNN model to find 5 nearest neighbors
knn = NearestNeighbors(n_neighbors=5, metric = 'euclidean')
knn.fit(X_normalized)

def get_recommendations(image_path):
    # Process the uploaded image to get its features
    features = process_image(image_path)

    if features:
        hue, saturation, value = features[0]
        body_proportions = features[1]
        
        # Prepare the feature vector for the uploaded image
        test_vector = [[hue, saturation, value] + list(body_proportions)]
        
        # Normalize the test_vector using the same parameters as the training data
        test_vector_normalized = (test_vector - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
        
        # Get recommendations using the KNN model
        distances, indices = knn.kneighbors(test_vector_normalized)
        
        # Retrieve the image names of recommended items
        recommended_images = df['image_name'].iloc[indices[0]].tolist()
        recommended_distances = distances[0]
        recommended_images_paths = [BASE_DIR + "\\" + img for img in recommended_images]
        return list(zip(recommended_images_paths, recommended_distances))

    else:
        print(f"Failed to extract features from {image_path}.")
        return []

def upload_image():
    filepath = filedialog.askopenfilename()
    if not filepath:
        return

    # Display the uploaded image
    image = Image.open(filepath).resize((300, 300))
    photo = ImageTk.PhotoImage(image)
    uploaded_image_label.config(image=photo)
    uploaded_image_label.image = photo

    # Get recommendations
    recommended_images_paths = get_recommendations(filepath)
    display_recommendations(recommended_images_paths)

def display_recommendations(image_paths):
    for i, (image_path, distance) in enumerate(image_paths):
        image = Image.open(image_path).resize((300, 300))
        photo = ImageTk.PhotoImage(image)
        recommended_image_labels[i].config(image=photo)
        recommended_image_labels[i].image = photo
        # Add the image name below each image
        image_name = image_path.split("\\")[-1]
        distance_text = f"{image_name}\nDistance: {distance:.2f}"
        label = Label(recommendation_frame, text=distance_text)
        label.grid(row=1, column=i)

app = tk.Tk()
app.title("Image Recommender")

upload_button = Button(app, text="Upload Image", command=upload_image)
upload_button.pack(pady=20)

input_image_frame = Frame(app)
input_image_frame.pack(pady=20)

Label(input_image_frame, text="Input Image").grid(row=0, column=0)
uploaded_image_label = Label(input_image_frame)
uploaded_image_label.grid(row=1, column=0)

recommendation_frame = Frame(app)
recommendation_frame.pack(pady=20)

recommended_image_labels = [Label(recommendation_frame) for _ in range(5)]
for i, label in enumerate(recommended_image_labels):
    label.grid(row=0, column=i, padx=10)

app.mainloop()
