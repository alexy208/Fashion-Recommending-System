# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 17:26:17 2023

@author: 60183
"""


import pandas as pd
import numpy as np
from skin_tone_and_body_extraction import process_image
import tkinter as tk
from tkinter import filedialog, Label, Button, Frame
from PIL import Image, ImageTk
import tensorflow as tf
from sklearn.metrics import pairwise_distances

BASE_DIR = "../Dataset - imgpool fullbody/Men"

# Load the dataset
df = pd.read_csv("skin_tone_and_body.csv")


X = df[["hue", "saturation", "value", "body_proportion_1", "body_proportion_2", "body_proportion_3"]].values
# Normalize the data
X_normalized = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))

# Load the trained encoder model
encoder = tf.keras.models.load_model('../Models/encoder_model.h5')

# Get the encoded representation of the dataset
X_encoded = encoder.predict(X_normalized)

def get_recommendations(image_path):
    # Process the uploaded image to get its features
    features = process_image(image_path)

    if features:
        hue, saturation, value = features[0]
        body_proportions = features[1]
        
        # Prepare the feature vector for the uploaded image
        test_vector = np.array([[hue, saturation, value] + list(body_proportions)])
        
        # Normalize the test_vector
        test_vector_normalized = (test_vector - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
        
        # Get the encoded representation of the uploaded image
        test_vector_encoded = encoder.predict(test_vector_normalized)
        
        # Compute distances between the new image and the images in the dataset in the encoded space
        distances = pairwise_distances(test_vector_encoded, X_encoded, metric='euclidean').flatten()
        
        # Get the indices of the 5 nearest images
        nearest_indices = np.argsort(distances)[:5]
        
        # Retrieve the image names of recommended items
        recommended_images = df['image_name'].iloc[nearest_indices].tolist()
        recommended_images_paths = [BASE_DIR + "\\" + img for img in recommended_images]
        return recommended_images_paths

    else:
        print(f"Failed to extract features from {image_path}.")
        return []

def upload_image():
    filepath = filedialog.askopenfilename()
    if not filepath:
        return

    # Display the uploaded image
    image = Image.open(filepath).resize((150, 150))
    photo = ImageTk.PhotoImage(image)
    uploaded_image_label.config(image=photo)
    uploaded_image_label.image = photo

    # Get recommendations
    recommended_images_paths = get_recommendations(filepath)
    display_recommendations(recommended_images_paths)

def display_recommendations(image_paths):
    for i, image_path in enumerate(image_paths):
        image = Image.open(image_path).resize((150, 150))
        photo = ImageTk.PhotoImage(image)
        recommended_image_labels[i].config(image=photo)
        recommended_image_labels[i].image = photo
        # Add the image name below each image
        image_name = image_path.split("\\")[-1]
        label = Label(recommendation_frame, text=image_name)
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

recommended_image_labels = [Label(recommendation_frame) for _ in range(5)]  # Assuming 5 recommendations
for i, label in enumerate(recommended_image_labels):
    label.grid(row=0, column=i, padx=10)

app.mainloop()
