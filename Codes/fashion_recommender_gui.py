# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 12:18:00 2023

@author: 60183
"""

import tkinter as tk
from tkinter import Label, Button, Frame, filedialog
from PIL import Image, ImageTk

def get_recommendations(image_path, gender):
    # Your recommendation logic goes here
    print(f"Getting recommendations for {gender}")
    return []  # This should return a list of image paths

def create_app():
    app = tk.Tk()
    app.title("Image Recommender")
    
    # Set the window to full screen
    app.state('zoomed')

    # Function to hide the start screen and show the main app screen.
    def show_main_app(gender):
        start_frame.pack_forget()
        input_image_frame.pack(pady=20)
        recommendation_frame.pack(pady=20)
        upload_button.pack(pady=20)
        # You can now use the 'gender' variable to customize the app behavior

    # Start screen frame
    start_frame = Frame(app)
    start_frame.pack(expand=True, fill='both')

    male_button = Button(start_frame, text="Male", command=lambda: show_main_app('male'))
    male_button.pack(side='left', expand=True, padx=200, pady=100)

    female_button = Button(start_frame, text="Female", command=lambda: show_main_app('female'))
    female_button.pack(side='right', expand=True, padx=200, pady=100)

    # The rest of the GUI code goes here
    input_image_frame = Frame(app)
    Label(input_image_frame, text="Input Image").pack()

    uploaded_image_label = Label(input_image_frame)
    uploaded_image_label.pack()

    recommendation_frame = Frame(app)
    recommended_image_labels = [Label(recommendation_frame) for _ in range(5)]  # Assuming 5 recommendations
    for label in recommended_image_labels:
        label.pack(side='left', padx=10)

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
        # Pass the gender variable if needed to your get_recommendations function
        recommended_images_paths = get_recommendations(filepath, selected_gender.get())
        display_recommendations(recommended_images_paths)

    def display_recommendations(image_paths):
        for i, image_path in enumerate(image_paths):
            image = Image.open(image_path).resize((150, 150))
            photo = ImageTk.PhotoImage(image)
            recommended_image_labels[i].config(image=photo)
            recommended_image_labels[i].image = photo

    upload_button = Button(app, text="Upload Image", command=upload_image)

    # Variable to store selected gender
    selected_gender = tk.StringVar()

    app.mainloop()

if __name__ == "__main__":
    create_app()
