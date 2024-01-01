# Fashion Recommending System based on skin-tone and body proportion

## Project Description
This repository contains the implementation of a novel Fashion Recommendation System that leverages advanced machine learning techniques to offer personalized fashion item recommendations. The system focuses on analyzing users' skin tone and body proportions to suggest fashion items that complement their unique features.

## Methodology

The project involves several stages:

- Feature Extraction: Implementing advanced computer vision techniques to extract relevant features (skin tone and body proportions) from user images.

- KNN for Similarity Measurement: Utilizing the K-Nearest Neighbors (KNN) algorithm to find the most similar fashion items based on the extracted features.

# How to Use

To use this recommendation system:

1) Clone the repository.
2) Install required dependencies:
   ```bash
   pip install -r Requirements.txt
   ```
3) Running the Application Launch the Application: Open a terminal or command prompt. Navigate to the project directory (codes folder). Run the following command:
   ```bash
   python kNN.py
   ```

## How It Works

1) Image Upload: Begin by uploading an image of yourself or the subject. This image is used as the reference point for the recommendation system.

2) Feature Analysis: The system then analyzes the uploaded image to accurately extract key features:
    - Skin Tone Extraction: Utilizing advanced computer vision techniques, the system identifies and analyzes the facial region to determine the skin tone, converting it to the HSV color space for precise color representation.
    - Body Proportion Calculation: Employing MoveNet Thunder, the system extracts critical keypoints to calculate and understand the subject's body proportions accurately.

3) Personalized Recommendations: Based on the extracted skin tone and body proportions, the system searches the dataset to identify images with the closest matching features.

4) Fashion Inspiration: The recommended images, showcasing individuals with similar physical attributes, are displayed. You can use these images as a style reference, observing and drawing inspiration from the fashion choices and clothing styles that best complement similar skin tones and body proportions.

Examples of Output:

  ![image](https://github.com/alexy208/Capstone-Project/assets/126884588/051b4f7c-b441-4d54-8039-9e1a00861c2c)
  ![image](https://github.com/alexy208/Capstone-Project/assets/126884588/620da537-d460-4361-b6e6-212dc3f110a4)
  ![image](https://github.com/alexy208/Capstone-Project/assets/126884588/55dd6af0-4894-43c5-8719-b53db8e67625)

For a comprehensive understanding of this project, including its design philosophy, development process, and technical specifications, please refer to our detailed project report. The report offers in-depth insights into the project's creation and can serve as a valuable resource for those interested in the finer details or the educational aspects of this project.

You can access the report here: [Fashion Recommending System Report](https://github.com/alexy208/Capstone-Project/blob/ce476b7f7e58cf038ce2f1a78a7929419a399829/Capstone%20Thesis_%20TP072780%20KOH%20JIA%20YI.docx)

