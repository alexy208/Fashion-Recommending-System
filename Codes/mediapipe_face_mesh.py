import cv2
import math
#import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
#from body_pro_extract import get_body_proportion


DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480

def resize_and_show(image):
    h, w = image.shape[:2]
    if h < w:
        img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h / (w / DESIRED_WIDTH))))
    else:
        img = cv2.resize(image, (math.floor(w / (h / DESIRED_HEIGHT)), DESIRED_HEIGHT))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')  # To hide axis values
    plt.show()

# Read the image using OpenCV
image = cv2.imread('C:/Users/Alex/Downloads/capstone selfie test.jpg')
image = cv2.imread('../Dataset - imgpool fullbody/Women/9333.jpg')
# Get the dimensions of the image
if image is not None:
    # Get the dimensions of the image
    h, w, _ = image.shape

    # Calculate the half of the height
    half_h = h // 3

    # Crop the image from top to half
    # The crop syntax is image[start_row:end_row, start_col:end_col]
    image = image[0:half_h, 50:w-50]

plt.imshow(image[..., ::-1])  


mp_face_mesh = mp.solutions.face_mesh

# Load drawing_utils and drawing_styles
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Run MediaPipe Face Mesh
with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        refine_landmarks=True,
        max_num_faces=1,
        min_detection_confidence=0.1) as face_mesh:

    # Convert the BGR image to RGB and process it with MediaPipe Face Mesh
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Draw face landmarks
    if results.multi_face_landmarks:
        annotated_image = image.copy()
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())
        resize_and_show(annotated_image)
    else:
        print("No face landmarks detected in the image.")
