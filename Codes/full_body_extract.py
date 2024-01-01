import tensorflow as tf
import tensorflow_hub as hub
from tensorflow_docs.vis import embed
import numpy as np
import cv2

# Import matplotlib libraries
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches

# modules to display an animation using imageio.
import imageio
from IPython.display import HTML, display

import os
from shutil import copyfile

# Dictionary that maps from joint names to keypoint indices.
KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

# Maps bones to a matplotlib color name.
KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

def _keypoints_and_edges_for_display(keypoints_with_scores,
                                     height,
                                     width,
                                     keypoint_threshold=0.11):
  """Returns high confidence keypoints and edges for visualization.

  Args:
    keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
      the keypoint coordinates and scores returned from the MoveNet model.
    height: height of the image in pixels.
    width: width of the image in pixels.
    keypoint_threshold: minimum confidence score for a keypoint to be
      visualized.

  Returns:
    A (keypoints_xy, edges_xy, edge_colors) containing:
      * the coordinates of all keypoints of all detected entities;
      * the coordinates of all skeleton edges of all detected entities;
      * the colors in which the edges should be plotted.
  """
  keypoints_all = []
  keypoint_edges_all = []
  edge_colors = []
  num_instances, _, _, _ = keypoints_with_scores.shape
  for idx in range(num_instances):
    kpts_x = keypoints_with_scores[0, idx, :, 1]
    kpts_y = keypoints_with_scores[0, idx, :, 0]
    kpts_scores = keypoints_with_scores[0, idx, :, 2]
    kpts_absolute_xy = np.stack(
        [width * np.array(kpts_x), height * np.array(kpts_y)], axis=-1)
    kpts_above_thresh_absolute = kpts_absolute_xy[
        kpts_scores > keypoint_threshold, :]
    keypoints_all.append(kpts_above_thresh_absolute)

    for edge_pair, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
      if (kpts_scores[edge_pair[0]] > keypoint_threshold and
          kpts_scores[edge_pair[1]] > keypoint_threshold):
        x_start = kpts_absolute_xy[edge_pair[0], 0]
        y_start = kpts_absolute_xy[edge_pair[0], 1]
        x_end = kpts_absolute_xy[edge_pair[1], 0]
        y_end = kpts_absolute_xy[edge_pair[1], 1]
        line_seg = np.array([[x_start, y_start], [x_end, y_end]])
        keypoint_edges_all.append(line_seg)
        edge_colors.append(color)
  if keypoints_all:
    keypoints_xy = np.concatenate(keypoints_all, axis=0)
  else:
    keypoints_xy = np.zeros((0, 17, 2))

  if keypoint_edges_all:
    edges_xy = np.stack(keypoint_edges_all, axis=0)
  else:
    edges_xy = np.zeros((0, 2, 2))
  return keypoints_xy, edges_xy, edge_colors


def draw_prediction_on_image(
    image, keypoints_with_scores, crop_region=None, close_figure=False,
    output_image_height=None):
  """Draws the keypoint predictions on image.

  Args:
    image: A numpy array with shape [height, width, channel] representing the
      pixel values of the input image.
    keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
      the keypoint coordinates and scores returned from the MoveNet model.
    crop_region: A dictionary that defines the coordinates of the bounding box
      of the crop region in normalized coordinates (see the init_crop_region
      function below for more detail). If provided, this function will also
      draw the bounding box on the image.
    output_image_height: An integer indicating the height of the output image.
      Note that the image aspect ratio will be the same as the input image.

  Returns:
    A numpy array with shape [out_height, out_width, channel] representing the
    image overlaid with keypoint predictions.
  """
  height, width, channel = image.shape
  aspect_ratio = float(width) / height
  fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))
  # To remove the huge white borders
  fig.tight_layout(pad=0)
  ax.margins(0)
  ax.set_yticklabels([])
  ax.set_xticklabels([])
  plt.axis('off')

  im = ax.imshow(image)
  line_segments = LineCollection([], linewidths=(4), linestyle='solid')
  ax.add_collection(line_segments)
  # Turn off tick labels
  scat = ax.scatter([], [], s=60, color='#FF1493', zorder=3)

  (keypoint_locs, keypoint_edges,
   edge_colors) = _keypoints_and_edges_for_display(
       keypoints_with_scores, height, width)

  line_segments.set_segments(keypoint_edges)
  line_segments.set_color(edge_colors)
  if keypoint_edges.shape[0]:
    line_segments.set_segments(keypoint_edges)
    line_segments.set_color(edge_colors)
  if keypoint_locs.shape[0]:
    scat.set_offsets(keypoint_locs)

  if crop_region is not None:
    xmin = max(crop_region['x_min'] * width, 0.0)
    ymin = max(crop_region['y_min'] * height, 0.0)
    rec_width = min(crop_region['x_max'], 0.99) * width - xmin
    rec_height = min(crop_region['y_max'], 0.99) * height - ymin
    rect = patches.Rectangle(
        (xmin,ymin),rec_width,rec_height,
        linewidth=1,edgecolor='b',facecolor='none')
    ax.add_patch(rect)

  fig.canvas.draw()
  image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
  image_from_plot = image_from_plot.reshape(
      fig.canvas.get_width_height()[::-1] + (3,))
  plt.close(fig)
  if output_image_height is not None:
    output_image_width = int(output_image_height / height * width)
    image_from_plot = cv2.resize(
        image_from_plot, dsize=(output_image_width, output_image_height),
         interpolation=cv2.INTER_CUBIC)
  return image_from_plot

def to_gif(images, duration):
  """Converts image sequence (4D numpy array) to gif."""
  imageio.mimsave('./animation.gif', images, duration=duration)
  return embed.embed_file('./animation.gif')

def progress(value, max=100):
  return HTML("""
      <progress
          value='{value}'
          max='{max}',
          style='width: 100%'
      >
          {value}
      </progress>
  """.format(value=value, max=max))

# Load MoveNet Thunder model
module = hub.load("../models/")
input_size = 256

def movenet(input_image):
    
    """Runs detection on an input image.

    Args:
      input_image: A [1, height, width, 3] tensor represents the input image
        pixels. Note that the height/width should already be resized and match the
        expected input resolution of the model before passing into this function.

    Returns:
      A [1, 1, 17, 3] float numpy array representing the predicted keypoint
      coordinates and scores.
    """
    model = module.signatures['serving_default']

    # SavedModel format expects tensor type of int32.
    input_image = tf.cast(input_image, dtype=tf.int32)
    # Run model inference.
    outputs = model(input_image)
    # Output is a [1, 1, 17, 3] tensor.
    keypoints_with_scores = outputs['output_0'].numpy()
    return keypoints_with_scores
def calculate_proportions(keypoints_with_scores):
    keypoints = keypoints_with_scores[0, 0, :, :2]
    scores = keypoints_with_scores[0, 0, :, 2]
    threshold = 0.1
        
    # Ensure all required keypoints are present
    required_keypoints = [
        'left_shoulder', 'right_shoulder',
        'left_hip', 'right_hip',
        'left_ankle', 'right_ankle'
    ]
    if not all(scores[KEYPOINT_DICT[kpt]] > threshold for kpt in required_keypoints):
        return None, None, None
    
    # Create a mask for valid keypoints
    valid_mask = scores > threshold

    # Get the coordinates using the valid_mask
    left_shoulder = keypoints[KEYPOINT_DICT['left_shoulder']] if valid_mask[KEYPOINT_DICT['left_shoulder']] else None
    right_shoulder = keypoints[KEYPOINT_DICT['right_shoulder']] if valid_mask[KEYPOINT_DICT['right_shoulder']] else None
    left_hip = keypoints[KEYPOINT_DICT['left_hip']] if valid_mask[KEYPOINT_DICT['left_hip']] else None
    right_hip = keypoints[KEYPOINT_DICT['right_hip']] if valid_mask[KEYPOINT_DICT['right_hip']] else None
    left_ankle = keypoints[KEYPOINT_DICT['left_ankle']] if valid_mask[KEYPOINT_DICT['left_ankle']] else None
    right_ankle = keypoints[KEYPOINT_DICT['right_ankle']] if valid_mask[KEYPOINT_DICT['right_ankle']] else None

    # Calculate the midpoint between shoulders as the top of the torso
    top_of_torso = (left_shoulder + right_shoulder) / 2

    # Calculate the proportions
    shoulder_width = np.linalg.norm(right_shoulder - left_shoulder)
    hip_width = np.linalg.norm(right_hip - left_hip)

    body_height = np.linalg.norm(top_of_torso - left_hip) + np.linalg.norm(left_hip - left_ankle)
    leg_height = np.linalg.norm(left_hip - left_ankle)
    torso_height = np.linalg.norm(top_of_torso - left_hip)

    vertical_body_proportion = torso_height / leg_height
    horizontal_body_proportion = shoulder_width / hip_width
    leg_to_body_ratio = leg_height / body_height

    return vertical_body_proportion, horizontal_body_proportion, leg_to_body_ratio

def get_body_proportion(image_path):

    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image)


    # Resize and pad the image to keep the aspect ratio and fit the expected size.
    input_image = tf.expand_dims(image, axis=0)
    input_image = tf.image.resize_with_pad(input_image, input_size, input_size)
    input_image = tf.cast(input_image, dtype=tf.int32)  # cast to int32

    # Run model inference.
    keypoints_with_scores = movenet(input_image)
    vertical_body_proportion, horizontal_body_proportion, leg_to_body_ratio = calculate_proportions(keypoints_with_scores)
    
    if vertical_body_proportion is None and leg_to_body_ratio is None:
        print("Full body not detected in the image:", image_path)
        return None

    return vertical_body_proportion, horizontal_body_proportion, leg_to_body_ratio


def main(SOURCE_DIR,TARGET_DIR): 
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)
    
    def filter_and_save_images(src_dir, target_dir):
        """Filter out images with all necessary keypoints and save to another directory."""
        global counter
        counter = 0
        # Loop over every file in the directory
        for filename in os.listdir(src_dir):
            counter+=1
            if filename.endswith(".jpg"):  # Assuming all images are jpg, modify if there are other formats
                filepath = os.path.join(src_dir, filename)
                
                # Get body proportions for the image
                proportions = get_body_proportion(filepath)
                
                # If proportions are not None, save this image to the target directory
                if proportions is None:
                    continue
                target_filepath = os.path.join(target_dir, filename)
                copyfile(filepath, target_filepath)
    # Call the function
    filter_and_save_images(SOURCE_DIR, TARGET_DIR)

if __name__ == "__main__":
    # Define paths
    SOURCE_DIR = r"C:\Users\Alex\Desktop\Msc APU\Sem 3\Capstone\Dataset - imgpool\Female"
    TARGET_DIR = r"C:\Users\Alex\Desktop\Msc APU\Sem 3\Capstone\Dataset - imgpool fullbody\Women"
    
    main(SOURCE_DIR,TARGET_DIR)
    
# # Run model inference.
# keypoints_with_scores = movenet(input_image)

# # Calculate proportions
# vertical_body_proportion, horizontal_body_proportion, leg_to_body_ratio = calculate_proportions(keypoints_with_scores)

# print("Vertical Body Proportion: ", vertical_body_proportion)
# print("Horizontal Body Proportion: ", horizontal_body_proportion)
# print("Leg to Body Ratio: ", leg_to_body_ratio)


