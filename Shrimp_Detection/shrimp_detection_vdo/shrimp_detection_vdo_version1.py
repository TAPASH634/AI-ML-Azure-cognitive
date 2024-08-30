import cv2
import requests
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageFilter, ImageEnhance
import random
from prettytable import PrettyTable
import numpy as np
import math
# Azure Custom Vision Prediction URL and Key
PREDICTION_URL = "https://bariflolabscustomvision-prediction.cognitiveservices.azure.com/customvision/v3.0/Prediction/d959f2f1-9c20-4f31-881a-f44e0673e8c0/detect/iterations/Iteration1/image"
KEY = "feb492932e424ae4a152dc509b6d1bb4"

PIXEL_OF_REFERENCE = 523.71  # Values of the reference object
REFERENCE_LENGTH_CM = 11.858  # 6.858
REFERENCE_WEIGHT_GM = 2.8
PIXELS_PER_CM = PIXEL_OF_REFERENCE / REFERENCE_LENGTH_CM

# Headers for the request
headers = {
    "Content-Type": "application/octet-stream",
    "Prediction-Key": KEY
}

def random_deep_color():
    """Generate a random deep color in RGB format."""
    return (random.uniform(0, 0.5), random.uniform(0, 0.5), random.uniform(0, 0.5))

def sharpen_image(image):
    """Sharpen the image to reduce blur."""
    return image.filter(ImageFilter.SHARPEN)

def enhance_image(image):
    """Enhance image by adjusting contrast, sharpness, and brightness."""
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.5)
    
    # Enhance sharpness
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(2.0)
    
    # Enhance brightness
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(1.2)
    
    return image

def stabilize_frame(frame, previous_frame):
    """Stabilize the frame by comparing it to the previous one."""
    if previous_frame is None:
        return frame
    else:
        # Apply optical flow to stabilize the frame
        prev_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        height, width = flow.shape[:2]
        remap = np.float32([[[j + flow[i, j, 0], i + flow[i, j, 1]] for j in range(width)] for i in range(height)])
        stabilized_frame = cv2.remap(frame, remap, None, cv2.INTER_LINEAR)
        return stabilized_frame

def detect_shrimp(image_path, output_image_path):
    """Detect shrimp in the image and annotate it."""
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()

    response = requests.post(PREDICTION_URL, headers=headers, data=image_data)

    if response.status_code == 200:
        predictions = response.json()["predictions"]
        print("Detected objects:")

        image = Image.open(image_path)
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        ax = plt.gca()

        shrimp_data = []
        unique_id_counter = 1  # Initialize a counter for unique IDs

        for prediction in predictions:
            probability = prediction['probability']

            if probability > 0.95:
                bounding_box = prediction['boundingBox']
                print(f"Probability: {probability:.2%}")

                left = bounding_box['left'] * image.width
                top = bounding_box['top'] * image.height
                width = bounding_box['width'] * image.width
                height = bounding_box['height'] * image.height

                diagonal_pixels = math.sqrt(width**2 + height**2)
                diagonal_cm = diagonal_pixels / PIXELS_PER_CM
                print(f"Bounding Box Diagonal: {diagonal_pixels:.2f} pixels, {diagonal_cm:.2f} cm")
                shrimp_weight = (diagonal_cm / REFERENCE_LENGTH_CM) * REFERENCE_WEIGHT_GM
                print(f"Estimated Shrimp Weight: {shrimp_weight:.2f} gm")

                shrimp_data.append((unique_id_counter, diagonal_cm, shrimp_weight))
                unique_id_counter += 1

                color = random_deep_color()
                rect = patches.Rectangle((left, top), width, height, linewidth=2, edgecolor=color, facecolor='none')
                ax.add_patch(rect)

                plt.text(left, top - 10, f"ID: {unique_id_counter - 1}\nProb: {probability:.2%}\nSize: {diagonal_cm:.2f} cm\nWeight: {shrimp_weight:.2f} gm",
                         color=color, fontsize=12, weight='bold')

        plt.axis("off")
        plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
        print(f"Annotated image saved to: {output_image_path}")
        plt.show()

        print("\nShrimp Sizes and Weights:")
        table = PrettyTable()
        table.field_names = ["Unique ID", "Size (cm)", "Weight (gm)", "Percentage Error (%)"]
        average_size = np.mean([data[1] for data in shrimp_data])
        average_weight = np.mean([data[2] for data in shrimp_data])

        for shrimp in shrimp_data:
            size_error = abs(shrimp[1] - average_size) / average_size * 100
            weight_error = abs(shrimp[2] - average_weight) / average_weight * 100
            table.add_row([shrimp[0], shrimp[1], shrimp[2], f"Size: {size_error:.2f}%, Weight: {weight_error:.2f}%"])

        print(table)

    else:
        print(f"Failed to make prediction: {response.status_code}")
        print(response.json())

def process_video(video_path, output_folder):
    """Convert video to frames, enhance them, stabilize, and detect shrimp."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(2 * frame_rate)
    frame_count = 0
    success = True
    previous_frame = None

    while success:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count * frame_interval)
        success, frame = cap.read()

        if not success:
            break

        # Stabilize the frame
        stabilized_frame = stabilize_frame(frame, previous_frame)
        previous_frame = stabilized_frame.copy()

        # Convert the frame to a PIL Image for enhancement
        image = Image.fromarray(cv2.cvtColor(stabilized_frame, cv2.COLOR_BGR2RGB))
        enhanced_image = enhance_image(image)

        # Save the enhanced frame
        frame_file_path = os.path.join(output_folder, f"frame_{frame_count:04d}_6.jpeg")
        enhanced_image.save(frame_file_path)

        # Detect shrimp in the enhanced image
        detect_shrimp(frame_file_path, frame_file_path.replace(".jpeg", "_detected.jpeg"))

        frame_count += 1

    cap.release()
    print(f"Processed {frame_count} frames from the video.")

if __name__ == "__main__":
    # Replace with the path to your video
    video_path = "/home/jyoti/Documents/shrimp_vdo_13.mp4"
    output_folder = "/home/jyoti/Documents/Detected_shrimp_vdo_frames"

    process_video(video_path, output_folder)
