"""Author : Tapash Ranjan Nandi
   """
import cv2
import requests
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageFilter, ImageEnhance
import math
import random
from prettytable import PrettyTable

# Azure Custom Vision Prediction URL and Key
PREDICTION_URL = "https://humanspermdetection-prediction.cognitiveservices.azure.com/customvision/v3.0/Prediction/f8487da5-df88-4e03-9652-5ac544d17aed/detect/iterations/Iteration1/image"
KEY = "a95cd4c9254140ee93be024fd2e3a471"

# Headers for the request
headers = {
    "Content-Type": "application/octet-stream",
    "Prediction-Key": KEY
}

def random_deep_color():
    """Generate a random deep color in RGB format."""
    return (random.uniform(0, 0.5), random.uniform(0, 0.5), random.uniform(0, 0.5))  # Values closer to 0 for deeper colors

def sharpen_image(image):
    """Sharpen the image to reduce blur."""
    return image.filter(ImageFilter.SHARPEN)

def enhance_image(image, brightness_factor=1.2, contrast_factor=1.5, sharpness_factor=2.0, detail_filter=True):
    """
    Enhance the input image by adjusting brightness, contrast, sharpness, and applying a detail filter.

    :param image: PIL Image object to enhance.
    :param brightness_factor: Factor to adjust brightness (default: 1.2).
    :param contrast_factor: Factor to adjust contrast (default: 1.5).
    :param sharpness_factor: Factor to adjust sharpness (default: 2.0).
    :param detail_filter: Apply a detail filter to the image if True (default: True).
    :return: Enhanced PIL Image object.
    """
    # Adjust brightness
    enhancer = ImageEnhance.Brightness(image)
    image_enhanced = enhancer.enhance(brightness_factor)

    # Adjust contrast
    enhancer = ImageEnhance.Contrast(image_enhanced)
    image_enhanced = enhancer.enhance(contrast_factor)

    # Adjust sharpness
    enhancer = ImageEnhance.Sharpness(image_enhanced)
    image_enhanced = enhancer.enhance(sharpness_factor)

    # Apply detail filter (optional)
    if detail_filter:
        image_enhanced = image_enhanced.filter(ImageFilter.DETAIL)

    return image_enhanced

def detect_sperm(image_path, output_image_path):
    """Detect sperm in the image and annotate it."""
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

        sperm_data = []
        unique_id_counter = 1  # Initialize a counter for unique IDs

        for prediction in predictions:
            probability = prediction['probability']

            if probability > 0.96:
                bounding_box = prediction['boundingBox']
                print(f"Probability: {probability:.2%}")

                left = bounding_box['left'] * image.width
                top = bounding_box['top'] * image.height
                width = bounding_box['width'] * image.width
                height = bounding_box['height'] * image.height

                diagonal_pixels = math.sqrt(width**2 + height**2)
                area = diagonal_pixels * width
                print(f"Bounding Box Diagonal: {area:.2f} pixels")

                sperm_data.append((unique_id_counter, area))
                unique_id_counter += 1

                color = random_deep_color()
                rect = patches.Rectangle((left, top), width, height, linewidth=2, edgecolor=color, facecolor='none')
                ax.add_patch(rect)

                plt.text(left, top - 10, f"ID: {unique_id_counter - 1}\nProb: {probability:.2%}\nArea: {area:.2f} px",
                         color=color, fontsize=12, weight='bold')
                
        plt.axis("off")
        plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
        print(f"Annotated image saved to: {output_image_path}")
        plt.show()

        print("\nsperm Sizes:")
        table = PrettyTable()
        table.field_names = ["Unique ID", "Size (px)"]
        for sperm in sperm_data:
            table.add_row(sperm)
        print(table)

        return sperm_data  # Return the sperm data for further processing

    else:
        print(f"Failed to make prediction: {response.status_code}")
        print(response.json())
        return []

def process_video(video_path, output_folder):
    """Convert video to frames, enhance them, and detect sperm."""
    # Check if the output folder exists; if not, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        print(f"The directory '{output_folder}' already exists. Proceeding with the existing directory.")

    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)  # Get the frame rate of the video
    frame_interval = int(2 * frame_rate)    # Set the interval to 2 seconds
    frame_count = 0
    success = True

    all_sperm_data = []  # List to store all sperm data across frames

    while success:
        # Set the frame position to extract one frame every 2 seconds
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count * frame_interval)
        success, frame = cap.read()

        if not success:
            break

        # Convert the frame to a PIL Image for enhancement
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        enhanced_image = sharpen_image(image)
        # Save the enhanced frame
        frame_file_path = os.path.join(output_folder, f"frame_{frame_count:04d}_6.jpeg")
        enhanced_image.save(frame_file_path)

        # Detect sperm in the enhanced image and collect the data
        sperm_data = detect_sperm(frame_file_path, frame_file_path.replace(".jpeg", "_detected.jpeg"))
        all_sperm_data.extend(sperm_data)

        frame_count += 1

    cap.release()
    print(f"Processed {frame_count} frames from the video.")

    if all_sperm_data:
        # Calculate the average size
        total_size = sum(sperm[1] for sperm in all_sperm_data)
        avg_size = total_size / len(all_sperm_data)

        print(f"\nAverage sperm Size: {avg_size:.2f} px")

        # Segregate sperm data based on the error percentage
        large_sperm = []
        small_sperm = []
        for sperm in all_sperm_data:
            unique_id, size_px = sperm
            error_percentage = abs((size_px - avg_size) / avg_size) * 100
            if error_percentage > 20:
                if size_px > avg_size:
                    large_sperm.append((unique_id, size_px))
                else:
                    small_sperm.append((unique_id, size_px))

        # Print the table for large sperm
        if large_sperm:
            print("\nSperm Larger than Average:")

            large_table = PrettyTable()
            large_table.field_names = ["Unique ID", "Size (px)"]
            for sperm in large_sperm:
                large_table.add_row(sperm)
            print(large_table)

        # Print the table for small sperm
        if small_sperm:
            print("\nSperm Smaller than Average:")

            small_table = PrettyTable()
            small_table.field_names = ["Unique ID", "Size (px)"]
            for sperm in small_sperm:
                small_table.add_row(sperm)
            print(small_table)

    else:
        print("No sperm detected in the video.")

if __name__ == "__main__":
    # Replace with the path to your video
    video_path = "/home/jyoti/Documents/sperm_detection/sperm_vdo_2.mp4"
    output_folder = "/home/jyoti/Documents/sperm_detection/Detected_sperm_test_66"

    process_video(video_path, output_folder)
