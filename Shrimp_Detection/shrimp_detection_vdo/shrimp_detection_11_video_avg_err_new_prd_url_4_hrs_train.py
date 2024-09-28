"""Author : Tapash Ranjan Nandi
   """
import cv2
import requests
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageFilter , ImageEnhance
import math
import random
from prettytable import PrettyTable

# Azure Custom Vision Prediction URL and Key
PREDICTION_URL = "https://bariflocustomvision-prediction.cognitiveservices.azure.com/customvision/v3.0/Prediction/300ad940-da23-4789-864d-ee6f8dd69e2d/detect/iterations/Iteration2/image"
KEY = "9fd8c8196f1d479380faa10c0de24c05"

PIXEL_OF_REFERENCE = 238.65  # Values of the reference object
REFERENCE_LENGTH_CM = 5.858  # 6.858
PIXELS_PER_CM = PIXEL_OF_REFERENCE / REFERENCE_LENGTH_CM

# Headers for the request
headers = {
    "Content-Type": "application/octet-stream",
    "Prediction-Key": KEY
}

def random_deep_color():
    """Generate a random deep color in RGB format."""
    return (random.uniform(0, 0.5), random.uniform(0, 0.5), random.uniform(0, 0.5))  # Values closer to 0 for deeper colors

def sharpen_image(image):
    # """Sharpen the image to reduce blur."""
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

# Example usage:
# image = Image.open("input_image.jpeg")
# enhanced_image = enhance_image(image)
# enhanced_image.show()  # Display the enhanced image
# enhanced_image.save("enhanced_image.jpeg")  # Save the enhanced image

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

                shrimp_data.append((unique_id_counter, diagonal_cm))
                unique_id_counter += 1

                color = random_deep_color()
                rect = patches.Rectangle((left, top), width, height, linewidth=2, edgecolor=color, facecolor='none')
                ax.add_patch(rect)

                plt.text(left, top - 10, f"ID: {unique_id_counter - 1}\nProb: {probability:.2%}\nSize: {diagonal_cm:.2f} cm",
                         color=color, fontsize=12, weight='bold')
                
        plt.axis("off")
        plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
        print(f"Annotated image saved to: {output_image_path}")
        plt.show()

        print("\nShrimp Sizes:")
        table = PrettyTable()
        table.field_names = ["Unique ID", "Size (cm)"]
        for shrimp in shrimp_data:
            table.add_row(shrimp)
        print(table)

        return shrimp_data  # Return the shrimp data for further processing

    else:
        print(f"Failed to make prediction: {response.status_code}")
        print(response.json())
        return []

def process_video(video_path, output_folder):
    """Convert video to frames, enhance them, and detect shrimp."""
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

    all_shrimp_data = []  # List to store all shrimp data across frames

    while success:
        # Set the frame position to extract one frame every 2 seconds
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count * frame_interval)
        success, frame = cap.read()

        if not success:
            break

        # Convert the frame to a PIL Image for enhancement
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # enhanced_image = enhance_image(image, brightness_factor=1.3, contrast_factor=1.8, sharpness_factor=2.2, detail_filter=True)
        enhanced_image = sharpen_image(image)
        # Save the enhanced frame
        frame_file_path = os.path.join(output_folder, f"frame_{frame_count:04d}_6.jpeg")
        enhanced_image.save(frame_file_path)

        # Detect shrimp in the enhanced image and collect the data
        shrimp_data = detect_shrimp(frame_file_path, frame_file_path.replace(".jpeg", "_detected.jpeg"))
        all_shrimp_data.extend(shrimp_data)

        frame_count += 1

    cap.release()
    print(f"Processed {frame_count} frames from the video.")

    if all_shrimp_data:
        # Calculate the average size
        total_size = sum(shrimp[1] for shrimp in all_shrimp_data)
        avg_size = total_size / len(all_shrimp_data)

        print(f"\nAverage Shrimp Size: {avg_size:.2f} cm")

        # Segregate shrimp data based on the error percentage
        large_shrimp = []
        small_shrimp = []
        for shrimp in all_shrimp_data:
            unique_id, size_cm = shrimp
            error_percentage = abs((size_cm - avg_size) / avg_size) * 100
            if error_percentage > 20:
                if size_cm > avg_size:
                    large_shrimp.append((unique_id, size_cm))
                else:
                    small_shrimp.append((unique_id, size_cm))

        # Print the table for large shrimp
        if large_shrimp:
            # print("\nShrimp Larger than Average with >20% Error:")
            print("\nShrimp Larger than Average :")

            large_table = PrettyTable()
            large_table.field_names = ["Unique ID", "Size (cm)"]
            for shrimp in large_shrimp:
                large_table.add_row(shrimp)
            print(large_table)

        # Print the table for small shrimp
        if small_shrimp:
            # print("\nShrimp Smaller than Average with >20% Error:")
            print("\nShrimp Smaller than Average :")

            small_table = PrettyTable()
            small_table.field_names = ["Unique ID", "Size (cm)"]
            for shrimp in small_shrimp:
                small_table.add_row(shrimp)
            print(small_table)

    else:
        print("No shrimp detected in the video.")

if __name__ == "__main__":
    # Replace with the path to your video
    video_path = "/home/jyoti/Documents/DOC_ 28__2024-08-26 at 12.42.34 PM_pond_4.mp4"
    output_folder = "/home/jyoti/Documents/Detected_shrimp_test_39"

    process_video(video_path, output_folder)
