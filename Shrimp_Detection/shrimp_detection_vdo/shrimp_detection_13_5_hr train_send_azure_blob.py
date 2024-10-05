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
from azure.storage.blob import BlobServiceClient

# Azure Custom Vision Prediction URL and Key
PREDICTION_URL = "https://bariflocustomvision-prediction.cognitiveservices.azure.com/customvision/v3.0/Prediction/300ad940-da23-4789-864d-ee6f8dd69e2d/detect/iterations/Iteration3/image"
KEY = "9fd8c8196f1d479380faa10c0de24c05"

PIXEL_OF_REFERENCE = 238.65  # Values of the reference object
REFERENCE_LENGTH_CM = 5.858
PIXELS_PER_CM = PIXEL_OF_REFERENCE / REFERENCE_LENGTH_CM

# Headers for the request
headers = {
    "Content-Type": "application/octet-stream",
    "Prediction-Key": KEY
}

# Azure Blob Storage SAS Token and URL
BLOB_SAS_URL = "https://checktray.blob.core.windows.net/shrimpdata?sp=racwli&st=2024-10-01T04:51:15Z&se=2024-12-31T12:51:15Z&sv=2022-11-02&sr=c&sig=Uug2oLDr05I7WAgmzBE91ymnguoadWw6zH8tyv%2BZDy4%3D"
BLOB_SAS_TOKEN = "sp=racwli&st=2024-10-01T04:51:15Z&se=2024-12-31T12:51:15Z&sv=2022-11-02&sr=c&sig=Uug2oLDr05I7WAgmzBE91ymnguoadWw6zH8tyv%2BZDy4%3D"
CONTAINER_NAME = "shrimpdata"

def upload_to_blob(file_path, blob_name):
    """Upload a file to Azure Blob Storage."""
    try:
        blob_service_client = BlobServiceClient(account_url=BLOB_SAS_URL, credential=BLOB_SAS_TOKEN)
        blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=blob_name)

        with open(file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
        
        print(f"File {blob_name} uploaded successfully to Blob Storage.")
    except Exception as e:
        print(f"Error uploading {blob_name} to Blob Storage: {e}")

def random_deep_color():
    """Generate a random deep color in RGB format."""
    return (random.uniform(0, 0.5), random.uniform(0, 0.5), random.uniform(0, 0.5))  # Values closer to 0 for deeper colors

def sharpen_image(image):
    """Sharpen the image to reduce blur."""
    return image.filter(ImageFilter.SHARPEN)

def enhance_image(image, brightness_factor=1.2, contrast_factor=1.5, sharpness_factor=2.0, detail_filter=True):
    """
    Enhance the input image by adjusting brightness, contrast, sharpness, and applying a detail filter.
    """
    enhancer = ImageEnhance.Brightness(image)
    image_enhanced = enhancer.enhance(brightness_factor)

    enhancer = ImageEnhance.Contrast(image_enhanced)
    image_enhanced = enhancer.enhance(contrast_factor)

    enhancer = ImageEnhance.Sharpness(image_enhanced)
    image_enhanced = enhancer.enhance(sharpness_factor)

    if detail_filter:
        image_enhanced = image_enhanced.filter(ImageFilter.DETAIL)

    return image_enhanced

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

            if probability > 0.96:
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
    """Convert video to frames, enhance them, detect shrimp, and upload images to Azure Blob Storage."""
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
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count * frame_interval)
        success, frame = cap.read()

        if not success:
            break

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        enhanced_image = sharpen_image(image)
        frame_file_path = os.path.join(output_folder, f"frame_{frame_count:04d}_6.jpeg")
        enhanced_image.save(frame_file_path)

        shrimp_data = detect_shrimp(frame_file_path, frame_file_path.replace(".jpeg", "_detected.jpeg"))
        all_shrimp_data.extend(shrimp_data)

        # Upload detected images to Blob Storage
        blob_name = f"frame_{frame_count:04d}_detected.jpeg"
        upload_to_blob(frame_file_path.replace(".jpeg", "_detected.jpeg"), blob_name)

        frame_count += 1

    cap.release()
    print(f"Processed {frame_count} frames from the video.")

if __name__ == "__main__":
    video_path = "/home/jyoti/Documents/shrimp_vdo_60.mp4"
    output_folder = "/home/jyoti/Documents/Detected_shrimp_test_66"
    process_video(video_path, output_folder)
