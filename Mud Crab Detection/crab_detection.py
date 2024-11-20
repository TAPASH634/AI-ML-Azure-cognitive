"""
 Author : Tapash Ranjan Nandi
"""
import cv2
import requests
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageFilter
import random
# from prettytable import prettyTable

# Azure Custom Vision Prediction URL and Key
PREDICTION_URL = "https://mudcrab-prediction.cognitiveservices.azure.com/customvision/v3.0/Prediction/35bcb58a-1479-4ece-9559-b9064b17903a/detect/iterations/Iteration1/image"
KEY = "GFKIgNWeRHV1eG7bCM4wnXHNHOc8xzYTzAxIqcV6xAjy6ymaZIV5JQQJ99AKACGhslBXJ3w3AAAIACOGY695"

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

def detect_crab(image_path, output_image_path):
    """Detect crab in the image and annotate it."""
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()

    response = requests.post(PREDICTION_URL, headers=headers, data=image_data)

    if response.status_code == 200:
        predictions = response.json()["predictions"]

        image = Image.open(image_path)
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        ax = plt.gca()

        for prediction in predictions:
            probability = prediction['probability']

            if probability > 0.90:
                bounding_box = prediction['boundingBox']
                left = bounding_box['left'] * image.width
                top = bounding_box['top'] * image.height
                width = bounding_box['width'] * image.width
                height = bounding_box['height'] * image.height

                color = random_deep_color()
                rect = patches.Rectangle((left, top), width, height, linewidth=2, edgecolor=color, facecolor='none')
                ax.add_patch(rect)

                plt.text(left, top - 10, f"Prob: {probability:.2%}",
                         color=color, fontsize=12, weight='bold')

        plt.axis("off")
        plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
        print(f"Annotated image saved to: {output_image_path}")
        plt.show()

    else:
        print(f"Failed to make prediction: {response.status_code}")
        print(response.json())

def process_video(video_path, output_folder):
    """Convert video to frames, enhance them, and detect crab."""
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

        # Detect crab in the enhanced image and annotate it
        detect_crab(frame_file_path, frame_file_path.replace(".jpeg", "_detected.jpeg"))

        frame_count += 1
        
        
    cap.release()
    print(f"Processed {frame_count} frames from the video.")

if __name__ == "__main__":
    # Replace with the path to your video
    video_path = "/home/jyoti/Documents/mudcrab_detection/crablet_10.mp4"
    output_folder = "/home/jyoti/Documents/mudcrab_detection/Detected_crab_test_1"

    process_video(video_path, output_folder)