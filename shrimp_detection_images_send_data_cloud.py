import requests
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import math
import random
from prettytable import PrettyTable  # Importing PrettyTable for tabular representation
import mysql.connector  # Importing MySQL connector
from azure.storage.blob import BlobServiceClient  # Importing Azure Blob Service Client

# Azure Custom Vision Prediction URL and Key
PREDICTION_URL = "https://bariflolabscustomvision-prediction.cognitiveservices.azure.com/customvision/v3.0/Prediction/d959f2f1-9c20-4f31-881a-f44e0673e8c0/detect/iterations/Iteration1/image"
KEY = "feb492932e424ae4a152dc509b6d1bb4"

# Azure Blob Storage credentials
BLOB_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=shrimpdata;AccountKey=BnGYCGBxk9bgIltZr2KkzSJWjVlLBSpdoOiHcDKVCQZXJv+qN1b/WZKWUy9gwr+DlzCIPFpqO+PX+AStRp+UQA==;EndpointSuffix=core.windows.net"
BLOB_CONTAINER_NAME = "container1"  # Update this to match your actual container name

# container_name = "container1"
# MySQL database credentials
DB_HOST = 'shrimpdata.mysql.database.azure.com'
DB_PORT = 3306
DB_USER = 'Bariflolabs'
DB_PASSWORD = 'Bfl@2024'
DB_NAME = 'shrimp_data'

PIXEL_OF_REFERENCE = 523.71  # Values of the reference object
REFERENCE_LENGTH_CM = 6.858
REFERENCE_WEIGHT_GM = 2.8
PIXELS_PER_CM = PIXEL_OF_REFERENCE / REFERENCE_LENGTH_CM

# Headers for the request
headers = {
    "Content-Type": "application/octet-stream",
    "Prediction-Key": KEY
}

def random_deep_color():
    """Generate a random deep color in RGB format."""
    return (random.uniform(0, 0.5), random.uniform(0, 0.5), random.uniform(0, 0.5))  # Values closer to 0 for deeper colors

def upload_image_to_blob(image_path, image_name):
    """Upload the detected image to Azure Blob Storage."""
    blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)
    blob_client = blob_service_client.get_blob_client(container=BLOB_CONTAINER_NAME, blob=image_name)

    with open(image_path, "rb") as image_file:
        blob_client.upload_blob(image_file, overwrite=True)

    # Construct the URL of the uploaded image
    image_url = f"https://shrimpdata.blob.core.windows.net/{BLOB_CONTAINER_NAME}/{image_name}"
    return image_url

def insert_shrimp_data(image_id, image_url, shrimp_data):
    """Insert shrimp detection data into the MySQL database."""
    connection = mysql.connector.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME
    )

    try:
        with connection.cursor() as cursor:
            for unique_id, size, weight in shrimp_data:
                sql = "INSERT INTO shrimp_data_table (Image_id, Image_url, unique_shrimp_id, shrimp_size) VALUES (%s, %s, %s, %s)"
                cursor.execute(sql, (image_id, image_url, str(unique_id), size))
            connection.commit()
    finally:
        connection.close()

def detect_shrimp(image_path, output_image_path):
    # Open the image file
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()

    # Make the prediction request
    response = requests.post(PREDICTION_URL, headers=headers, data=image_data)

    # Check if the request was successful
    if response.status_code == 200:
        predictions = response.json()["predictions"]
        print("Detected objects:")

        # Load the image for visualization
        image = Image.open(image_path)
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        ax = plt.gca()

        # Create a list to store shrimp data
        shrimp_data = []
        unique_id_counter = 1  # Initialize a counter for unique IDs

        for prediction in predictions:
            probability = prediction['probability']

            # Only draw bounding boxes for predictions above 95%
            if probability > 0.95:
                bounding_box = prediction['boundingBox']
                print(f"Probability: {probability:.2%}")

                # Calculate bounding box dimensions in pixels
                left = bounding_box['left'] * image.width
                top = bounding_box['top'] * image.height
                width = bounding_box['width'] * image.width
                height = bounding_box['height'] * image.height

                # Calculate the diagonal of the bounding box in pixels
                diagonal_pixels = math.sqrt(width ** 2 + height ** 2)
                # Convert diagonal from pixels to centimeters
                diagonal_cm = diagonal_pixels / PIXELS_PER_CM
                print(f"Bounding Box Diagonal: {diagonal_pixels:.2f} pixels, {diagonal_cm:.2f} cm")
                shrimp_weight = (diagonal_cm / REFERENCE_LENGTH_CM) * REFERENCE_WEIGHT_GM
                print(f"Estimated Shrimp Weight: {shrimp_weight:.2f} gm")

                # Store the shrimp data with a unique ID
                shrimp_data.append((unique_id_counter, diagonal_cm, shrimp_weight))
                unique_id_counter += 1  # Increment the unique ID counter

                # Generate a random deep color for the bounding box
                color = random_deep_color()

                # Create a rectangle patch for each detected object
                rect = patches.Rectangle((left, top), width, height, linewidth=2, edgecolor=color, facecolor='none')
                ax.add_patch(rect)

                # Add label, unique ID, confidence score, size, and weight
                plt.text(left, top - 10, f"ID: {unique_id_counter - 1}\nProb: {probability:.2%}\nSize: {diagonal_cm:.2f} cm\nWeight: {shrimp_weight:.2f} gm",
                         color=color, fontsize=12, weight='bold')

        plt.axis("off")
        # Save the annotated image
        plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
        print(f"Annotated image saved to: {output_image_path}")
        plt.show()

        # Upload the annotated image to Azure Blob Storage
        image_name = output_image_path.split("/")[-1]
        image_url = upload_image_to_blob(output_image_path, image_name)
        print(f"Uploaded image to blob storage. Image URL: {image_url}")

        # Insert shrimp detection data into MySQL database
        image_id = f"im{image_name.split('_')[1]}"  # Example: im1, im2, etc.
        insert_shrimp_data(image_id, image_url, shrimp_data)

        # Print the table of shrimp sizes and weights
        print("\nShrimp Sizes and Weights:")
        table = PrettyTable()
        table.field_names = ["Unique ID", "Size (cm)", "Weight (gm)"]
        for shrimp in shrimp_data:
            table.add_row(shrimp)
        print(table)

    else:
        print(f"Failed to make prediction: {response.status_code}")
        print(response.json())

if __name__ == "__main__":
    # Replace with the path to your image
    image_path = "/home/jyoti/Documents/test_16_shrimp.jpeg"
    output_image_path = "/home/jyoti/Documents/Detected_shrimp_with_weight/test_16_shrimp.jpeg"

    detect_shrimp(image_path, output_image_path)
