import cv2
import os

# Parameters
input_video = 'crablet_7.mp4'  # Path to the input video file
output_folder = 'Crab_dataset_1'  # Folder to save extracted frames
# frame_duration = 1 # Duration of each frame in seconds

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Open the video file
cap = cv2.VideoCapture(input_video)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# Calculate the number of frames to skip to get frames at the specified duration
# frame = int(fps)
current_frame = 0
frame_number = 0

while True:
    # Read the next frame
    ret, frame = cap.read()
    
    if not ret:
        break  # End of video

    # Save the frame if it's one of the desired frames
    # if current_frame % frame_skip == 0:
    output_path = os.path.join(output_folder, f'frame_10_{frame_number:04d}.png')
    cv2.imwrite(output_path, frame)
    frame_number += 1

    current_frame += 1

# Release video capture object and close windows
cap.release()
cv2.destroyAllWindows()

print(f'Frames saved in folder: {output_folder}')