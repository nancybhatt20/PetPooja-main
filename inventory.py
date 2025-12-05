import cv2
import torch
import os
import csv
from ultralytics import YOLO
import datetime

# Load pre-trained YOLO model (use a custom-trained model for better accuracy on ingredients)
model = YOLO('yolov8n.pt')  # Replace with your trained model for food detection

# Define dataset and output CSV file
inventory_folder = "inventory_images"  # Folder containing images of kitchen shelves or refrigerator
csv_filename = "inventory_log.csv"

# Ensure CSV file exists and has a header
if not os.path.exists(csv_filename) or os.stat(csv_filename).st_size == 0:
    with open(csv_filename, 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['Timestamp', 'Image Path', 'Detected Items'])

# Process each image in the inventory folder
for image_name in os.listdir(inventory_folder):
    image_path = os.path.join(inventory_folder, image_name)
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        continue
    
    # Run YOLOv8 inference
    results = model(image)
    
    # Extract detected objects
    detected_items = set()
    for detection in results[0].boxes:
        item_class = model.names[int(detection.cls)]  # Get item name
        detected_items.add(item_class)
    
    # Log detected items
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(csv_filename, 'a', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow([timestamp, image_path, ', '.join(detected_items)])
    
    print(f"Processed {image_name}: {', '.join(detected_items)}")

print(f"Inventory log saved to {csv_filename}")
