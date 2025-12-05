import cv2
import torch
import csv
import os
import matplotlib.pyplot as plt
from datetime import datetime
from ultralytics import YOLO
import pandas as pd

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Dataset and CSV file paths
dataset_folder = "human detection dataset"
csv_filename = "human_detection_results.csv"

# Ensure CSV file exists and has a header
if not os.path.exists(csv_filename) or os.stat(csv_filename).st_size == 0:
    with open(csv_filename, 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['Timestamp', 'Weekday', 'Image Path', 'Detected People Count', 'Estimated People Count'])

# Process each image and detect people
for label in ['0', '1']:  # Assuming '0' and '1' are class folders
    image_folder = os.path.join(dataset_folder, label)
    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            continue
        
        # Run YOLOv8 inference
        results = model(image)
        
        # Count people detected (class 0 in COCO dataset is 'person')
        people_count = sum(1 for detection in results[0].boxes if detection.cls == 0)
        
        # Get current weekday
        weekday = datetime.now().strftime("%A")
        
        # Log results
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(csv_filename, 'a', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([timestamp, weekday, image_path, people_count, ''])
        
        print(f"Processed {image_name}: {people_count} people detected on {weekday}")

print(f"Detection results saved to {csv_filename}")

# Read CSV and predict today's estimated people count
weekday_counts = {}
with open(csv_filename, 'r') as f:
    csv_reader = csv.reader(f)
    next(csv_reader)  # Skip header
    for row in csv_reader:
        if row[3].isdigit():  # Ensure the value is a valid number
            day, count = row[1], int(row[3])
            if day in weekday_counts:
                weekday_counts[day].append(count)
            else:
                weekday_counts[day] = [count]

today = datetime.now().strftime("%A")
estimated_count = "No Data"
if today in weekday_counts and len(weekday_counts[today]) > 0:
    estimated_count = round(sum(weekday_counts[today]) / len(weekday_counts[today]))
    print(f"Estimated people count for {today}: {estimated_count}")

# Store estimated count in CSV
with open(csv_filename, 'a', newline='') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), today, "Estimate", "", estimated_count])

print(f"Estimated people count for {today} saved to {csv_filename}")

# Generate graphical representation
weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
actual_counts = [sum(weekday_counts.get(day, [0])) / max(1, len(weekday_counts.get(day, []))) for day in weekdays]

plt.figure(figsize=(10, 5))
plt.bar(weekdays, actual_counts, color='blue', alpha=0.6, label='Actual Count')
if today in weekdays:
    plt.bar(today, estimated_count, color='red', alpha=0.6, label='Predicted Count')

plt.xlabel("Weekday")
plt.ylabel("People Count")
plt.title("People Count Per Weekday")
plt.legend()
plt.xticks(rotation=45)
plt.show()
