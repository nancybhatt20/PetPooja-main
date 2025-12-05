import cv2
import pandas as pd
import numpy as np
import datetime
import os
from ultralytics import YOLO
from tracker import Tracker

# Load YOLO model
model = YOLO('yolov8s.pt')

# Define Entry & Exit areas
entry_area = [(312, 388), (289, 390), (474, 469), (497, 462)]  # Area 1 (Entry)
exit_area = [(279, 392), (250, 397), (423, 477), (454, 469)]   # Area 2 (Exit)

# Initialize tracker
tracker = Tracker()

# Store customer entry & exit records
entered_customers = {}  # Stores ID & timestamp when entering
leaved_customers = {}   # Stores ID & timestamp when leaving
customer_status = {}    # Tracks if a customer is inside

# Function to save data to CSV
def save_to_csv():
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.datetime.now().strftime("%H:%M:%S")
    csv_filename = f"customer_data_{current_date}.csv"

    # Count of total people who entered that day
    total_entries = len(entered_customers)
    summary_df = pd.DataFrame([[current_date, current_time, total_entries]], 
                               columns=["Date", "Timestamp", "Total_Entries"])

    # Save entry-exit records
    if not os.path.exists(csv_filename):
        summary_df.to_csv("daily_summary.csv", index=False)
    else:
        summary_df.to_csv("daily_summary.csv", mode="a", header=False, index=False)

    print(f"✅ Customer data and daily summary saved to CSV!")

# Function to check if a new day has started (12:00 AM)
def check_new_day():
    global entered_customers, leaved_customers, customer_status, csv_filename

    # Get current date
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")

    csv_filename = r"C:\Users\vansh\Downloads\peoplecounteryolov8-main\peoplecounteryolov8-main\customer_log.csv"
    # Check if CSV file exists and last recorded date is different
    if os.path.exists(csv_filename):
        df = pd.read_csv(csv_filename)
        if not df.empty:
            last_date = df.iloc[-1]["Entry_Time"].split(" ")[0]  # Extract last recorded date
            if last_date != current_date:
                save_to_csv()  # Save previous day's data before resetting
                entered_customers.clear()
                leaved_customers.clear()
                customer_status.clear()
                print("🔄 New day detected! Resetting customer count.")

frame_count = 0

# Open video file
cap = cv2.VideoCapture('peoplecount1.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    if frame_count % 2 != 0:
        continue  # Skip alternate frames for efficiency

    frame = cv2.resize(frame, (1020, 500))
    results = model.predict(frame)

    # Check if a new day has started
    check_new_day()

    # Extract bounding boxes
    detections = results[0].boxes.data
    df = pd.DataFrame(detections).astype("float")
    
    detected_people = []
    
    for _, row in df.iterrows():
        x1, y1, x2, y2, _, class_id = map(int, row[:6])
        if class_id == 0:  # Only detect 'person' class
            detected_people.append([x1, y1, x2, y2])

    # Update tracker
    tracked_objects = tracker.update(detected_people)

    for obj in tracked_objects:
        x3, y3, x4, y4, person_id = obj

        # Check if person is in Entry or Exit Area
        in_entry = cv2.pointPolygonTest(np.array(entry_area, np.int32), (x4, y4), False) >= 0
        in_exit = cv2.pointPolygonTest(np.array(exit_area, np.int32), (x4, y4), False) >= 0

        # Fix: Prevent double counting when a person moves back into the entry area
        if in_entry and person_id not in entered_customers and person_id not in customer_status:
            entry_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            entered_customers[person_id] = entry_time
            customer_status[person_id] = "inside"
            print(f"🟢 Customer {person_id} Entered at {entry_time}")

        # If exiting, count as "leaved" and remove from "inside" tracking
        if in_exit and person_id in customer_status and person_id not in leaved_customers:
            exit_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            leaved_customers[person_id] = exit_time
            del customer_status[person_id]  # Remove from inside tracking
            print(f"🔴 Customer {person_id} Left at {exit_time}")

        # Draw green box for Entry Area (Only if customer is inside)
        if in_entry:
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {person_id}", (x3, y3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw red box for Exit Area
        if in_exit:
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)

    # Draw Entry & Exit Area Polygons
    cv2.polylines(frame, [np.array(entry_area, np.int32)], True, (255, 0, 0), 2)
    cv2.putText(frame, 'Entry', (504, 471), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)

    cv2.polylines(frame, [np.array(exit_area, np.int32)], True, (255, 0, 0), 2)
    cv2.putText(frame, 'Exit', (466, 485), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)

    # Display live counts
    cv2.putText(frame, f"Entered: {len(entered_customers)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Leaved: {len(leaved_customers)}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("RGB", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
        break

# Save final data before exiting
save_to_csv()
cap.release()
cv2.destroyAllWindows()
