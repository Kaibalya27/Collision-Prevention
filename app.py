import os
from flask import Flask, render_template, request, send_from_directory, url_for
import cv2
import numpy as np
import math
from ultralytics import YOLO

app=Flask(__name__)

model = YOLO("yolov8n.pt")

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "static/outputs"

FOCAL_LENGTH = 700
KNOWN_WIDTH = 50
SAFE_DISTANCE = 150

LANE_LEFT = 300
LANE_RIGHT = 1300

LEFT_CLEARANCE = 250
RIGHT_CLEARANCE = 970
PREV=False

def estimate_distance(bbox_width):
    return float("inf") if bbox_width == 0 else (KNOWN_WIDTH * FOCAL_LENGTH) / bbox_width

def calculate_angle(x1, y1, x2, y2):
    return (x1 + x2) // 2

def is_path_clear(frame, results, direction):
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            object_center_x = (x1 + x2) // 2

            if direction == "left" and object_center_x < LANE_LEFT and object_center_x > LEFT_CLEARANCE:
                return False  
            if direction == "right" and object_center_x > LANE_RIGHT and object_center_x < RIGHT_CLEARANCE:
                return False  
    return True  

def get_movement_action(angle, distance, frame, results):
    global PREV
    if distance < SAFE_DISTANCE:
        left_clear = is_path_clear(frame, results, "left")
        right_clear = is_path_clear(frame, results, "right")

        if angle < 850 and right_clear:
            PREV=True
            return "Move Right ->"
        if angle > 910 and left_clear:
            if PREV:
                return "<x> BRAKE! STOP <x>"           
            else:
                return "Move Left <-"
        # PREV=False
        return "<x> BRAKE! STOP <x>"
    
    return "Keep Moving"

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("Error: Could not open video file!")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # FourCC for mp4 (platform dependent) - mp4v is a common choice
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bbox_width = x2 - x1
                object_center_x = (x1 + x2) // 2

                if LANE_LEFT <= object_center_x <= LANE_RIGHT:
                    distance = estimate_distance(bbox_width)
                    angle = calculate_angle(x1, y1, x2, y2)


                    action = get_movement_action(angle, distance, frame, results)

                    color = (0, 255, 0) if distance > SAFE_DISTANCE else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"Dist: {int(distance)} cm", (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.putText(frame, action, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    print(action)

        writer.write(frame)
    # cv2.imshow("Detection & Warning", frame)
        

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/output', methods=['POST'])
def output():
    file = request.files['video']
    input_path = os.path.join(UPLOAD_FOLDER, file.filename)
    output_path = os.path.join(OUTPUT_FOLDER, file.filename)

    file.save(input_path)
    process_video(input_path, output_path)

    video_url = url_for('static', filename="outputs/" + file.filename)
    return render_template('results.html', video_url=video_url)

if __name__=='__main__':
    app.run(debug=True)