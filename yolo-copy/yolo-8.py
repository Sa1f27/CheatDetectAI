import cv2
import mediapipe as mp
import torch
import numpy as np
from ultralytics import YOLO  # YOLOv8
from groq import Groq
import time
import os


# Initialize Mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=2, min_detection_confidence=0.5, refine_landmarks=True)

# Load YOLOv8 model (download yolov8l.pt first)
model = YOLO("yolo-models/yolov8l.pt")  # Use 'yolov8m.pt' for a balanced model

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def get_gaze_direction(landmarks):
    try:
        # Iris-based gaze detection
        left_eye_left_x = landmarks.landmark[33].x
        left_eye_right_x = landmarks.landmark[133].x
        left_iris_x = landmarks.landmark[468].x
        left_gaze = (left_iris_x - left_eye_left_x) / (left_eye_right_x - left_eye_left_x) if left_eye_right_x != left_eye_left_x else 0.5
        
        right_eye_left_x = landmarks.landmark[362].x
        right_eye_right_x = landmarks.landmark[263].x
        right_iris_x = landmarks.landmark[473].x
        right_gaze = (right_iris_x - right_eye_left_x) / (right_eye_right_x - right_eye_left_x) if right_eye_right_x != right_eye_left_x else 0.5
        
        # Head pose yaw
        nose_tip_x = landmarks.landmark[1].x
        eye_mid_x = (left_eye_left_x + right_eye_right_x) / 2
        yaw = (nose_tip_x - eye_mid_x) * 100
        
        if abs(yaw) > 20:
            return "left" if yaw < 0 else "right"
        elif left_gaze < 0.35 or right_gaze < 0.35:
            return "left"
        elif left_gaze > 0.65 or right_gaze > 0.65:
            return "right"
        return "center"
    except IndexError:
        left_eye_x = landmarks.landmark[33].x
        right_eye_x = landmarks.landmark[263].x
        eye_center_x = (left_eye_x + right_eye_x) / 2
        if eye_center_x < 0.35:
            return "left"
        elif eye_center_x > 0.65:
            return "right"
        return "center"

def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return []
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    N = max(1, int(fps / 6))
    
    log = []
    frame_count = 0
    start_time = time.time()
    object_timers = {}
    gaze_state = "center"
    gaze_start_time = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (640, 480))  # Normalize resolution
        timestamp = frame_count / fps
        
        if frame_count % N == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                if len(results.multi_face_landmarks) > 1:
                    log.append(f"{timestamp:.2f}s - Multiple faces detected (suspicious)")
                
                for face_landmarks in results.multi_face_landmarks:
                    current_gaze = get_gaze_direction(face_landmarks)
                    if current_gaze != gaze_state:
                        if gaze_state != "center" and gaze_start_time is not None:
                            duration = timestamp - gaze_start_time
                            if duration > 1.5:  # Suspicious if >1.5s
                                log.append(f"{gaze_start_time:.2f}s to {timestamp:.2f}s - "
                                          f"Eyes looking {gaze_state} for {duration:.2f}s (suspicious)")
                        gaze_state = current_gaze
                        gaze_start_time = timestamp if current_gaze != "center" else None
            
            yolo_results = model(frame)  # Run YOLOv8 detection
            detections = yolo_results[0]  # Extract first batch of results
            current_objects = set()
            
            for result in detections.boxes:
                confidence = result.conf.item()
                label = model.names[int(result.cls.item())]  # Get class name
                
                if confidence > 0.3 and label in ["cell phone", "book", "laptop"]:
                    current_objects.add(label)
                    if label not in object_timers:
                        object_timers[label] = timestamp
            
            for label in list(object_timers.keys()):
                if label not in current_objects:
                    start_time = object_timers.pop(label)
                    duration = timestamp - start_time
                    if duration > 1:
                        log.append(f"{start_time:.2f}s to {timestamp:.2f}s - "
                                  f"{label.capitalize()} detected for {duration:.2f}s (suspicious)")
        
        frame_count += 1
    
    if gaze_state != "center" and gaze_start_time is not None:
        duration = timestamp - gaze_start_time
        if duration > 1.5:
            log.append(f"{gaze_start_time:.2f}s to {timestamp:.2f}s - "
                      f"Eyes looking {gaze_state} for {duration:.2f}s (suspicious)")
    for label, start_time in object_timers.items():
        duration = timestamp - start_time
        if duration > 1:
            log.append(f"{start_time:.2f}s to {timestamp:.2f}s - "
                      f"{label.capitalize()} detected for {duration:.2f}s (suspicious)")
    
    cap.release()
    elapsed_time = time.time() - start_time
    print(f"Processed {frame_count} frames in {elapsed_time:.2f} seconds")
    return log

def detect_cheating(log):
    if not log:
        return "No suspicious behavior detected."
    
    prompt = (
        "Analyze the following events from an interview video to determine if the candidate was likely cheating. "
        "Focus on suspicious behaviors: looking away for more than 1.5 seconds, presence of unauthorized objects "
        "(e.g., cell phone, book, laptop) for more than 1 second, and multiple faces in the frame. Ignore non-suspicious events:\n"
        "\n".join(log) + "\n\nProvide a detailed explanation of your reasoning, including whether each event suggests cheating."
    )
    
    completion = client.chat.completions.create(
        model="llama-3.2-3b-preview",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1024,
        top_p=1,
        stream=False,
    )
    
    return completion.choices[0].message.content

if __name__ == "__main__":
    video_path = r"videos/cheat-2.mp4"
    print("Analyzing video...")
    log = analyze_video(video_path)
    print("Detection log:")
    for entry in log:
        print(entry)
    print("\nLLM Analysis:")
    result = detect_cheating(log)
    print(result)
