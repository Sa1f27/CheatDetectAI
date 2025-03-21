import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO 
import time
import os
import threading
import speech_recognition as sr
from datetime import datetime

try:
    import win32gui  # For screen monitoring on Windows
except ImportError:
    win32gui = None

try:
    import pyperclip  # For clipboard monitoring
except ImportError:
    pyperclip = None

from groq import Groq  # Groq client for LLM analysis

# -------------------------------
# Global Logging Setup
# -------------------------------
global_log = []
log_lock = threading.Lock()
program_start_time = time.time()

def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log_event(message):
    with log_lock:
        elapsed = time.time() - program_start_time
        timestamp = f"{get_current_time()} (+{elapsed:.1f}s)"
        log_entry = f"[{timestamp}] {message}"
        global_log.append(log_entry)
        print(log_entry)

def save_logs_to_file(filename="cheating_detection_log.txt"):
    with open(filename, "w") as f:
        for entry in global_log:
            f.write(entry + "\n")
    log_event(f"Logs saved to {filename}")

# -------------------------------
# Initialize Mediapipe Models
# -------------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=2,
    min_detection_confidence=0.5,
    refine_landmarks=True
)

mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5
)

# -------------------------------
# Load YOLOv8 Model
# -------------------------------
yolo_model_path = "yolo-models/yolov8l.pt"
if not os.path.exists(yolo_model_path):
    raise FileNotFoundError(f"YOLO model not found at {yolo_model_path}")
yolo_model = YOLO(yolo_model_path)
# Define additional suspicious object labels
suspicious_objects = {"cell phone", "book", "laptop", "earphone", "screen", "tablet"}

# -------------------------------
# Initialize Groq Client for LLM Analysis
# -------------------------------
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    log_event("Warning: GROQ_API_KEY not set. LLM analysis will fail.")
groq_client = Groq(api_key=groq_api_key)

# -------------------------------
# Gaze Detection Function
# -------------------------------
def get_gaze_direction(landmarks):
    """
    Determines gaze direction using iris landmarks.
    Returns "left", "right", or "center".
    """
    try:
        # Left eye calculations
        left_eye_left_x = landmarks.landmark[33].x
        left_eye_right_x = landmarks.landmark[133].x
        left_iris_x = landmarks.landmark[468].x
        left_gaze = ((left_iris_x - left_eye_left_x) / 
                     (left_eye_right_x - left_eye_left_x)) if left_eye_right_x != left_eye_left_x else 0.5

        # Right eye calculations
        right_eye_left_x = landmarks.landmark[362].x
        right_eye_right_x = landmarks.landmark[263].x
        right_iris_x = landmarks.landmark[473].x
        right_gaze = ((right_iris_x - right_eye_left_x) / 
                      (right_eye_right_x - right_eye_left_x)) if right_eye_right_x != right_eye_left_x else 0.5

        # Rough head yaw using nose position relative to eyes
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
        try:
            left_eye_x = landmarks.landmark[33].x
            right_eye_x = landmarks.landmark[263].x
            eye_center_x = (left_eye_x + right_eye_x) / 2
            if eye_center_x < 0.35:
                return "left"
            elif eye_center_x > 0.65:
                return "right"
            return "center"
        except Exception:
            return "center"

# -------------------------------
# Blink Detection Functions
# -------------------------------
def calculate_eye_aspect_ratio(landmarks, image_size, eye_points):
    """
    Calculates the eye aspect ratio (EAR) for given eye landmarks.
    eye_points: tuple (left_corner, top, right_corner, bottom)
    """
    left_corner = np.array([
        landmarks.landmark[eye_points[0]].x * image_size[0],
        landmarks.landmark[eye_points[0]].y * image_size[1]
    ])
    top = np.array([
        landmarks.landmark[eye_points[1]].x * image_size[0],
        landmarks.landmark[eye_points[1]].y * image_size[1]
    ])
    right_corner = np.array([
        landmarks.landmark[eye_points[2]].x * image_size[0],
        landmarks.landmark[eye_points[2]].y * image_size[1]
    ])
    bottom = np.array([
        landmarks.landmark[eye_points[3]].x * image_size[0],
        landmarks.landmark[eye_points[3]].y * image_size[1]
    ])
    horizontal_distance = np.linalg.norm(right_corner - left_corner)
    vertical_distance = np.linalg.norm(top - bottom)
    return vertical_distance / horizontal_distance if horizontal_distance != 0 else 0.0

def detect_blink(landmarks, image_size, ear_threshold=0.25):
    """
    Detects a blink based on the average EAR of both eyes.
    Returns True if a blink is detected.
    """
    # Approximate indices for eyes (may require fine-tuning)
    left_eye_points = (33, 159, 133, 145)
    right_eye_points = (263, 386, 362, 374)
    left_ear = calculate_eye_aspect_ratio(landmarks, image_size, left_eye_points)
    right_ear = calculate_eye_aspect_ratio(landmarks, image_size, right_eye_points)
    avg_ear = (left_ear + right_ear) / 2
    return avg_ear < ear_threshold

# -------------------------------
# Head Pose Estimation Function
# -------------------------------
def estimate_head_pose(landmarks, image_size):
    """
    Uses selected facial landmarks to estimate head pose.
    Returns pitch, yaw, and roll angles (in degrees).
    """
    try:
        image_points = np.array([
            [landmarks.landmark[1].x * image_size[0], landmarks.landmark[1].y * image_size[1]],     # Nose tip
            [landmarks.landmark[152].x * image_size[0], landmarks.landmark[152].y * image_size[1]], # Chin
            [landmarks.landmark[33].x * image_size[0], landmarks.landmark[33].y * image_size[1]],   # Left eye left corner
            [landmarks.landmark[263].x * image_size[0], landmarks.landmark[263].y * image_size[1]], # Right eye right corner
            [landmarks.landmark[61].x * image_size[0], landmarks.landmark[61].y * image_size[1]],   # Left mouth corner
            [landmarks.landmark[291].x * image_size[0], landmarks.landmark[291].y * image_size[1]]  # Right mouth corner
        ], dtype="double")
        
        model_points = np.array([
            [0.0, 0.0, 0.0],             # Nose tip
            [0.0, -63.6, -12.5],         # Chin
            [-43.3, 32.7, -26.0],        # Left eye left corner
            [43.3, 32.7, -26.0],         # Right eye right corner
            [-28.9, -28.9, -24.1],       # Left mouth corner
            [28.9, -28.9, -24.1]         # Right mouth corner
        ])
        
        focal_length = image_size[0]
        center = (image_size[0] / 2, image_size[1] / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )
        dist_coeffs = np.zeros((4, 1))
        
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs
        )
        if not success:
            log_event("Head pose estimation failed.")
            return 0.0, 0.0, 0.0
        
        rmat, _ = cv2.Rodrigues(rotation_vector)
        sy = np.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
        singular = sy < 1e-6
        if not singular:
            x_angle = np.arctan2(rmat[2, 1], rmat[2, 2])
            y_angle = np.arctan2(-rmat[2, 0], sy)
            z_angle = np.arctan2(rmat[1, 0], rmat[0, 0])
        else:
            x_angle = np.arctan2(-rmat[1, 2], rmat[1, 1])
            y_angle = np.arctan2(-rmat[2, 0], sy)
            z_angle = 0
        
        pitch = np.degrees(x_angle)
        yaw = np.degrees(y_angle)
        roll = np.degrees(z_angle)
        return pitch, yaw, roll
    except Exception as e:
        log_event(f"Error estimating head pose: {e}")
        return 0.0, 0.0, 0.0

# -------------------------------
# Hand Detection Function
# -------------------------------
def detect_hands(frame):
    """Detects hands in the frame using MediaPipe Hands and returns bounding boxes."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(rgb_frame)
    hands_info = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            hands_info.append((min_x, min_y, max_x, max_y))
    return hands_info

# -------------------------------
# Audio Monitoring Functions
# -------------------------------
suspicious_audio_keywords = {"google", "search", "cheat", "answer", "lookup"}

def audio_callback(recognizer, audio):
    """Callback for background audio capture with keyword spotting."""
    try:
        text = recognizer.recognize_google(audio)
        if text:
            log_event(f"Audio detected: '{text}'")
            lower_text = text.lower()
            for keyword in suspicious_audio_keywords:
                if keyword in lower_text:
                    log_event(f"Suspicious keyword '{keyword}' detected in audio.")
                    break
    except sr.UnknownValueError:
        pass
    except sr.RequestError as e:
        log_event(f"Audio recognition error: {e}")

def start_audio_monitoring():
    """Starts background audio monitoring using SpeechRecognition."""
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
    stop_listening = recognizer.listen_in_background(mic, audio_callback)
    return stop_listening

def verify_speaker(audio_data):
    # Placeholder: Integrate a speaker recognition library for real verification.
    return True

def detect_background_noise(audio_data):
    # Placeholder: Analyze audio_data for unusual background noise.
    return False

# -------------------------------
# Screen & Clipboard Monitoring Functions
# -------------------------------
def screen_monitoring():
    """
    Monitors the active window title and logs if the active window is not in an allowed list.
    Only works on Windows.
    """
    allowed_windows = ["Zoom", "Teams", "InterviewApp"]
    while True:
        if win32gui:
            active_window = win32gui.GetWindowText(win32gui.GetForegroundWindow())
            if active_window and not any(allowed in active_window for allowed in allowed_windows):
                log_event(f"Suspicious active window: '{active_window}'")
        time.sleep(2)

def clipboard_monitoring():
    """
    Monitors the system clipboard for any changes.
    """
    if not pyperclip:
        log_event("Clipboard monitoring not available (pyperclip not installed).")
        return
    recent_value = pyperclip.paste()
    while True:
        current_value = pyperclip.paste()
        if current_value != recent_value and current_value.strip() != "":
            log_event(f"Clipboard changed: '{current_value[:50]}...'")
            recent_value = current_value
        time.sleep(3)

# -------------------------------
# Video Analysis Function (from file)
# -------------------------------
def analyze_video(video_path):
    """
    Processes the video file frame-by-frame, performing face, hand, object detection,
    gaze tracking, blink detection, head pose estimation, and absence of face detection.
    Logs events that suggest possible cheating.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log_event(f"Error: Could not open video file {video_path}")
        return []
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = 0
    analysis_start_time = time.time()
    gaze_state = "center"
    gaze_start_time = None
    no_face_start = None
    blink_count = 0
    blink_flag = False  # To avoid repeated logging for a single blink
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        timestamp = frame_count / fps
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = face_mesh.process(rgb_frame)

        if face_results.multi_face_landmarks:
            no_face_start = None  # Reset when a face is detected
            if len(face_results.multi_face_landmarks) > 1:
                log_event("Multiple faces detected")
            
            for face_landmarks in face_results.multi_face_landmarks:
                # Gaze detection with duration check
                current_gaze = get_gaze_direction(face_landmarks)
                if current_gaze != gaze_state:
                    if gaze_state != "center" and gaze_start_time is not None:
                        duration = timestamp - gaze_start_time
                        if duration > 1.5:
                            log_event(f"Eyes looking {gaze_state} for {duration:.2f}s")
                    gaze_state = current_gaze
                    gaze_start_time = timestamp if current_gaze != "center" else None

                # Blink detection
                if detect_blink(face_landmarks, (640, 480)):
                    if not blink_flag:
                        blink_count += 1
                        log_event(f"Blink detected. Total blinks: {blink_count}")
                        blink_flag = True
                else:
                    blink_flag = False

                # Head pose estimation
                pitch, yaw, roll = estimate_head_pose(face_landmarks, (640, 480))
                if abs(pitch) > 20 or abs(yaw) > 20:
                    log_event(f"Abnormal head pose: pitch={pitch:.2f}, yaw={yaw:.2f}, roll={roll:.2f}")
        else:
            if no_face_start is None:
                no_face_start = timestamp
            elif timestamp - no_face_start > 3:
                log_event(f"No face detected for {timestamp - no_face_start:.2f}s")

        # YOLOv8 Object Detection
        yolo_results = yolo_model(frame)
        detections = yolo_results[0]
        for result in detections.boxes:
            confidence = result.conf.item()
            label = yolo_model.names[int(result.cls.item())]
            if confidence > 0.3 and label in suspicious_objects:
                log_event(f"Suspicious object '{label}' detected (conf: {confidence:.2f})")

        # Hand Detection
        hands = detect_hands(frame)
        if hands:
            if len(hands) > 1:
                log_event("Multiple hands detected")
            else:
                log_event("Hand detected")
        
        frame_count += 1

    cap.release()
    total_time = time.time() - analysis_start_time
    log_event(f"Processed {frame_count} frames in {total_time:.2f} seconds")
    return global_log

# -------------------------------
# Live Camera Analysis Function
# -------------------------------
def analyze_live_camera():
    """
    Captures video from the webcam in real time, performing the same analysis as with a video file.
    Press 'q' in the preview window to stop the capture.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        log_event("Error: Could not open camera")
        return []
    
    analysis_start_time = time.time()
    frame_count = 0
    gaze_state = "center"
    gaze_start_time = None
    no_face_start = None
    blink_count = 0
    blink_flag = False

    while True:
        ret, frame = cap.read()
        if not ret:
            log_event("Failed to grab frame from camera.")
            break

        frame = cv2.resize(frame, (640, 480))
        timestamp = time.time() - analysis_start_time
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = face_mesh.process(rgb_frame)

        if face_results.multi_face_landmarks:
            no_face_start = None
            if len(face_results.multi_face_landmarks) > 1:
                log_event("Multiple faces detected")
            
            for face_landmarks in face_results.multi_face_landmarks:
                current_gaze = get_gaze_direction(face_landmarks)
                if current_gaze != gaze_state:
                    if gaze_state != "center" and gaze_start_time is not None:
                        duration = timestamp - gaze_start_time
                        if duration > 1.5:
                            log_event(f"Eyes looking {gaze_state} for {duration:.2f}s")
                    gaze_state = current_gaze
                    gaze_start_time = timestamp if current_gaze != "center" else None

                if detect_blink(face_landmarks, (640, 480)):
                    if not blink_flag:
                        blink_count += 1
                        log_event(f"Blink detected. Total blinks: {blink_count}")
                        blink_flag = True
                else:
                    blink_flag = False

                pitch, yaw, roll = estimate_head_pose(face_landmarks, (640, 480))
                if abs(pitch) > 20 or abs(yaw) > 20:
                    log_event(f"Abnormal head pose: pitch={pitch:.2f}, yaw={yaw:.2f}, roll={roll:.2f}")
        else:
            if no_face_start is None:
                no_face_start = timestamp
            elif timestamp - no_face_start > 3:
                log_event(f"No face detected for {timestamp - no_face_start:.2f}s")

        yolo_results = yolo_model(frame)
        detections = yolo_results[0]
        for result in detections.boxes:
            confidence = result.conf.item()
            label = yolo_model.names[int(result.cls.item())]
            if confidence > 0.3 and label in suspicious_objects:
                log_event(f"Suspicious object '{label}' detected (conf: {confidence:.2f})")

        hands = detect_hands(frame)
        if hands:
            if len(hands) > 1:
                log_event("Multiple hands detected")
            else:
                log_event("Hand detected")
        
        cv2.imshow("Live Interview Cheating Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            log_event("Live camera capture terminated by user.")
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    total_time = time.time() - analysis_start_time
    log_event(f"Processed {frame_count} frames from camera in {total_time:.2f} seconds")
    return global_log

# -------------------------------
# Cheating Analysis Function (LLM)
# -------------------------------
def detect_cheating(logs):
    """
    Sends the collected logs to an LLM (via Groq) for detailed analysis.
    Returns the LLM’s explanation on whether the candidate likely cheated.
    """
    if not logs:
        return "No suspicious behavior detected."
    
    prompt = (
        "Analyze the following events from an interview session to determine if the candidate was likely cheating. "
        "Focus on unusual gaze direction, abnormal head pose, absence of face, unauthorized objects (e.g., cell phone, book, laptop, earphone, screen, tablet), "
        "hand movements, unexpected audio cues, and suspicious active window detections.\n\n"
        + "\n".join(logs) +
        "\n\nProvide a detailed explanation of your reasoning for each event."
    )
    
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.2-3b-preview",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1024,
            top_p=1,
            stream=False,
        )
        return completion.choices[0].message.content
    except Exception as e:
        log_event(f"LLM analysis error: {e}")
        return "Error during LLM analysis."

# -------------------------------
# Main Function
# -------------------------------
def main():
    video_path = r"videos/cheat-21.mp4"  # Update this path to your video file

    log_event("Starting audio monitoring...")
    stop_audio = start_audio_monitoring()
    
    # Start screen monitoring thread (Windows only)
    if win32gui:
        log_event("Starting screen monitoring...")
        screen_thread = threading.Thread(target=screen_monitoring, daemon=True)
        screen_thread.start()
    else:
        log_event("Screen monitoring is not available on this platform.")
    
    # Start clipboard monitoring thread if available
    if pyperclip:
        log_event("Starting clipboard monitoring...")
        clipboard_thread = threading.Thread(target=clipboard_monitoring, daemon=True)
        clipboard_thread.start()
    else:
        log_event("Clipboard monitoring not available.")
    
    # Use video file if it exists; otherwise, fall back to live camera capture.
    if os.path.exists(video_path):
        log_event("Video file found. Starting video analysis...")
        logs = analyze_video(video_path)
    else:
        log_event("Video file not found. Starting live camera interview...")
        logs = analyze_live_camera()
    
    log_event("Analysis complete. Collected logs:")
    for entry in logs:
        print(entry)
    
    log_event("Performing LLM-based cheating analysis...")
    result = detect_cheating(logs)
    print("\nLLM Analysis Result:\n" + result)
    
    # Stop audio monitoring (non-blocking)
    stop_audio(wait_for_stop=False)
    
    # Save logs to file for record keeping
    save_logs_to_file()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_event(f"An error occurred in main: {e}")
