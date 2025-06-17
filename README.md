# Cheat Detection System

## Overview
This is a simple proctoring system that uses computer vision and audio analysis to monitor users during an online exam or interview. The system detects faces, tracks head movements, listens for background noise, and identifies potential cheating behaviors. It's built using Yolo, OpenCV, DeepFace, and some Python magic.

## Features
- **Face Detection:** Ensures the user is present in front of the camera.
- **Head Movement Tracking:** Detects if the user frequently looks away.
- **Multiple Face Detection:** Flags when more than one person is visible.
- **Audio Monitoring:** Identifies background noises that may indicate cheating.
- **Real-time Alerts:** Logs suspicious behavior for further review.

## Requirements
Make sure you have the following installed:
- Python 3.x
- Yolo
- OpenCV (`cv2`)
- DeepFace
- Pyaudio
- Numpy
- Matplotlib
- Sounddevice
- Librosa

## Setup Instructions  

### 1. Create a Virtual Environment (Recommended)  
```sh
python -m venv venv  
```
Activate it:  
- **Windows (cmd)**: `venv\Scripts\activate`   

### 2. Install Dependencies  
```sh
pip install -r requirements.txt  
```

## How to Run
1. Clone this repo or download the script.
2. Make sure your webcam and microphone are enabled.
3. Run the script:
   ```bash
   python app.py
   ```
4. The system will start detecting and logging any suspicious activity.

## How It Works
- The script continuously captures frames from the webcam.
- It uses OpenCV, Yolo to detect faces and track movement.
- DeepFace helps analyze facial expressions (if enabled).
- The microphone records short audio clips, analyzing background noise.
- Any unusual activity (e.g., looking away too often, extra voices) is flagged and logged.

## Future Improvements
- Integrate with an online platform for real-time monitoring.
- Add eye-tracking for more precise cheating detection.
- Implement a better machine learning model for identifying voice anomalies.
- Enhance the logging system with timestamps and detailed reports.

## Disclaimer
This is a basic proctoring system and is not foolproof. It can assist in monitoring but should be used alongside other security measures for best results.

---
