# PolySpectra (Project for AI Verse 2.0)

**Theme:** AI for Safety, Security, and Smart Living

---

## Team Members

* Avilash Rout
* A R Sagar
* Chaithanya D S
* Bhavana Balachandra Hegde
* Yashwanth E S

---

AI Proctoring System – Project Documentation
===========================================

Project Overview
----------------
This project implements an AI-powered proctoring system that monitors user attention and environment during online assessments. It integrates computer vision, speech detection, and liveness verification to ensure exam integrity. The system provides real-time alerts, calculates an attention score, and generates session reports.

Key Features
------------
- Real-time face detection and head pose estimation using MediaPipe Face Mesh
- Eye closure detection and iris-based gaze tracking
- YOLOv8-based device detection (phones, laptops, keyboards)
- Speech activity detection using background voice activity detection
- Liveness checks through hand-raise or head-nod challenges
- Tab switch detection from the browser
- Time-based scoring system with penalties and recovery
- Web dashboard with live video, alerts, focus score chart, and alert counters
- Automatic session report generation at the end of monitoring

System Architecture
-------------------
1. Video Capture: CameraGrabber class streams webcam frames with minimal latency
2. Preprocessing: Frames resized and prepared for MediaPipe and YOLO
3. Models:
   - MediaPipe Face Mesh for landmarks and pose
   - MediaPipe Hands for liveness detection
   - YOLOv8 for device detection
   - SpeechRecognition for speech activity
4. Logic:
   - Calibration of head pose and iris ratios
   - Priority-based alert system
   - Liveness challenges at intervals
   - Scoring system with penalties and recovery
5. Frontend:
   - Flask and Socket.IO serve a dashboard
   - Real-time alerts and score updates
   - Tab switch events sent from client to server

Installation and Setup
----------------------
1. Install Python 3.12
2. Install dependencies:
   - flask, flask-socketio, eventlet
   - opencv-python, numpy, mediapipe
   - ultralytics, torch, torchvision, torchaudio
   - SpeechRecognition, pyaudio
3. Run the server:
   python PolySpectra(final).py
4. Open the browser at http://127.0.0.1:5000
5. Click "Start Monitoring" to begin calibration and monitoring

Configuration
-------------
- Camera source index
- Head pose thresholds (yaw, pitch)
- Eye closure and gaze thresholds
- YOLO device classes and confidence levels
- Scoring penalties and recovery rates
- Speech detection enable/disable
- Liveness challenge intervals and thresholds

Usage
-----
- Start the server and open the dashboard
- Calibrate by looking straight at the camera
- Monitor alerts and focus score in real time
- Recalibrate if needed using the dashboard button
- At the end of the session, review the generated report file

Alerts
------
- Distracted (head pose deviation)
- Away (no face detected)
- Multiple people detected
- Eyes closed
- Gaze distraction
- Device detected (phone, laptop, etc.)
- Talking detected
- Tab switch detected
- Liveness challenge failed

Scoring
-------
- Starts at 100
- Recovers at 0.5 points per second when focused
- Penalties:
  - Major (away, multiple, device): -2.5 per second
  - Distracted, eyes closed, gaze: -0.75 per second
  - Tab switch: -10 instant penalty
- Clamped between 0 and 100


Future Enhancements
-------------------
- Per-user session isolation for multi-user deployment
- Detailed CSV/JSON logs with timestamps
- Improved liveness challenges with multimodal prompts
- Enhanced YOLO class filtering for fewer false positives
- Integration with exam platforms via API
- Emotional Anomaly Detection

---
