import time
import math
import threading
from typing import Optional, Tuple

import numpy as np
import cv2
from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import mediapipe as mp
from ultralytics import YOLO
import torch
import speech_recognition as sr

app = Flask(__name__)
app.config['SECRET_KEY'] = 'aiforsecurity'
socketio = SocketIO(app)

CAMERA_SOURCE = "http://192.168.0.102:4747/mjpegfeed"

YAW_THRESH = 20
PITCH_DOWN_THRESH = 40
PITCH_UP_THRESH = -15
EMA_ALPHA = 0.3
CALIB_N = 20
DISTRACT_DWELL_S = 1.2
AWAY_DWELL_S = 1.5

DEVICE_CLASSES = [67, 73, 76]
PHONE_CLASS = 67
PHONE_CONF = 0.45
OTHER_CONF = 0.45
YOLO_MIN_MS = 250
YOLO_IMGSZ = 416
DEVICE_HOLD_S = 2.5

# --- Time-Based Scoring Config (Points per Second) ---
SCORE_RECOVERY_RATE_PER_SEC = 0.5    # Recover 0.5 points per second when focused
SCORE_PENALTY_DISTRACT_PER_SEC = 0.75 # Lose 0.75 points per second when distracted
SCORE_PENALTY_MAJOR_PER_SEC = 2.5    # Lose 2.5 points per second for major infractions
# ---

class CameraGrabber:
    def __init__(self, src=0, req_w=1280, req_h=720, buffer_size=2):
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open webcam source: {src}")

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
        if req_w and req_h:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, req_w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, req_h)

        try:
            cv2.setUseOptimized(True)
            cv2.setNumThreads(0)
        except Exception:
            pass

        self.lock = threading.Lock()
        self.latest = None
        self.stopped = False
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while not self.stopped:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self.lock:
                self.latest = frame

    def read(self):
        with self.lock:
            return None if self.latest is None else self.latest.copy()

    def release(self):
        self.stopped = True
        try:
            self.thread.join(timeout=0.5)
        except Exception:
            pass
        self.cap.release()

def resize_keep_aspect(img, target_w=640) -> Tuple[np.ndarray, int, int]:
    h, w = img.shape[:2]
    if w == target_w:
        return img, w, h
    scale = target_w / float(w)
    new_w = target_w
    new_h = int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized, new_w, new_h

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=4,
    refine_landmarks=True,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.5
)

try:
    yolo_model = YOLO('yolov8m.pt')
    print("Loaded YOLO: yolov8m.pt")
except Exception:
    try:
        yolo_model = YOLO('yolov8s.pt')
        print("Loaded YOLO: yolov8s.pt (m not found)")
    except Exception:
        yolo_model = YOLO('yolov8n.pt')
        print("Loaded YOLO: yolov8n.pt (m and s not found)")

DEVICE = 0 if torch.cuda.is_available() else 'cpu'
print(f"YOLO device: {DEVICE}")

recognizer = sr.Recognizer()
microphone = sr.Microphone()

face_3d = np.array([
    [0.0, 0.0, 0.0],
    [0.0, -330.0, -65.0],
    [-225.0, 170.0, -135.0],
    [225.0, 170.0, -135.0],
    [-150.0, -150.0, -125.0],
    [150.0, -150.0, -125.0]
], dtype=np.float64)
landmarks_indices = [1, 152, 33, 263, 61, 291]

def rotationMatrixToEulerAngles(R):
    sy = math.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6
    if not singular:
        roll  = math.degrees(math.atan2(R[2, 1], R[2, 2]))
        pitch = math.degrees(math.atan2(-R[2, 0], sy))
        yaw   = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    else:
        roll  = math.degrees(math.atan2(-R[1, 2], R[1, 1]))
        pitch = math.degrees(math.atan2(-R[2, 0], sy))
        yaw   = 0.0
    return yaw, pitch, roll

current_alert_message = "STATUS: INITIALIZING..."
current_alert_status = "ok"
calibrated = False
calib_frames = []
yaw_base = 0.0
pitch_base = 0.0
yaw_ema = 0.0
pitch_ema = 0.0
distract_start: Optional[float] = None
away_start: Optional[float] = None
last_seen_face_time: Optional[float] = None
last_device_time: float = 0.0
last_device_name: Optional[str] = None
last_device_box: Optional[Tuple[int, int, int, int]] = None
last_yolo_time: float = 0.0

attention_score = 100.0
current_penalty_reason = "FOCUSED"
last_score_update_time = time.monotonic() # Initialize time for score calculation

@app.route("/")
def index():
    return render_template("index3.html")

@socketio.on('recalibrate')
def handle_recalibrate():
    global calibrated, calib_frames, yaw_base, pitch_base, current_alert_message, current_alert_status
    print("Recalibration triggered by client.")
    calibrated = False
    calib_frames = []
    yaw_base = 0.0
    pitch_base = 0.0
    
    current_alert_message = "RECALIBRATING: Look straight at camera..."
    current_alert_status = "calib"
    socketio.emit('proctor_alert', {
        'message': current_alert_message,
        'status': current_alert_status
    })

def generate_frames():
    global current_alert_message, current_alert_status
    global calibrated, calib_frames, yaw_base, pitch_base, yaw_ema, pitch_ema
    global distract_start, away_start, last_seen_face_time
    global last_device_time, last_device_name, last_device_box, last_yolo_time
    global attention_score, current_penalty_reason, last_score_update_time

    cam = CameraGrabber(src=CAMERA_SOURCE, req_w=1280, req_h=720, buffer_size=2)
    
    # Send initial message (Calibration or Connecting)
    initial_message = "CALIBRATION: Look straight at camera..." if not calibrated else "Connecting..."
    initial_status = "calib" if not calibrated else "ok"
    current_alert_message = initial_message
    current_alert_status = initial_status
    socketio.emit('proctor_alert', {
        'message': current_alert_message,
        'status': current_alert_status
    })
    if not calibrated:
        print("Hold still and look straight for auto-calibration...")
    print("Or, click 'Recalibrate' button on the webpage to restart.")


    cam_matrix = None
    dist_coeffs = np.zeros((4, 1))

    _, test_w, test_h = resize_keep_aspect(np.zeros((720, 1280, 3), dtype=np.uint8), target_w=640)
    MP_PROCESS_W = 480
    MP_PROCESS_H = int(test_h * (float(MP_PROCESS_W) / test_w))
    print(f"MediaPipe processing size: {MP_PROCESS_W}x{MP_PROCESS_H}")
    
    last_score_update_time = time.monotonic() # Reset timer when starting generate_frames

    try:
        while True:
            frame_raw = cam.read()
            if frame_raw is None:
                time.sleep(0.01)
                continue

            frame_small, FRAME_W, FRAME_H = resize_keep_aspect(frame_raw, target_w=640)

            if cam_matrix is None:
                focal_length = FRAME_W
                center = (FRAME_W / 2.0, FRAME_H / 2.0)
                cam_matrix = np.array([
                    [focal_length, 0, center[0]],
                    [0, focal_length, center[1]],
                    [0, 0, 1]
                ], dtype=np.float64)

            now = time.monotonic()
            delta_time = now - last_score_update_time # Time elapsed since last frame

            # Default status for this frame, might be overridden
            frame_alert_message = "STATUS: FOCUSED"
            frame_alert_status = "ok"

            # --- Face Mesh Processing ---
            frame_mp = cv2.resize(frame_small, (MP_PROCESS_W, MP_PROCESS_H), interpolation=cv2.INTER_AREA)
            rgb_mp = cv2.cvtColor(frame_mp, cv2.COLOR_BGR2RGB)
            rgb_mp.flags.writeable = False
            results = face_mesh.process(rgb_mp)
            
            num_faces = len(results.multi_face_landmarks) if results.multi_face_landmarks else 0

            # --- Detection Logic (Sets frame_alert_message & frame_alert_status) ---
            # Priority 1: Calibration
            if not calibrated:
                 frame_alert_message = "CALIBRATION: Look straight..."
                 frame_alert_status = "calib"
                 if num_faces == 1:
                     lms = results.multi_face_landmarks[0]
                     face_2d_calib = []
                     valid_calib = True
                     for idx in landmarks_indices:
                         if idx < len(lms.landmark):
                             lm = lms.landmark[idx]
                             if not (0 <= lm.x <= 1 and 0 <= lm.y <= 1): valid_calib = False; break
                             x, y = int(lm.x * FRAME_W), int(lm.y * FRAME_H)
                             face_2d_calib.append([x, y])
                         else: valid_calib = False; break

                     if valid_calib and len(face_2d_calib) == 6:
                        face_2d_calib = np.array(face_2d_calib, dtype=np.float64)
                        try:
                           success_pnp, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d_calib, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
                           if success_pnp:
                               rmat, _ = cv2.Rodrigues(rot_vec)
                               yaw_raw, pitch_raw, _ = rotationMatrixToEulerAngles(rmat)
                               calib_frames.append((yaw_raw, pitch_raw))
                               cv2.putText(frame_small, f"Calibrating... {len(calib_frames)}/{CALIB_N}", (10, FRAME_H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                               if len(calib_frames) >= CALIB_N:
                                   yaw_base = float(np.median([y for y, _ in calib_frames]))
                                   pitch_base = float(np.median([p for _, p in calib_frames]))
                                   calibrated = True
                                   print(f"Calibration done: yaw_base={yaw_base:.1f}, pitch_base={pitch_base:.1f}")
                                   frame_alert_message = "STATUS: CALIBRATED. Monitoring..." # Temp message for next frame
                                   frame_alert_status = "ok"
                        except cv2.error: pass # Ignore solvePnP errors during calibration

            # Priority 2: Multiple People
            elif num_faces > 1:
                frame_alert_message = "ALERT: MULTIPLE PEOPLE DETECTED!"
                frame_alert_status = "alert"
                last_seen_face_time = now
                away_start = None
                distract_start = None
            
            # Priority 3: Away
            elif num_faces == 0:
                if away_start is None: away_start = now
                if last_seen_face_time and (now - last_seen_face_time) < AWAY_DWELL_S:
                    frame_alert_message = "STATUS: Searching for face..."
                    frame_alert_status = "ok" # Still OK during grace period
                elif (now - away_start) >= AWAY_DWELL_S:
                    frame_alert_message = "ALERT: STUDENT AWAY!"
                    frame_alert_status = "alert"
                else: # Within grace period but after dwell time started
                    frame_alert_message = "STATUS: Searching for face..."
                    frame_alert_status = "ok"
                distract_start = None
            
            # Priority 4: Pose Estimation (Distraction)
            else: # Exactly one face, and calibrated
                last_seen_face_time = now
                away_start = None

                lms = results.multi_face_landmarks[0]
                face_2d = []
                valid = True
                for idx in landmarks_indices:
                     if idx < len(lms.landmark):
                         lm = lms.landmark[idx]
                         if not (0 <= lm.x <= 1 and 0 <= lm.y <= 1): valid = False; break
                         x, y = int(lm.x * FRAME_W), int(lm.y * FRAME_H)
                         face_2d.append([x, y])
                     else: valid = False; break

                if valid and len(face_2d) == 6:
                    face_2d = np.array(face_2d, dtype=np.float64)
                    try:
                        success_pnp, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
                        if success_pnp:
                            rmat, _ = cv2.Rodrigues(rot_vec)
                            yaw_raw, pitch_raw, _ = rotationMatrixToEulerAngles(rmat)
                            yaw = (yaw_raw - yaw_base)
                            pitch = -(pitch_raw - pitch_base) # Invert pitch

                            yaw_ema = (1 - EMA_ALPHA) * yaw_ema + EMA_ALPHA * yaw
                            pitch_ema = (1 - EMA_ALPHA) * pitch_ema + EMA_ALPHA * pitch

                            over_thresh = (yaw_ema < -YAW_THRESH or yaw_ema > YAW_THRESH or
                                           pitch_ema > PITCH_DOWN_THRESH or pitch_ema < PITCH_UP_THRESH)

                            if over_thresh:
                                if distract_start is None: distract_start = now
                                if (now - distract_start) >= DISTRACT_DWELL_S:
                                    if yaw_ema < -YAW_THRESH: reason = f"Left {int(abs(yaw_ema))}°"
                                    elif yaw_ema > YAW_THRESH: reason = f"Right {int(abs(yaw_ema))}°"
                                    elif pitch_ema > PITCH_DOWN_THRESH: reason = f"Down {int(abs(pitch_ema))}°"
                                    else: reason = f"Up {int(abs(pitch_ema))}°"
                                    frame_alert_message = f"ALERT: DISTRACTED ({reason})"
                                    frame_alert_status = "alert"
                                else: # Distracted but within dwell time
                                    frame_alert_message = "STATUS: FOCUSED" # Show focused until dwell time expires
                                    frame_alert_status = "ok"
                            else: # Within thresholds
                                distract_start = None
                                frame_alert_message = "STATUS: FOCUSED"
                                frame_alert_status = "ok"

                            cv2.putText(frame_small, f"Yaw: {int(yaw_ema)}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            cv2.putText(frame_small, f"Pitch: {int(pitch_ema)}", (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        else:
                             frame_alert_message, frame_alert_status = "STATUS: Pose Estimation Failed", "ok"
                    except cv2.error as e:
                        print(f"solvePnP error: {e}")
                        frame_alert_message, frame_alert_status = "STATUS: Pose Calc Error", "ok"
                else:
                    frame_alert_message, frame_alert_status = "STATUS: Landmarks Unclear", "ok"
            
            # --- YOLO device detection (Run periodically) ---
            if (now - last_yolo_time) * 1000.0 >= YOLO_MIN_MS:
                last_yolo_time = now
                try:
                    with torch.inference_mode():
                        results_yolo = yolo_model(frame_small, imgsz=YOLO_IMGSZ, conf=min(PHONE_CONF, OTHER_CONF), classes=DEVICE_CLASSES, device=DEVICE, verbose=False)

                    best_name, best_box, best_priority = None, None, 0.0
                    for res in results_yolo:
                        for box in res.boxes:
                            cls_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            name = yolo_model.names.get(cls_id, str(cls_id))
                            thr = PHONE_CONF if cls_id == PHONE_CLASS else OTHER_CONF
                            if conf < thr: continue
                            priority = conf + (2.0 if cls_id == PHONE_CLASS else 1.0)
                            if priority > best_priority:
                                best_priority = priority
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                best_box = (x1, y1, x2, y2)
                                best_name = name

                    if best_name is not None:
                        last_device_time = now
                        last_device_name = best_name
                        last_device_box = best_box
                    #else: # Optional: Clear detection if nothing found this frame (more reactive, less stable)
                    #    last_device_name = None 
                    #    last_device_box = None

                except Exception as e:
                    print(f"YOLO error: {e}")

            # --- Apply Device Detection Alert (Overrides other alerts if necessary) ---
            # Check if a device was detected recently (within hold time)
            if last_device_name and (now - last_device_time) <= DEVICE_HOLD_S:
                # Override only if status is currently OK (don't override MULTIPLE/AWAY)
                if frame_alert_status == "ok":
                    if last_device_name != 'keyboard':
                        frame_alert_message = f"ALERT: {last_device_name.upper()} DETECTED!"
                        frame_alert_status = "alert"
                    else: # If keyboard detected and not distracted, show typing status
                        frame_alert_message = "STATUS: TYPING (Keyboard)"
                        frame_alert_status = "ok"

                # Draw box regardless of override
                if last_device_box:
                    color = (0, 0, 255) if frame_alert_status == "alert" and last_device_name != 'keyboard' else (0, 255, 0)
                    cv2.rectangle(frame_small, (last_device_box[0], last_device_box[1]), (last_device_box[2], last_device_box[3]), color, 2)
                    cv2.putText(frame_small, f"{last_device_name}", (last_device_box[0], max(20, last_device_box[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


            # --- Emit Alert Status Update (Only if changed) ---
            if (frame_alert_message != current_alert_message) or (frame_alert_status != current_alert_status):
                current_alert_message = frame_alert_message
                current_alert_status = frame_alert_status
                socketio.emit('proctor_alert', {
                    'message': current_alert_message,
                    'status': current_alert_status
                })
            
            # --- Time-Based Score Calculation ---
            new_penalty_reason = "FOCUSED"
            # Apply penalty or recovery based on the final status for this frame
            if current_alert_status == "alert":
                 if "MULTIPLE" in current_alert_message:
                     attention_score -= SCORE_PENALTY_MAJOR_PER_SEC * delta_time
                     new_penalty_reason = "Multiple People"
                 elif "AWAY" in current_alert_message:
                     attention_score -= SCORE_PENALTY_MAJOR_PER_SEC * delta_time
                     new_penalty_reason = "User Away"
                 elif "DETECTED" in current_alert_message: # Device detection is major
                     attention_score -= SCORE_PENALTY_MAJOR_PER_SEC * delta_time
                     new_penalty_reason = "Device Detected"
                 elif "DISTRACTED" in current_alert_message:
                     attention_score -= SCORE_PENALTY_DISTRACT_PER_SEC * delta_time
                     new_penalty_reason = "Distracted"
            elif current_alert_status == "ok": # Recover only if status is OK
                 attention_score += SCORE_RECOVERY_RATE_PER_SEC * delta_time
                 new_penalty_reason = "FOCUSED"
            # No score change during calibration ('calib' status)
            
            attention_score = max(0.0, min(100.0, attention_score)) # Clamp score
            
            # Update penalty reason if it changed
            if new_penalty_reason != current_penalty_reason:
                current_penalty_reason = new_penalty_reason
            
            # Emit score update every frame
            socketio.emit('score_update', {
                'score': f"{attention_score:.2f}",
                'reason': current_penalty_reason
            })
            
            # Update time for next frame's delta calculation
            last_score_update_time = now
            # --- End Score Calculation ---

            # --- Stream frame ---
            ret, buffer = cv2.imencode(".jpg", frame_small, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ret:
                continue
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

    except GeneratorExit:
        pass # Expected when client disconnects
    except Exception as e:
        print(f"Error in generate_frames: {e}")
    finally:
        print("Releasing camera.")
        cam.release()

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    print("Starting the AI Proctoring server...")
    print(f"Attempting to use camera source: {CAMERA_SOURCE}")
    print("Go to http://127.0.0.1:5000 in your browser.")
    socketio.run(app, debug=False, allow_unsafe_werkzeug=True, host='0.0.0.0', port=5000)

