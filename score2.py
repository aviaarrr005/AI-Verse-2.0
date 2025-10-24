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

CAMERA_SOURCE = "http://192.168.0.101:4747/mjpegfeed" # NOTE: Ensure this IP is correct!

# Pose Thresholds
YAW_THRESH = 15
PITCH_DOWN_THRESH = 40
PITCH_UP_THRESH = -8
EMA_ALPHA = 0.3
CALIB_N = 20
DISTRACT_DWELL_S = 1.2
AWAY_DWELL_S = 1.5

# Device Detection Config
DEVICE_CLASSES = [67, 73, 76] # phone, book, keyboard
PHONE_CLASS = 67
PHONE_CONF = 0.25
OTHER_CONF = 0.25
YOLO_MIN_MS = 400
YOLO_IMGSZ = 416
DEVICE_HOLD_S = 2.5

# Scoring Config (Points per Second)
SCORE_RECOVERY_RATE_PER_SEC = 0.25
SCORE_PENALTY_DISTRACT_PER_SEC = 0.75
SCORE_PENALTY_MAJOR_PER_SEC = 2.5

class CameraGrabber:
    # ... (CameraGrabber class code remains the same) ...
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
        except Exception: pass
        self.lock = threading.Lock()
        self.latest = None
        self.stopped = False
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()
    def _reader(self):
        while not self.stopped:
            ok, frame = self.cap.read()
            if not ok:
                print(f"Warning: Failed to read frame from {CAMERA_SOURCE}")
                time.sleep(0.5)
                continue
            with self.lock: self.latest = frame
    def read(self):
        with self.lock: return None if self.latest is None else self.latest.copy()
    def release(self):
        self.stopped = True
        try: self.thread.join(timeout=0.5)
        except Exception: pass
        self.cap.release()

def resize_keep_aspect(img, target_w=640) -> Tuple[np.ndarray, int, int]:
    # ... (resize_keep_aspect function code remains the same) ...
    h, w = img.shape[:2]
    if w == 0 or h == 0: return None, 0, 0
    if w == target_w: return img, w, h
    scale = target_w / float(w)
    new_w = target_w
    new_h = int(round(h * scale))
    if new_w > 0 and new_h > 0:
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return resized, new_w, new_h
    else: return None, 0, 0

# --- Model Initialization ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=4, refine_landmarks=True,
    min_detection_confidence=0.25,
    min_tracking_confidence=0.5
)
try: yolo_model = YOLO('yolov8m.pt'); print("Loaded YOLO: yolov8m.pt")
except Exception:
    try: yolo_model = YOLO('yolov8s.pt'); print("Loaded YOLO: yolov8s.pt (m not found)")
    except Exception: yolo_model = YOLO('yolov8n.pt'); print("Loaded YOLO: yolov8n.pt (m and s not found)")
DEVICE = 0 if torch.cuda.is_available() else 'cpu'; print(f"YOLO device: {DEVICE}")
recognizer = sr.Recognizer(); microphone = sr.Microphone()
# --- End Model Initialization ---

# --- Pose Utilities ---
face_3d = np.array([
    [0.0, 0.0, 0.0], [0.0, -330.0, -65.0], [-225.0, 170.0, -135.0],
    [225.0, 170.0, -135.0], [-150.0, -150.0, -125.0], [150.0, -150.0, -125.0]
], dtype=np.float64)
landmarks_indices = [1, 152, 33, 263, 61, 291]
def rotationMatrixToEulerAngles(R):
    # ... (rotationMatrixToEulerAngles function code remains the same) ...
    sy = math.sqrt(R[0, 0]**2 + R[1, 0]**2); singular = sy < 1e-6
    if not singular:
        roll = math.degrees(math.atan2(R[2, 1], R[2, 2]))
        pitch = math.degrees(math.atan2(-R[2, 0], sy))
        yaw = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    else:
        roll = math.degrees(math.atan2(-R[1, 2], R[1, 1]))
        pitch = math.degrees(math.atan2(-R[2, 0], sy)); yaw = 0.0
    return yaw, pitch, roll
# --- End Pose Utilities ---

# --- State Variables ---
current_alert_message = "STATUS: INITIALIZING..."; current_alert_status = "ok"
calibrated = False; calib_frames = []; yaw_base = 0.0; pitch_base = 0.0
yaw_ema = 0.0; pitch_ema = 0.0
distract_start: Optional[float] = None; away_start: Optional[float] = None
last_seen_face_time: Optional[float] = None; last_device_time: float = 0.0
last_device_name: Optional[str] = None; last_device_box = None
last_yolo_time: float = 0.0; last_detected_conf: float = 0.0
attention_score = 100.0; current_penalty_reason = "FOCUSED"
last_score_update_time = time.monotonic()
# --- End State Variables ---

@app.route("/")
def index(): return render_template("index3.html") # Assuming index3.html is your latest

@socketio.on('recalibrate')
def handle_recalibrate():
    global calibrated, calib_frames, yaw_base, pitch_base, current_alert_message, current_alert_status, attention_score, yaw_ema, pitch_ema
    print("Recalibration triggered by client.")
    calibrated = False; calib_frames = []; yaw_base = 0.0; pitch_base = 0.0
    attention_score = 100.0; yaw_ema = 0.0; pitch_ema = 0.0 # Reset pose EMA too
    current_alert_message = "RECALIBRATING: Look straight..."; current_alert_status = "calib"
    socketio.emit('proctor_alert', {'message': current_alert_message, 'status': current_alert_status})
    socketio.emit('focus_percentage_update', {'percentage': '---'}) # Reset instant percentage
    socketio.emit('score_update', {'score': f"{attention_score:.2f}", 'reason': "RECALIBRATING"})

def calculate_instant_focus_percentage(yaw, pitch):
    # ... (calculate_instant_focus_percentage remains the same) ...
    """Calculates focus percentage based on current deviation."""
    yaw_deviation = abs(yaw)
    pitch_deviation = abs(pitch)
    yaw_loss = min(100, (yaw_deviation / abs(YAW_THRESH)) * 100) if YAW_THRESH != 0 else 0
    pitch_loss = 0
    if pitch > 0 and PITCH_DOWN_THRESH > 0: # Looking down
        pitch_loss = min(100, (pitch / PITCH_DOWN_THRESH) * 100)
    elif pitch < 0 and PITCH_UP_THRESH < 0: # Looking up
        pitch_loss = min(100, (abs(pitch) / abs(PITCH_UP_THRESH)) * 100)
    max_loss = max(yaw_loss, pitch_loss)
    percentage = max(0.0, 100.0 - max_loss)
    return percentage


def generate_frames():
    global current_alert_message, current_alert_status, calibrated, calib_frames
    global yaw_base, pitch_base, yaw_ema, pitch_ema, distract_start, away_start
    global last_seen_face_time, last_device_time, last_device_name, last_device_box
    global last_yolo_time, last_detected_conf, attention_score, current_penalty_reason
    global last_score_update_time

    cam = CameraGrabber(src=CAMERA_SOURCE, req_w=1280, req_h=720, buffer_size=2)
    initial_message = "CALIBRATION: Look straight..." if not calibrated else "Connecting..."
    initial_status = "calib" if not calibrated else "ok"
    # ... (Initial emit and prints remain the same) ...
    current_alert_message = initial_message; current_alert_status = initial_status
    socketio.emit('proctor_alert', {'message': current_alert_message, 'status': current_alert_status})
    if not calibrated: print("Hold still and look straight for auto-calibration...")
    print("Or, click 'Recalibrate' button on the webpage to restart.")

    cam_matrix = None; dist_coeffs = np.zeros((4, 1))
    _, test_w_display, test_h_display = resize_keep_aspect(np.zeros((720, 1280, 3), dtype=np.uint8), target_w=640)
    MP_PROCESS_W = 480
    MP_PROCESS_H = int(test_h_display * (float(MP_PROCESS_W) / test_w_display)) if test_w_display > 0 else 360
    if MP_PROCESS_H <= 0: MP_PROCESS_H = 360; print("Warning: Using default MP height 360.")
    print(f"MediaPipe processing size: {MP_PROCESS_W}x{MP_PROCESS_H}")
    last_score_update_time = time.monotonic() # Reset timer

    try:
        while True:
            frame_raw = cam.read()
            if frame_raw is None:
                # ... (Camera read error handling) ...
                current_alert_message = "ERROR: Cannot read camera feed!"; current_alert_status = "alert"
                socketio.emit('proctor_alert', {'message': current_alert_message, 'status': current_alert_status})
                current_penalty_reason = "Camera Error"
                socketio.emit('score_update', {'score': f"{attention_score:.2f}", 'reason': current_penalty_reason})
                time.sleep(1); continue

            frame_small, FRAME_W, FRAME_H = resize_keep_aspect(frame_raw, target_w=640)
            if frame_small is None or FRAME_W == 0 or FRAME_H == 0:
                print("Warning: Resized frame is invalid."); time.sleep(0.1); continue

            if cam_matrix is None:
                focal_length = FRAME_W; center = (FRAME_W / 2.0, FRAME_H / 2.0)
                cam_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype=np.float64)

            now = time.monotonic(); delta_time = now - last_score_update_time
            if delta_time <= 0: delta_time = 0.01 # Prevent division by zero or negative time

            # --- Defaults for Frame ---
            frame_alert_message = "STATUS: FOCUSED"; frame_alert_status = "ok"
            pose_focused_this_frame = True # Assume focused unless proven otherwise
            instant_focus_perc = 100.0 # Default to 100% focus

            # --- Face Mesh ---
            results = None; num_faces = 0
            if MP_PROCESS_H > 0:
                 try:
                     frame_mp = cv2.resize(frame_small, (MP_PROCESS_W, MP_PROCESS_H), interpolation=cv2.INTER_AREA)
                     rgb_mp = cv2.cvtColor(frame_mp, cv2.COLOR_BGR2RGB)
                     rgb_mp.flags.writeable = False
                     results = face_mesh.process(rgb_mp)
                     num_faces = len(results.multi_face_landmarks) if results.multi_face_landmarks else 0
                 except Exception as e:
                     print(f"Error during face mesh processing: {e}")
            else: print("Warning: Skipping Face Mesh due to invalid dimensions.")


            # --- Core Detection Logic ---
            if not calibrated:
                 # ... (Calibration logic) ...
                 frame_alert_message = "CALIBRATION: Look straight..."; frame_alert_status = "calib"
                 pose_focused_this_frame = False; instant_focus_perc = 0.0
                 if num_faces == 1 and results:
                     lms = results.multi_face_landmarks[0]; face_2d_calib = []; valid_calib = True
                     for idx in landmarks_indices:
                         if idx < len(lms.landmark):
                             lm = lms.landmark[idx];
                             if not (0 <= lm.x <= 1 and 0 <= lm.y <= 1): valid_calib = False; break
                             x, y = int(lm.x * FRAME_W), int(lm.y * FRAME_H); face_2d_calib.append([x, y])
                         else: valid_calib = False; break
                     if valid_calib and len(face_2d_calib) == 6:
                         face_2d_calib = np.array(face_2d_calib, dtype=np.float64)
                         try:
                            success_pnp, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d_calib, cam_matrix, dist_coeffs, cv2.SOLVEPNP_ITERATIVE)
                            if success_pnp:
                                rmat, _ = cv2.Rodrigues(rot_vec); yaw_raw, pitch_raw, _ = rotationMatrixToEulerAngles(rmat)
                                calib_frames.append((yaw_raw, pitch_raw))
                                cv2.putText(frame_small, f"Calibrating... {len(calib_frames)}/{CALIB_N}", (10, FRAME_H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                                if len(calib_frames) >= CALIB_N:
                                    yaw_base = float(np.median([y for y, _ in calib_frames])); pitch_base = float(np.median([p for _, p in calib_frames]))
                                    calibrated = True; print(f"Calibration done: yaw={yaw_base:.1f}, pitch={pitch_base:.1f}")
                                    frame_alert_message = "STATUS: CALIBRATED."; frame_alert_status = "ok"; pose_focused_this_frame = True; instant_focus_perc = 100.0
                         except cv2.error: pass
            elif num_faces > 1:
                 # ... (Multiple people logic) ...
                 frame_alert_message = "ALERT: MULTIPLE PEOPLE!"; frame_alert_status = "alert"; pose_focused_this_frame = False; instant_focus_perc = 0.0
                 last_seen_face_time = now; away_start = None; distract_start = None
            elif num_faces == 0:
                 # ... (Away logic) ...
                 pose_focused_this_frame = False; instant_focus_perc = 0.0
                 if away_start is None: away_start = now
                 if last_seen_face_time and (now - last_seen_face_time) < AWAY_DWELL_S: frame_alert_message = "STATUS: Searching..."; frame_alert_status = "ok"
                 elif (now - away_start) >= AWAY_DWELL_S: frame_alert_message = "ALERT: STUDENT AWAY!"; frame_alert_status = "alert"
                 else: frame_alert_message = "STATUS: Searching..."; frame_alert_status = "ok"
                 distract_start = None
            else: # Single face, calibrated
                 last_seen_face_time = now; away_start = None
                 if results and results.multi_face_landmarks:
                     lms = results.multi_face_landmarks[0]; face_2d = []; valid = True
                     for idx in landmarks_indices:
                          if idx < len(lms.landmark):
                              lm = lms.landmark[idx];
                              if not (0 <= lm.x <= 1 and 0 <= lm.y <= 1): valid = False; break
                              x, y = int(lm.x * FRAME_W), int(lm.y * FRAME_H); face_2d.append([x, y])
                          else: valid = False; break
                     if valid and len(face_2d) == 6:
                         face_2d = np.array(face_2d, dtype=np.float64)
                         try:
                             success_pnp, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_coeffs, cv2.SOLVEPNP_ITERATIVE)
                             if success_pnp:
                                 rmat, _ = cv2.Rodrigues(rot_vec); yaw_raw, pitch_raw, _ = rotationMatrixToEulerAngles(rmat)
                                 # ** FIX: Re-invert Yaw **
                                 yaw = -(yaw_raw - yaw_base)
                                 # ** FIX: Re-invert Pitch **
                                 pitch = -(pitch_raw - pitch_base)
                                 yaw_ema = (1 - EMA_ALPHA) * yaw_ema + EMA_ALPHA * yaw
                                 pitch_ema = (1 - EMA_ALPHA) * pitch_ema + EMA_ALPHA * pitch

                                 instant_focus_perc = calculate_instant_focus_percentage(yaw_ema, pitch_ema)

                                 over_thresh = (yaw_ema < -YAW_THRESH or yaw_ema > YAW_THRESH or
                                                pitch_ema > PITCH_DOWN_THRESH or pitch_ema < PITCH_UP_THRESH)

                                 if over_thresh:
                                     pose_focused_this_frame = False # Pose is not focused
                                     if distract_start is None: distract_start = now
                                     if (now - distract_start) >= DISTRACT_DWELL_S:
                                         # Prioritize Yaw for reason
                                         if yaw_ema < -YAW_THRESH: reason = f"Left {int(abs(yaw_ema))}°"
                                         elif yaw_ema > YAW_THRESH: reason = f"Right {int(abs(yaw_ema))}°"
                                         # Only check pitch if yaw is within threshold
                                         elif pitch_ema > PITCH_DOWN_THRESH: reason = f"Down {int(abs(pitch_ema))}°"
                                         elif pitch_ema < PITCH_UP_THRESH: reason = f"Up {int(abs(pitch_ema))}°"
                                         else: reason = "Unknown"

                                         frame_alert_message = f"ALERT: DISTRACTED ({reason})"; frame_alert_status = "alert"
                                     else: # Still within dwell time
                                         frame_alert_message = "STATUS: Looking Away..."; frame_alert_status = "ok"
                                 else: # Pose is focused
                                     pose_focused_this_frame = True
                                     distract_start = None; frame_alert_message = "STATUS: FOCUSED"; frame_alert_status = "ok"
                                 # Draw HUD
                                 cv2.putText(frame_small, f"Yaw: {int(yaw_ema)}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                 cv2.putText(frame_small, f"Pitch: {int(pitch_ema)}", (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                             else: # PnP failed
                                  pose_focused_this_frame = False; instant_focus_perc = 0.0
                                  frame_alert_message = "STATUS: Pose Estimation Failed"; frame_alert_status = "ok"
                         except cv2.error as e: # SolvePnP error
                             pose_focused_this_frame = False; instant_focus_perc = 0.0
                             print(f"solvePnP error: {e}"); frame_alert_message = "STATUS: Pose Calc Error"; frame_alert_status = "ok"
                     else: # Invalid landmarks
                         pose_focused_this_frame = False; instant_focus_perc = 0.0
                         frame_alert_message = "STATUS: Landmarks Unclear"; frame_alert_status = "ok"
                 else: # Should not happen if num_faces = 1, but safety check
                     pose_focused_this_frame = False; instant_focus_perc = 0.0
                     frame_alert_message = "STATUS: Face Mesh Error"; frame_alert_status = "ok"

            # --- YOLO ---
            if (now - last_yolo_time) * 1000.0 >= YOLO_MIN_MS: # Runs less often now
                last_yolo_time = now
                try:
                    with torch.inference_mode():
                        results_yolo = yolo_model(frame_small, imgsz=YOLO_IMGSZ, conf=min(PHONE_CONF, OTHER_CONF), classes=DEVICE_CLASSES, device=DEVICE, verbose=False)
                    best_name, best_box, best_priority, best_conf = None, None, 0.0, 0.0
                    for res in results_yolo:
                        for box in res.boxes:
                            cls_id = int(box.cls[0]); conf = float(box.conf[0]); name = yolo_model.names.get(cls_id, str(cls_id))
                            thr = PHONE_CONF if cls_id == PHONE_CLASS else OTHER_CONF
                            if conf < thr: continue
                            priority = conf + (2.0 if cls_id == PHONE_CLASS else 1.0)
                            if priority > best_priority:
                                best_priority = priority; best_conf = conf
                                x1, y1, x2, y2 = map(int, box.xyxy[0]); best_box = (x1, y1, x2, y2); best_name = name
                    if best_name is not None:
                        last_device_time = now; last_device_name = best_name; last_device_box = best_box; last_detected_conf = best_conf
                except Exception as e: print(f"YOLO error: {e}")

            # --- Apply Device Detection Alert ---
            device_detected_now = False
            if last_device_name and (now - last_device_time) <= DEVICE_HOLD_S:
                can_override = frame_alert_status != "alert" or "DISTRACTED" in frame_alert_message or "Looking Away" in frame_alert_message
                if can_override:
                    if last_device_name != 'keyboard':
                        frame_alert_message = f"ALERT: {last_device_name.upper()} DETECTED!"
                        frame_alert_status = "alert"; device_detected_now = True
                    elif frame_alert_status == "ok":
                        frame_alert_message = "STATUS: TYPING (Keyboard)"; frame_alert_status = "ok"
                if last_device_box:
                    is_alert_now = frame_alert_status == "alert" and last_device_name != 'keyboard'
                    color = (0, 0, 255) if is_alert_now else (0, 255, 0)
                    cv2.rectangle(frame_small, (last_device_box[0], last_device_box[1]), (last_device_box[2], last_device_box[3]), color, 2)
                    display_text = f"{last_device_name} {last_detected_conf:.2f}" if last_device_name != 'keyboard' else last_device_name
                    cv2.putText(frame_small, display_text, (last_device_box[0], max(20, last_device_box[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


            # --- Emit Alert Status Update ---
            if (frame_alert_message != current_alert_message) or (frame_alert_status != current_alert_status):
                is_flicker = (current_alert_message == "STATUS: FOCUSED" and frame_alert_message.startswith("STATUS: TYPING")) or \
                             (current_alert_message.startswith("STATUS: TYPING") and frame_alert_message == "STATUS: FOCUSED") or \
                             (current_alert_message == "STATUS: FOCUSED" and frame_alert_message.startswith("STATUS: Looking Away")) or \
                             (current_alert_message.startswith("STATUS: Looking Away") and frame_alert_message == "STATUS: FOCUSED")

                if not is_flicker:
                    current_alert_message = frame_alert_message
                    current_alert_status = frame_alert_status
                    socketio.emit('proctor_alert', {'message': current_alert_message, 'status': current_alert_status})
                elif frame_alert_status != current_alert_status:
                     current_alert_status = frame_alert_status


            # --- Emit Instant Focus Percentage ---
            if calibrated:
                 socketio.emit('focus_percentage_update', {'percentage': f"{instant_focus_perc:.1f}%"})
            else:
                 socketio.emit('focus_percentage_update', {'percentage': '---'})

            # --- Time-Based Score Calculation ---
            new_penalty_reason = "FOCUSED"; score_changed = False
            is_truly_focused_final = pose_focused_this_frame and current_alert_status == "ok" and not (device_detected_now and last_device_name != 'keyboard')

            if current_alert_status == "alert":
                 score_changed = True
                 if "MULTIPLE" in current_alert_message: attention_score -= SCORE_PENALTY_MAJOR_PER_SEC * delta_time; new_penalty_reason = "Multiple People"
                 elif "AWAY" in current_alert_message: attention_score -= SCORE_PENALTY_MAJOR_PER_SEC * delta_time; new_penalty_reason = "User Away"
                 elif device_detected_now and last_device_name != 'keyboard': attention_score -= SCORE_PENALTY_MAJOR_PER_SEC * delta_time; new_penalty_reason = "Device Detected"
                 elif "DISTRACTED" in current_alert_message: attention_score -= SCORE_PENALTY_DISTRACT_PER_SEC * delta_time; new_penalty_reason = "Distracted"
                 else: score_changed = False
            elif is_truly_focused_final:
                 if attention_score < 100.0:
                     attention_score += SCORE_RECOVERY_RATE_PER_SEC * delta_time; score_changed = True
                 new_penalty_reason = "FOCUSED"
            elif current_alert_status == "ok" and not is_truly_focused_final: # e.g., Looking Away or Typing
                new_penalty_reason = current_penalty_reason # Keep last reason
                if "Looking Away" in current_alert_message: new_penalty_reason = "Looking Away"
                elif "TYPING" in current_alert_message: new_penalty_reason = "Typing"
                # No score change in this specific state

            attention_score = max(0.0, min(100.0, attention_score))

            if new_penalty_reason != current_penalty_reason: current_penalty_reason = new_penalty_reason

            socketio.emit('score_update', {'score': f"{attention_score:.2f}", 'reason': current_penalty_reason})
            last_score_update_time = now
            # --- End Score Calculation ---

            # --- Stream frame ---
            ret, buffer = cv2.imencode(".jpg", frame_small, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ret: print("Warning: Failed to encode frame."); continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

    except GeneratorExit: print("Client disconnected.")
    except Exception as e: print(f"Error in generate_frames loop: {e}"); import traceback; traceback.print_exc()
    finally: print("Releasing camera."); cam.release()

@app.route("/video_feed")
def video_feed():
    try: return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")
    except Exception as e: print(f"Error starting video feed generator: {e}"); return "Error starting video feed.", 500

if __name__ == "__main__":
    print("Starting AI Proctoring server..."); print(f"Using camera: {CAMERA_SOURCE}")
    print("Access at http://127.0.0.1:5000 or http://<your-local-ip>:5000")
    try: socketio.run(app, debug=False, allow_unsafe_werkzeug=True, use_reloader=False, host='0.0.0.0', port=5000)
    except Exception as e: print(f"Failed to start server: {e}")

