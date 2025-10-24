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

# --- Configuration ---
# Pose Thresholds
YAW_THRESH = 15
PITCH_DOWN_THRESH = 40
PITCH_UP_THRESH = -8
# Gaze Thresholds (ratio 0.0=Left/Top, 0.5=Center, 1.0=Right/Bottom)
# Horizontal: How far left/right iris is relative to eye corners
GAZE_H_THRESH_LEFT = 0.35 # Iris must be > 35% towards left corner to be looking left
GAZE_H_THRESH_RIGHT = 0.65 # Iris must be > 65% towards right corner (less than 35% from right corner)
# Vertical: How far up/down iris is relative to eye top/bottom (simplified for now)
GAZE_V_THRESH_UP = 0.40 # Simplified: Iris needs to be high in socket
GAZE_V_THRESH_DOWN = 0.70 # Simplified: Iris needs to be low in socket
# Smoothing and Dwell Times
EMA_ALPHA = 0.3
CALIB_N = 20
DISTRACT_DWELL_S = 1.0 # Slightly reduced dwell time as gaze+pose is more sensitive
AWAY_DWELL_S = 1.5
# Device Detection
DEVICE_CLASSES = [67, 73, 76] # phone, book, keyboard
PHONE_CLASS = 67
PHONE_CONF = 0.25
OTHER_CONF = 0.25
YOLO_MIN_MS = 400
YOLO_IMGSZ = 416
DEVICE_HOLD_S = 2.5
# Scoring (Points per Second)
SCORE_RECOVERY_RATE_PER_SEC = 0.25
SCORE_PENALTY_DISTRACT_PER_SEC = 1.0 # Slightly increased penalty for combined distraction
SCORE_PENALTY_MAJOR_PER_SEC = 2.5
# --- End Configuration ---


class CameraGrabber:
    # ... (CameraGrabber class code remains the same) ...
    def __init__(self, src=0, req_w=1280, req_h=720, buffer_size=2):
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened(): raise RuntimeError(f"Could not open webcam: {src}")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
        if req_w and req_h:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, req_w); self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, req_h)
        try: cv2.setUseOptimized(True); cv2.setNumThreads(0)
        except Exception: pass
        self.lock = threading.Lock(); self.latest = None; self.stopped = False
        self.thread = threading.Thread(target=self._reader, daemon=True); self.thread.start()
    def _reader(self):
        while not self.stopped:
            ok, frame = self.cap.read()
            if not ok: print(f"Warn: Frame read fail: {CAMERA_SOURCE}"); time.sleep(0.5); continue
            with self.lock: self.latest = frame
    def read(self):
        with self.lock: return None if self.latest is None else self.latest.copy()
    def release(self):
        self.stopped = True;
        try: self.thread.join(timeout=0.5)
        except Exception: pass
        self.cap.release()

def resize_keep_aspect(img, target_w=640) -> Tuple[Optional[np.ndarray], int, int]:
    # ... (resize_keep_aspect function code remains the same) ...
    if img is None: return None, 0, 0
    h, w = img.shape[:2]
    if w == 0 or h == 0: return None, 0, 0
    if w == target_w: return img, w, h
    scale = target_w / float(w)
    new_w = target_w; new_h = int(round(h * scale))
    if new_w > 0 and new_h > 0:
        try: resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR); return resized, new_w, new_h
        except cv2.error as e: print(f"Resize error: {e}"); return None, 0, 0
    else: return None, 0, 0

# --- Model Initialization ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=4, refine_landmarks=True, # refine_landmarks needed for iris
    min_detection_confidence=0.25, min_tracking_confidence=0.5
)
# Iris landmark indices (relative to the 468 landmarks)
LEFT_IRIS = [474, 475, 476, 477]; RIGHT_IRIS = [469, 470, 471, 472]
LEFT_EYE = [33, 160, 158, 133, 153, 144] # Left_corner, Top_L, Bottom_L, Right_corner, Top_R, Bottom_R
RIGHT_EYE = [362, 385, 387, 263, 373, 380] # Left_corner, Top_L, Bottom_L, Right_corner, Top_R, Bottom_R

try: yolo_model = YOLO('yolov8m.pt'); print("Loaded YOLO: yolov8m.pt")
except Exception:
    try: yolo_model = YOLO('yolov8s.pt'); print("Loaded YOLO: yolov8s.pt (m not found)")
    except Exception: yolo_model = YOLO('yolov8n.pt'); print("Loaded YOLO: yolov8n.pt (m/s not found)")
DEVICE = 0 if torch.cuda.is_available() else 'cpu'; print(f"YOLO device: {DEVICE}")
recognizer = sr.Recognizer(); microphone = sr.Microphone()
# --- End Model Initialization ---

# --- Pose/Gaze Utilities ---
face_3d = np.array([ # Used for head pose
    [0.0,0.0,0.0], [0.0,-330.0,-65.0], [-225.0,170.0,-135.0],
    [225.0,170.0,-135.0], [-150.0,-150.0,-125.0], [150.0,-150.0,-125.0]
], dtype=np.float64)
landmarks_indices = [1, 152, 33, 263, 61, 291] # Indices for head pose landmarks
def rotationMatrixToEulerAngles(R):
    # ... (rotationMatrixToEulerAngles remains the same) ...
    sy=math.sqrt(R[0,0]**2+R[1,0]**2); singular=sy<1e-6
    if not singular:
        roll=math.degrees(math.atan2(R[2,1],R[2,2])); pitch=math.degrees(math.atan2(-R[2,0],sy)); yaw=math.degrees(math.atan2(R[1,0],R[0,0]))
    else:
        roll=math.degrees(math.atan2(-R[1,2],R[1,1])); pitch=math.degrees(math.atan2(-R[2,0],sy)); yaw=0.0
    return yaw, pitch, roll
# --- End Pose/Gaze Utilities ---

# --- State Variables ---
current_alert_message="STATUS: INITIALIZING..."; current_alert_status="ok"
calibrated=False; calib_frames=[]; yaw_base=0.0; pitch_base=0.0
yaw_ema=0.0; pitch_ema=0.0
# Gaze State
gaze_h_ema = 0.5; gaze_v_ema = 0.5 # Assume center initially
distract_start=None; away_start=None; last_seen_face_time=None
last_device_time=0.0; last_device_name=None; last_device_box=None
last_yolo_time=0.0; last_detected_conf=0.0
attention_score=100.0; current_penalty_reason="FOCUSED"
last_score_update_time=time.monotonic()
# --- End State Variables ---

@app.route("/")
def index(): return render_template("index3.html") # Check filename

@socketio.on('recalibrate')
def handle_recalibrate():
    global calibrated, calib_frames, yaw_base, pitch_base, current_alert_message, current_alert_status, attention_score, yaw_ema, pitch_ema, gaze_h_ema, gaze_v_ema
    print("Recalibration triggered.")
    calibrated=False; calib_frames=[]; yaw_base=0.0; pitch_base=0.0
    attention_score=100.0; yaw_ema=0.0; pitch_ema=0.0; gaze_h_ema=0.5; gaze_v_ema=0.5 # Reset gaze EMA
    current_alert_message="RECALIBRATING..."; current_alert_status="calib"
    socketio.emit('proctor_alert', {'message': current_alert_message, 'status': current_alert_status})
    socketio.emit('focus_percentage_update', {'percentage': '---'})
    socketio.emit('score_update', {'score': f"{attention_score:.2f}", 'reason': "RECALIBRATING"})

def calculate_instant_focus_percentage(yaw, pitch, gaze_h, gaze_v):
    """Calculates focus percentage based on head pose AND gaze deviation."""
    yaw_dev = abs(yaw); pitch_dev = abs(pitch)
    # Gaze deviation from center (0.5)
    gaze_h_dev = abs(gaze_h - 0.5)
    gaze_v_dev = abs(gaze_v - 0.5) # Simplified V gaze deviation

    # Normalize deviations to a 0-1 scale relative to thresholds
    yaw_norm = min(1, yaw_dev / YAW_THRESH) if YAW_THRESH > 0 else 0
    pitch_norm = 0
    if pitch > 0 and PITCH_DOWN_THRESH > 0: pitch_norm = min(1, pitch / PITCH_DOWN_THRESH)
    elif pitch < 0 and PITCH_UP_THRESH < 0: pitch_norm = min(1, abs(pitch) / abs(PITCH_UP_THRESH))

    # Normalize gaze deviation relative to its thresholds (distance from center to edge threshold)
    gaze_h_norm = 0
    if gaze_h < 0.5 and (0.5 - GAZE_H_THRESH_LEFT) > 0: # Looking left
        gaze_h_norm = min(1, gaze_h_dev / (0.5 - GAZE_H_THRESH_LEFT))
    elif gaze_h > 0.5 and (GAZE_H_THRESH_RIGHT - 0.5) > 0: # Looking right
        gaze_h_norm = min(1, gaze_h_dev / (GAZE_H_THRESH_RIGHT - 0.5))

    gaze_v_norm = 0 # Simplified vertical gaze check
    if gaze_v < 0.5 and (0.5 - GAZE_V_THRESH_UP) > 0: # Looking up
         gaze_v_norm = min(1, gaze_v_dev / (0.5 - GAZE_V_THRESH_UP))
    elif gaze_v > 0.5 and (GAZE_V_THRESH_DOWN - 0.5) > 0: # Looking down
         gaze_v_norm = min(1, gaze_v_dev / (GAZE_V_THRESH_DOWN - 0.5))

    # Max deviation determines focus loss (scale 0-1)
    # Give slightly more weight to head pose than gaze for percentage
    max_dev = max(yaw_norm, pitch_norm, gaze_h_norm * 0.8, gaze_v_norm * 0.8)

    percentage = max(0.0, (1.0 - max_dev) * 100.0)
    return percentage

def get_gaze_ratios(landmarks, frame_w, frame_h):
    """Calculates horizontal and vertical gaze ratio."""
    if landmarks is None or len(landmarks) <= max(max(LEFT_IRIS), max(RIGHT_IRIS), max(LEFT_EYE), max(RIGHT_EYE)):
         return 0.5, 0.5 # Not enough landmarks

    try:
        # --- Horizontal Ratio ---
        l_corner_x = landmarks[LEFT_EYE[0]].x * frame_w
        r_corner_x = landmarks[LEFT_EYE[3]].x * frame_w
        iris_l_points = np.array([(landmarks[i].x * frame_w, landmarks[i].y * frame_h) for i in LEFT_IRIS])
        # Check if points are valid before mean
        if iris_l_points.size == 0 or np.isnan(iris_l_points).any(): return 0.5, 0.5
        iris_l_center_x = np.mean(iris_l_points[:, 0])

        l_corner_x_r = landmarks[RIGHT_EYE[0]].x * frame_w
        r_corner_x_r = landmarks[RIGHT_EYE[3]].x * frame_w
        iris_r_points = np.array([(landmarks[i].x * frame_w, landmarks[i].y * frame_h) for i in RIGHT_IRIS])
        if iris_r_points.size == 0 or np.isnan(iris_r_points).any(): return 0.5, 0.5
        iris_r_center_x = np.mean(iris_r_points[:, 0])

        # Avoid division by zero if eye corners are the same
        h_ratio_l = (iris_l_center_x - l_corner_x) / (r_corner_x - l_corner_x + 1e-6)
        h_ratio_r = (iris_r_center_x - l_corner_x_r) / (r_corner_x_r - l_corner_x_r + 1e-6)
        avg_h_ratio = np.mean([h_ratio_l, h_ratio_r])

        # --- Vertical Ratio (Simplified) ---
        iris_l_center_y = np.mean(iris_l_points[:, 1])
        iris_r_center_y = np.mean(iris_r_points[:, 1])
        avg_iris_y = np.mean([iris_l_center_y, iris_r_center_y])
        avg_v_ratio = avg_iris_y / (frame_h + 1e-6) # Normalize by frame height

        # Clamp ratios
        avg_h_ratio = max(0.0, min(1.0, avg_h_ratio if not np.isnan(avg_h_ratio) else 0.5))
        avg_v_ratio = max(0.0, min(1.0, avg_v_ratio if not np.isnan(avg_v_ratio) else 0.5))

        return avg_h_ratio, avg_v_ratio
    except IndexError:
        print("Warn: Landmark index out of bounds during gaze calculation.")
        return 0.5, 0.5 # Return center on index error
    except Exception as e:
        print(f"Error in get_gaze_ratios: {e}")
        return 0.5, 0.5


def generate_frames():
    global current_alert_message, current_alert_status, calibrated, calib_frames
    global yaw_base, pitch_base, yaw_ema, pitch_ema, gaze_h_ema, gaze_v_ema # Added gaze EMA
    global distract_start, away_start, last_seen_face_time, last_device_time
    global last_device_name, last_device_box, last_yolo_time, last_detected_conf
    global attention_score, current_penalty_reason, last_score_update_time

    cam = CameraGrabber(src=CAMERA_SOURCE, req_w=1280, req_h=720, buffer_size=2)
    initial_message = "CALIBRATION: Look straight..." if not calibrated else "Connecting..."
    # ... (Initial setup, cam_matrix, MP size calc remain the same) ...
    initial_status = "calib" if not calibrated else "ok"
    current_alert_message = initial_message; current_alert_status = initial_status
    socketio.emit('proctor_alert', {'message': current_alert_message, 'status': current_alert_status})
    if not calibrated: print("Hold still for auto-calibration...")
    print("Recalibrate button available on webpage.")
    cam_matrix = None; dist_coeffs = np.zeros((4, 1))
    _, test_w_display, test_h_display = resize_keep_aspect(np.zeros((720, 1280, 3), dtype=np.uint8), target_w=640)
    MP_PROCESS_W = 480
    MP_PROCESS_H = int(test_h_display * (float(MP_PROCESS_W) / test_w_display)) if test_w_display > 0 else 360
    if MP_PROCESS_H <= 0: MP_PROCESS_H = 360; print("Warn: Using default MP height 360.")
    print(f"MP processing size: {MP_PROCESS_W}x{MP_PROCESS_H}")
    last_score_update_time = time.monotonic()


    try:
        while True:
            frame_raw = cam.read()
            if frame_raw is None: # Handle camera read failure
                current_alert_message = "ERR: NO CAM FEED!"; current_alert_status = "alert"
                socketio.emit('proctor_alert', {'message': current_alert_message, 'status': current_alert_status})
                current_penalty_reason = "Camera Error"
                socketio.emit('score_update', {'score': f"{attention_score:.2f}", 'reason': current_penalty_reason})
                time.sleep(1); continue

            frame_small, FRAME_W, FRAME_H = resize_keep_aspect(frame_raw, target_w=640)
            if frame_small is None or FRAME_W == 0 or FRAME_H == 0: # Handle resize failure
                 print("Warn: Resized frame invalid."); time.sleep(0.1); continue

            if cam_matrix is None: # Initialize camera matrix once
                focal_length = FRAME_W; center = (FRAME_W / 2.0, FRAME_H / 2.0)
                cam_matrix = np.array([[focal_length,0,center[0]], [0,focal_length,center[1]], [0,0,1]], dtype=np.float64)

            now = time.monotonic(); delta_time = now - last_score_update_time
            if delta_time <= 0: delta_time = 0.01 # Prevent negative/zero delta

            # --- Frame Defaults ---
            frame_alert_message = "STATUS: FOCUSED"; frame_alert_status = "ok"
            pose_focused_this_frame = True; gaze_focused_this_frame = True # Add gaze focus flag
            instant_focus_perc = 100.0; current_gaze_h = 0.5; current_gaze_v = 0.5 # Gaze defaults

            # --- Face Mesh ---
            results = None; num_faces = 0; landmarks = None
            if MP_PROCESS_H > 0:
                 try: # Process smaller frame for performance
                     frame_mp = cv2.resize(frame_small, (MP_PROCESS_W, MP_PROCESS_H), interpolation=cv2.INTER_AREA)
                     rgb_mp = cv2.cvtColor(frame_mp, cv2.COLOR_BGR2RGB); rgb_mp.flags.writeable = False
                     results = face_mesh.process(rgb_mp)
                     if results.multi_face_landmarks:
                         num_faces = len(results.multi_face_landmarks)
                         if num_faces == 1: landmarks = results.multi_face_landmarks[0].landmark # Get landmarks for single face
                 except Exception as e: print(f"MP Error: {e}")
            else: print("Warn: Skipping Face Mesh.")

            # --- Core Logic ---
            if not calibrated:
                 frame_alert_message = "CALIBRATING..."; frame_alert_status = "calib"
                 pose_focused_this_frame = False; gaze_focused_this_frame = False; instant_focus_perc = 0.0
                 if num_faces == 1 and landmarks:
                     face_2d_calib = []; valid_calib = True
                     # **FIX**: Correct indentation
                     for idx in landmarks_indices:
                         if idx < len(landmarks):
                             lm = landmarks[idx];
                             # **FIX**: Correct indentation
                             if not (0 <= lm.x <= 1 and 0 <= lm.y <= 1):
                                 valid_calib = False; break
                             x,y=int(lm.x*FRAME_W),int(lm.y*FRAME_H); face_2d_calib.append([x,y])
                         else:
                             valid_calib=False; break
                     if valid_calib and len(face_2d_calib) == 6:
                         face_2d_calib = np.array(face_2d_calib, dtype=np.float64)
                         try: # solvePnP for calibration
                            success_pnp, rvec, tvec = cv2.solvePnP(face_3d, face_2d_calib, cam_matrix, dist_coeffs, cv2.SOLVEPNP_ITERATIVE)
                            if success_pnp:
                                rmat, _ = cv2.Rodrigues(rvec); y_raw, p_raw, _ = rotationMatrixToEulerAngles(rmat)
                                calib_frames.append((y_raw, p_raw))
                                cv2.putText(frame_small, f"Calibrating {len(calib_frames)}/{CALIB_N}", (10, FRAME_H-12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                                if len(calib_frames) >= CALIB_N:
                                    yaw_base=float(np.median([y for y,_ in calib_frames])); pitch_base=float(np.median([p for _,p in calib_frames]))
                                    calibrated=True; print(f"Calibrated: Yaw={yaw_base:.1f}, Pitch={pitch_base:.1f}")
                                    frame_alert_message="STATUS: CALIBRATED"; frame_alert_status="ok"; pose_focused_this_frame=True; gaze_focused_this_frame=True; instant_focus_perc=100.0
                         except cv2.error: pass
            elif num_faces > 1:
                 # ... (Multiple people logic) ...
                 frame_alert_message = "ALERT: MULTIPLE!"; frame_alert_status = "alert"; pose_focused_this_frame=False; gaze_focused_this_frame=False; instant_focus_perc = 0.0
                 last_seen_face_time = now; away_start = None; distract_start = None
            elif num_faces == 0:
                 # ... (Away logic) ...
                 pose_focused_this_frame=False; gaze_focused_this_frame=False; instant_focus_perc = 0.0
                 if away_start is None: away_start = now
                 if last_seen_face_time and (now - last_seen_face_time) < AWAY_DWELL_S: frame_alert_message="STATUS: Searching..."; frame_alert_status="ok"
                 elif (now - away_start) >= AWAY_DWELL_S: frame_alert_message="ALERT: AWAY!"; frame_alert_status="alert"
                 else: frame_alert_message="STATUS: Searching..."; frame_alert_status="ok"
                 distract_start = None
            else: # Single face, calibrated
                 last_seen_face_time = now; away_start = None
                 if landmarks: # Ensure landmarks were successfully obtained
                     face_2d = []; valid = True
                     # **FIX**: Correct indentation
                     for idx in landmarks_indices:
                          if idx < len(landmarks):
                              lm = landmarks[idx];
                              # **FIX**: Correct indentation
                              if not (0<=lm.x<=1 and 0<=lm.y<=1):
                                   valid=False; break
                              x,y=int(lm.x*FRAME_W),int(lm.y*FRAME_H); face_2d.append([x,y])
                          else:
                              valid=False; break
                     if valid and len(face_2d) == 6:
                         face_2d = np.array(face_2d, dtype=np.float64)
                         try: # --- Head Pose Calculation ---
                             success_pnp, rvec, tvec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_coeffs, cv2.SOLVEPNP_ITERATIVE)
                             if success_pnp:
                                 rmat, _ = cv2.Rodrigues(rvec); yaw_raw, pitch_raw, _ = rotationMatrixToEulerAngles(rmat)
                                 yaw = -(yaw_raw - yaw_base) # Yaw inverted
                                 pitch = -(pitch_raw - pitch_base) # Pitch inverted
                                 yaw_ema = (1 - EMA_ALPHA) * yaw_ema + EMA_ALPHA * yaw
                                 pitch_ema = (1 - EMA_ALPHA) * pitch_ema + EMA_ALPHA * pitch

                                 # --- Gaze Calculation ---
                                 gaze_h_raw, gaze_v_raw = get_gaze_ratios(landmarks, FRAME_W, FRAME_H)
                                 gaze_h_ema = (1 - EMA_ALPHA) * gaze_h_ema + EMA_ALPHA * gaze_h_raw
                                 gaze_v_ema = (1 - EMA_ALPHA) * gaze_v_ema + EMA_ALPHA * gaze_v_raw
                                 current_gaze_h, current_gaze_v = gaze_h_ema, gaze_v_ema # Store for percentage calc

                                 # --- Check Pose Thresholds ---
                                 pose_over_thresh = (yaw_ema < -YAW_THRESH or yaw_ema > YAW_THRESH or
                                                     pitch_ema > PITCH_DOWN_THRESH or pitch_ema < PITCH_UP_THRESH)
                                 # --- Check Gaze Thresholds ---
                                 gaze_over_thresh = (gaze_h_ema < GAZE_H_THRESH_LEFT or gaze_h_ema > GAZE_H_THRESH_RIGHT or
                                                     gaze_v_ema < GAZE_V_THRESH_UP or gaze_v_ema > GAZE_V_THRESH_DOWN)

                                 # Update focus flags
                                 pose_focused_this_frame = not pose_over_thresh
                                 gaze_focused_this_frame = not gaze_over_thresh
                                 is_distracted = pose_over_thresh or gaze_over_thresh # Distracted if EITHER is off

                                 if is_distracted:
                                     if distract_start is None: distract_start = now
                                     if (now - distract_start) >= DISTRACT_DWELL_S:
                                         reason = ""
                                         # Prioritize Pose reason if available
                                         if pose_over_thresh:
                                             if yaw_ema < -YAW_THRESH: reason = f"Head Left {int(abs(yaw_ema))}°"
                                             elif yaw_ema > YAW_THRESH: reason = f"Head Right {int(abs(yaw_ema))}°"
                                             elif pitch_ema > PITCH_DOWN_THRESH: reason = f"Head Down {int(abs(pitch_ema))}°"
                                             elif pitch_ema < PITCH_UP_THRESH: reason = f"Head Up {int(abs(pitch_ema))}°"
                                         # If pose is okay, use Gaze reason
                                         elif gaze_over_thresh:
                                             if gaze_h_ema < GAZE_H_THRESH_LEFT: reason = "Gaze Left"
                                             elif gaze_h_ema > GAZE_H_THRESH_RIGHT: reason = "Gaze Right"
                                             elif gaze_v_ema < GAZE_V_THRESH_UP: reason = "Gaze Up"
                                             elif gaze_v_ema > GAZE_V_THRESH_DOWN: reason = "Gaze Down"
                                         else: reason = "Unknown" # Fallback

                                         frame_alert_message = f"ALERT: DISTRACTED ({reason})"; frame_alert_status = "alert"
                                     else: # Still within dwell time
                                         frame_alert_message = "STATUS: Looking Away..."; frame_alert_status = "ok"
                                 else: # Pose AND Gaze are focused
                                     distract_start = None; frame_alert_message = "STATUS: FOCUSED"; frame_alert_status = "ok"

                                 # Draw HUD (Pose)
                                 cv2.putText(frame_small, f"Yaw: {int(yaw_ema)}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                 cv2.putText(frame_small, f"Pitch: {int(pitch_ema)}", (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                             else: # PnP failed
                                  pose_focused_this_frame=False; gaze_focused_this_frame=False; instant_focus_perc = 0.0
                                  frame_alert_message = "STATUS: Pose Fail"; frame_alert_status = "ok"
                         except cv2.error as e: # SolvePnP error
                             pose_focused_this_frame=False; gaze_focused_this_frame=False; instant_focus_perc = 0.0
                             print(f"PnP Error: {e}"); frame_alert_message = "STATUS: Pose Error"; frame_alert_status = "ok"
                     else: # Invalid landmarks
                         pose_focused_this_frame=False; gaze_focused_this_frame=False; instant_focus_perc = 0.0
                         frame_alert_message = "STATUS: Landmark Error"; frame_alert_status = "ok"
                 else: # No landmarks found for single face
                     pose_focused_this_frame=False; gaze_focused_this_frame=False; instant_focus_perc = 0.0
                     frame_alert_message = "STATUS: Face Mesh Error"; frame_alert_status = "ok"

            # --- YOLO ---
            if (now - last_yolo_time) * 1000.0 >= YOLO_MIN_MS: # Runs less often now
                last_yolo_time = now
                try: # Run YOLO detection
                    with torch.inference_mode():
                        results_yolo = yolo_model(frame_small, imgsz=YOLO_IMGSZ, conf=min(PHONE_CONF, OTHER_CONF), classes=DEVICE_CLASSES, device=DEVICE, verbose=False)
                    best_name, best_box, best_priority, best_conf = None, None, 0.0, 0.0
                    for res in results_yolo: # Process results
                        for box in res.boxes:
                            cls_id=int(box.cls[0]); conf=float(box.conf[0]); name=yolo_model.names.get(cls_id, str(cls_id))
                            thr = PHONE_CONF if cls_id == PHONE_CLASS else OTHER_CONF
                            if conf < thr: continue # Skip low confidence
                            priority = conf + (2.0 if cls_id == PHONE_CLASS else 1.0) # Prioritize phone
                            if priority > best_priority: # Found better detection
                                best_priority=priority; best_conf=conf; best_name=name
                                x1,y1,x2,y2 = map(int, box.xyxy[0]); best_box=(x1,y1,x2,y2)
                    if best_name: # Update last known device if found
                        last_device_time=now; last_device_name=best_name; last_device_box=best_box; last_detected_conf=best_conf
                except Exception as e: print(f"YOLO Error: {e}")


            # --- Apply Device Detection Alert ---
            device_detected_now = False
            if last_device_name and (now - last_device_time) <= DEVICE_HOLD_S:
                can_override = frame_alert_status != "alert" or "DISTRACTED" in frame_alert_message or "Looking Away" in frame_alert_message
                if can_override:
                    if last_device_name != 'keyboard': # Don't alert for keyboard
                        frame_alert_message=f"ALERT: {last_device_name.upper()}!"; frame_alert_status="alert"; device_detected_now=True
                    elif frame_alert_status == "ok": # Show typing only if not already alerted/distracted
                        frame_alert_message="STATUS: TYPING"; frame_alert_status="ok"
                if last_device_box:
                    is_alert_device = frame_alert_status == "alert" and last_device_name != 'keyboard'
                    color = (0,0,255) if is_alert_device else (0,255,0) # Red if alert, Green if typing/ok
                    cv2.rectangle(frame_small, (last_device_box[0],last_device_box[1]), (last_device_box[2],last_device_box[3]), color, 2)
                    display_text = f"{last_device_name} {last_detected_conf:.2f}" if last_device_name!='keyboard' else last_device_name
                    cv2.putText(frame_small, display_text, (last_device_box[0], max(20, last_device_box[1]-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


            # --- Emit Alert Status Update ---
            if (frame_alert_message != current_alert_message) or (frame_alert_status != current_alert_status):
                is_flicker = (current_alert_message=="STATUS: FOCUSED" and frame_alert_message.startswith("STATUS: TYPING")) or \
                             (current_alert_message.startswith("STATUS: TYPING") and frame_alert_message=="STATUS: FOCUSED") or \
                             (current_alert_message=="STATUS: FOCUSED" and frame_alert_message.startswith("STATUS: Looking Away")) or \
                             (current_alert_message.startswith("STATUS: Looking Away") and frame_alert_message=="STATUS: FOCUSED")

                if not is_flicker:
                    current_alert_message = frame_alert_message; current_alert_status = frame_alert_status
                    socketio.emit('proctor_alert', {'message': current_alert_message, 'status': current_alert_status})
                elif frame_alert_status != current_alert_status:
                     current_alert_status = frame_alert_status


            # --- Emit Instant Focus Percentage (using pose AND gaze) ---
            if calibrated:
                 instant_focus_perc = calculate_instant_focus_percentage(yaw_ema, pitch_ema, current_gaze_h, current_gaze_v)
                 socketio.emit('focus_percentage_update', {'percentage': f"{instant_focus_perc:.1f}%"})
            else:
                 socketio.emit('focus_percentage_update', {'percentage': '---'})

            # --- Time-Based Score Calculation ---
            new_penalty_reason = "FOCUSED"; score_changed = False
            is_truly_focused_final = pose_focused_this_frame and gaze_focused_this_frame and current_alert_status == "ok" and not (device_detected_now and last_device_name != 'keyboard')

            if current_alert_status == "alert":
                 score_changed = True
                 if "MULTIPLE" in current_alert_message: attention_score -= SCORE_PENALTY_MAJOR_PER_SEC*delta_time; new_penalty_reason="Multiple People"
                 elif "AWAY" in current_alert_message: attention_score -= SCORE_PENALTY_MAJOR_PER_SEC*delta_time; new_penalty_reason="User Away"
                 elif device_detected_now and last_device_name!='keyboard': attention_score -= SCORE_PENALTY_MAJOR_PER_SEC*delta_time; new_penalty_reason="Device Detected"
                 elif "DISTRACTED" in current_alert_message: attention_score -= SCORE_PENALTY_DISTRACT_PER_SEC*delta_time; new_penalty_reason="Distracted" # Includes gaze/pose
                 else: score_changed = False # Unknown alert type
            elif is_truly_focused_final: # Recover only if truly focused
                 if attention_score < 100.0: attention_score += SCORE_RECOVERY_RATE_PER_SEC*delta_time; score_changed = True
                 new_penalty_reason = "FOCUSED"
            elif current_alert_status == "ok" and not is_truly_focused_final: # OK status but pose/gaze off, or typing
                new_penalty_reason = current_penalty_reason # Keep last non-focused reason
                if "Looking Away" in current_alert_message: new_penalty_reason = "Looking Away" # Pose or Gaze off
                elif "TYPING" in current_alert_message: new_penalty_reason = "Typing"
                # Score does not change in this intermediate state

            attention_score = max(0.0, min(100.0, attention_score)) # Clamp 0-100

            if new_penalty_reason != current_penalty_reason: current_penalty_reason = new_penalty_reason

            socketio.emit('score_update', {'score': f"{attention_score:.2f}", 'reason': current_penalty_reason})
            last_score_update_time = now # Update time for next delta
            # --- End Score Calculation ---

            # --- Stream frame ---
            ret, buffer = cv2.imencode(".jpg", frame_small, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ret: print("Warn: Encode fail."); continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

    except GeneratorExit: print("Client disconnected.")
    except Exception as e: print(f"Error in generate_frames: {e}"); import traceback; traceback.print_exc()
    finally: print("Releasing camera."); cam.release()

@app.route("/video_feed")
def video_feed():
    try: return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")
    except Exception as e: print(f"Video feed error: {e}"); return "Video feed error.", 500

if __name__ == "__main__":
    print("Starting server..."); print(f"Using camera: {CAMERA_SOURCE}")
    print("Access at http://127.0.0.1:5000 or http://<local-ip>:5000")
    try: socketio.run(app, debug=False, allow_unsafe_werkzeug=True, use_reloader=False, host='0.0.0.0', port=5000)
    except Exception as e: print(f"Server start fail: {e}")

