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

# --- CONFIGURATION ---
CAMERA_SOURCE = 0  # Change this to your camera source
YAW_THRESH = 20
PITCH_DOWN_THRESH = 40
PITCH_UP_THRESH = -15
EMA_ALPHA = 0.3
CALIB_N = 20
DISTRACT_DWELL_S = 1.2
AWAY_DWELL_S = 1.5

# Gaze Detection Config
EYE_CLOSED_THRESH = 0.18
EYE_CLOSED_DWELL_S = 2.0
GAZE_H_THRESH = 0.15
GAZE_V_THRESH = 0.10
GAZE_DWELL_S = 1.5

# YOLO Config
DEVICE_CLASSES = [67, 73, 76]
PHONE_CLASS = 67
PHONE_CONF = 0.45
OTHER_CONF = 0.45
YOLO_MIN_MS = 250
YOLO_IMGSZ = 416
DEVICE_HOLD_S = 2.5

# --- Time-Based Scoring Config ---
SCORE_RECOVERY_RATE_PER_SEC = 0.5
SCORE_PENALTY_DISTRACT_PER_SEC = 0.75
SCORE_PENALTY_MAJOR_PER_SEC = 2.5
# [NEW] Instant penalty for a tab switch
SCORE_PENALTY_TAB_SWITCH = 10.0

# --- [NEW] Speech Detection Config ---
SPEECH_ENABLE = True            # set False to disable if no mic
SPEECH_MIN_PHRASE_S = 0.40      # ignore very tiny blips
SPEECH_HOLD_S = 2.0             # keep alert active this long after last phrase

# --- [NEW] Liveness Check Config ---
LIVENESS_INTERVAL_S = 60.0     # every 5 minutes
LIVENESS_WINDOW_S = 20.0        # time window to respond
LIVENESS_MOVE_DEG = 15.0        # yaw/pitch delta considered "movement"
LIVENESS_HAND_DWELL_S = 0.6     # hand must be visible this long
LIVENESS_FAIL_HOLD_S = 6.0      # keep "failed" alert on-screen

# --- CameraGrabber CLASS ---
# (This class is unchanged)
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

# --- UTILITY FUNCTIONS ---
# (resize_keep_aspect is unchanged)
def resize_keep_aspect(img, target_w=640) -> Tuple[np.ndarray, int, int]:
    h, w = img.shape[:2]
    if w == target_w:
        return img, w, h
    scale = target_w / float(w)
    new_w = target_w
    new_h = int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized, new_w, new_h

# (get_coords is unchanged)
def get_coords(lms, idx, w, h) -> Tuple[int, int]:
    try:
        lm = lms.landmark[idx]
        return int(lm.x * w), int(lm.y * h)
    except Exception:
        return 0, 0

# (calculate_ear is unchanged)
def calculate_ear(lms, indices, w, h) -> float:
    p1 = get_coords(lms, indices[0], w, h)
    p2 = get_coords(lms, indices[1], w, h)
    p3 = get_coords(lms, indices[2], w, h)
    p4 = get_coords(lms, indices[3], w, h)
    p5 = get_coords(lms, indices[4], w, h)
    p6 = get_coords(lms, indices[5], w, h)
    
    A = np.linalg.norm(np.array(p2) - np.array(p6))
    B = np.linalg.norm(np.array(p3) - np.array(p5))
    C = np.linalg.norm(np.array(p1) - np.array(p4))
    
    if C == 0: return 0.3 
    ear = (A + B) / (2.0 * C)
    return ear

# (calculate_iris_ratios is unchanged)
def calculate_iris_ratios(lms, eye_indices, iris_indices, w, h) -> Tuple[Optional[float], Optional[float]]:
    try:
        p1 = np.array(get_coords(lms, eye_indices[0], w, h)) # Left corner
        p4 = np.array(get_coords(lms, eye_indices[3], w, h)) # Right corner
        p2 = np.array(get_coords(lms, eye_indices[1], w, h)) # Top-left
        p3 = np.array(get_coords(lms, eye_indices[2], w, h)) # Top-right
        p6 = np.array(get_coords(lms, eye_indices[5], w, h)) # Bottom-left
        p5 = np.array(get_coords(lms, eye_indices[4], w, h)) # Bottom-right
        
        eye_left_x = p1[0]
        eye_right_x = p4[0]
        eye_top_y = (p2[1] + p3[1]) / 2.0
        eye_bottom_y = (p5[1] + p6[1]) / 2.0
        
        eye_width = eye_right_x - eye_left_x
        eye_height = eye_bottom_y - eye_top_y
        
        if eye_width == 0 or eye_height == 0:
            return None, None
            
        iris_pts = [np.array(get_coords(lms, idx, w, h)) for idx in iris_indices]
        iris_center = np.mean(iris_pts, axis=0)
        
        h_ratio = (iris_center[0] - eye_left_x) / eye_width
        v_ratio = (iris_center[1] - eye_top_y) / eye_height
        
        return h_ratio, v_ratio
        
    except Exception as e:
        print(f"Error calculating iris ratio: {e}")
        return None, None

# --- INITIALIZE MODELS ---
# (MediaPipe Face Mesh is unchanged)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=4,
    refine_landmarks=True,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.5
)

# [NEW] MediaPipe Hands for liveness
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# (YOLO setup is unchanged)
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

# (Speech Recognition setup is unchanged)
recognizer = sr.Recognizer()
microphone = sr.Microphone()

# --- [NEW] Speech background listener helpers/state ---
speech_stopper = None
talking_active = False
last_speech_time = 0.0

def _speech_callback(recog: sr.Recognizer, audio: sr.AudioData):
    """Called by listen_in_background when a phrase is detected based on VAD."""
    if not SPEECH_ENABLE:
        return
    import time as _time
    global talking_active, last_speech_time
    try:
        raw = audio.get_raw_data()
        dur = len(raw) / float(audio.sample_rate * audio.sample_width)
    except Exception:
        dur = 0.5
    if dur >= SPEECH_MIN_PHRASE_S:
        last_speech_time = _time.monotonic()
        talking_active = True

def start_speech_listener():
    """Start background speech listener once per process."""
    global speech_stopper
    if not SPEECH_ENABLE or speech_stopper is not None:
        return
    try:
        with microphone as source:
            # Calibrate to ambient noise for better VAD
            recognizer.adjust_for_ambient_noise(source, duration=0.8)
        # Non-blocking listener; calls _speech_callback on phrases
        speech_stopper = recognizer.listen_in_background(
            microphone, _speech_callback, phrase_time_limit=3
        )
        print("Speech detection: background listener started.")
    except Exception as e:
        print(f"Speech detection disabled: {e}")
        speech_stopper = None

def stop_speech_listener():
    """Stop background listener if running."""
    global speech_stopper
    if speech_stopper is not None:
        try:
            speech_stopper(wait_for_stop=False)
        except Exception:
            pass
        speech_stopper = None

# --- LANDMARK INDICES ---
# (All landmark indices are unchanged)
face_3d = np.array([
    [0.0, 0.0, 0.0],
    [0.0, -330.0, -65.0],
    [-225.0, 170.0, -135.0],
    [225.0, 170.0, -135.0],
    [-150.0, -150.0, -125.0],
    [150.0, -150.0, -125.0]
], dtype=np.float64)
landmarks_indices = [1, 152, 33, 263, 61, 291]
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 158, 160, 133, 144, 153]
LEFT_IRIS_INDICES = [473, 474, 475, 476, 477]
RIGHT_IRIS_INDICES = [468, 469, 470, 471, 472]

# (rotationMatrixToEulerAngles is unchanged)
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

# --- GLOBAL STATE VARIABLES ---
# (Unchanged)
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
eye_closed_start: Optional[float] = None
gaze_distract_start: Optional[float] = None
last_seen_face_time: Optional[float] = None
last_device_time: float = 0.0
last_device_name: Optional[str] = None
last_device_box: Optional[Tuple[int, int, int, int]] = None
last_yolo_time: float = 0.0
calib_iris_frames = []
calib_iris_h_ratio: float = 0.5
calib_iris_v_ratio: float = 0.5

# --- Scoring Global Variables ---
attention_score = 100.0
current_penalty_reason = "FOCUSED"
last_score_update_time = time.monotonic() 
# [MODIFIED] Added new alert type to counter
alert_counts = {"DISTRACTED": 0, "AWAY": 0, "MULTIPLE": 0, "DEVICE": 0, "EYES_CLOSED": 0, "GAZE_DISTRACTION": 0, "TAB_SWITCH": 0, "TALKING": 0}

# [NEW] Liveness State
last_liveness_time = time.monotonic()
liveness_active = False
liveness_prompt_time = 0.0
liveness_pose_ref = (0.0, 0.0)  # (yaw_ema, pitch_ema) at challenge start
hand_seen_start: Optional[float] = None
liveness_fail_until: float = 0.0

# --- FLASK ROUTES ---
@app.route("/")
def index():
    return render_template("PolySpectra.html")

@socketio.on('recalibrate')
def handle_recalibrate():
    # (Unchanged from your last version)
    global calibrated, calib_frames, yaw_base, pitch_base, current_alert_message, current_alert_status
    global calib_iris_frames, calib_iris_h_ratio, calib_iris_v_ratio
    print("Recalibration triggered by client.")
    calibrated = False
    calib_frames = []
    calib_iris_frames = []
    yaw_base, pitch_base = 0.0, 0.0
    calib_iris_h_ratio, calib_iris_v_ratio = 0.5, 0.5
    
    current_alert_message = "RECALIBRATING: Look straight at camera..."
    current_alert_status = "calib"
    socketio.emit('proctor_alert', {
        'message': current_alert_message,
        'status': current_alert_status
    })

# [NEW] SocketIO handler for tab switching
@socketio.on('tab_switch_alert')
def handle_tab_switch():
    """Handles the tab switch alert from the frontend."""
    global attention_score, alert_counts, current_alert_message, current_alert_status, current_penalty_reason
    
    print("CLIENT ALERT: User switched tabs!")
    
    # Apply an instant, large penalty
    attention_score -= SCORE_PENALTY_TAB_SWITCH
    attention_score = max(0.0, attention_score) # Clamp score
    
    # Log the event
    alert_counts["TAB_SWITCH"] += 1
    
    # Force-update the UI with the new alert and score
    current_alert_message = "ALERT: USER SWITCHED TABS"
    current_alert_status = "alert"
    current_penalty_reason = "Switched Tabs"

    socketio.emit('proctor_alert', {
        'message': current_alert_message,
        'status': current_alert_status
    })
    socketio.emit('score_update', {
        'score': f"{attention_score:.2f}",
        'reason': current_penalty_reason
    })


# --- MAIN VIDEO PROCESSING LOOP ---
def generate_frames():
    """The main processing loop. Generates video frames, alerts, and scores."""
    # [MODIFIED] Added TAB_SWITCH to reset
    global current_alert_message, current_alert_status
    global calibrated, calib_frames, yaw_base, pitch_base, yaw_ema, pitch_ema
    global distract_start, away_start, eye_closed_start, gaze_distract_start, last_seen_face_time
    global last_device_time, last_device_name, last_device_box, last_yolo_time
    global attention_score, current_penalty_reason, last_score_update_time, alert_counts
    global calib_iris_frames, calib_iris_h_ratio, calib_iris_v_ratio
    # [NEW] speech globals
    global talking_active, last_speech_time, speech_stopper
    # [NEW] liveness globals
    global last_liveness_time, liveness_active, liveness_prompt_time, liveness_pose_ref
    global hand_seen_start, liveness_fail_until

    # [MODIFIED] Reset all session variables, including new tab switch counter
    print("New client connected, resetting session state.")
    attention_score = 100.0
    current_penalty_reason = "FOCUSED"
    alert_counts = {"DISTRACTED": 0, "AWAY": 0, "MULTIPLE": 0, "DEVICE": 0, "EYES_CLOSED": 0, "GAZE_DISTRACTION": 0, "TAB_SWITCH": 0, "TALKING": 0}
    last_score_update_time = time.monotonic()
    
    calibrated = False 
    calib_frames, calib_iris_frames = [], []
    yaw_base, pitch_base, yaw_ema, pitch_ema = 0.0, 0.0, 0.0, 0.0
    calib_iris_h_ratio, calib_iris_v_ratio = 0.5, 0.5
    distract_start, away_start, eye_closed_start, gaze_distract_start, last_seen_face_time = None, None, None, None, None
    
    last_device_time = 0.0
    last_device_name, last_device_box = None, None

    # [NEW] Reset liveness state
    last_liveness_time = time.monotonic()
    liveness_active = False
    liveness_prompt_time = 0.0
    liveness_pose_ref = (0.0, 0.0)
    hand_seen_start = None
    liveness_fail_until = 0.0

    # (Camera init is unchanged)
    try:
        cam = CameraGrabber(src=CAMERA_SOURCE, req_w=1280, req_h=720, buffer_size=2)
    except Exception as e:
        print(f"FATAL: Could not start camera. {e}")
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, "CAMERA ERROR", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        ret, buffer = cv2.imencode(".jpg", error_frame)
        if ret: yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
        return

    # [NEW] Start background speech detection (non-blocking)
    start_speech_listener()

    # (Initial client messages are unchanged)
    initial_message = "CALIBRATION: Look straight at camera..." if not calibrated else "Connecting..."
    initial_status = "calib" if not calibrated else "ok"
    current_alert_message = initial_message
    current_alert_status = initial_status
    socketio.emit('proctor_alert', { 'message': current_alert_message, 'status': current_alert_status })
    if not calibrated: print("Hold still and look straight for auto-calibration...")
    print("Or, click 'Recalibrate' button on the webpage to restart.")

    cam_matrix = None
    dist_coeffs = np.zeros((4, 1))

    # (MP processing size calc is unchanged)
    _, test_w, test_h = resize_keep_aspect(np.zeros((720, 1280, 3), dtype=np.uint8), target_w=640)
    MP_PROCESS_W = 480
    MP_PROCESS_H = int(test_h * (float(MP_PROCESS_W) / test_w))
    print(f"MediaPipe processing size: {MP_PROCESS_W}x{MP_PROCESS_H}")
    
    try:
        while True:
            # (Frame capture and resize is unchanged)
            frame_raw = cam.read()
            if frame_raw is None:
                time.sleep(0.01)
                continue
            frame_small, FRAME_W, FRAME_H = resize_keep_aspect(frame_raw, target_w=640)

            # (Camera matrix init is unchanged)
            if cam_matrix is None:
                focal_length = FRAME_W
                center = (FRAME_W / 2.0, FRAME_H / 2.0)
                cam_matrix = np.array([
                    [focal_length, 0, center[0]],
                    [0, focal_length, center[1]],
                    [0, 0, 1]
                ], dtype=np.float64)

            now = time.monotonic()
            delta_time = now - last_score_update_time 

            # IMPORTANT: Don't reset the alert message here,
            # as it could overwrite an alert from the tab_switch handler
            # We'll set it to "FOCUSED" only if no other alerts are active.
            frame_alert_message = current_alert_message 
            frame_alert_status = current_alert_status

            # (Face Mesh processing is unchanged)
            frame_mp = cv2.resize(frame_small, (MP_PROCESS_W, MP_PROCESS_H), interpolation=cv2.INTER_AREA)
            rgb_mp = cv2.cvtColor(frame_mp, cv2.COLOR_BGR2RGB)
            rgb_mp.flags.writeable = False
            results = face_mesh.process(rgb_mp)
            
            num_faces = len(results.multi_face_landmarks) if results.multi_face_landmarks else 0

            # [NEW] Hands detection for liveness
            results_hands = hands.process(rgb_mp)
            hand_present = bool(getattr(results_hands, "multi_hand_landmarks", None))

            # --- Detection Logic (Priority-based) ---
            
            # (Priority 1: Calibration - unchanged from last version)
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
                                 
                                 l_h_ratio, l_v_ratio = calculate_iris_ratios(lms, LEFT_EYE_INDICES, LEFT_IRIS_INDICES, FRAME_W, FRAME_H)
                                 r_h_ratio, r_v_ratio = calculate_iris_ratios(lms, RIGHT_EYE_INDICES, RIGHT_IRIS_INDICES, FRAME_W, FRAME_H)
                                 
                                 if l_h_ratio is not None and r_h_ratio is not None:
                                     calib_iris_frames.append(((l_h_ratio + r_h_ratio) / 2.0, (l_v_ratio + r_v_ratio) / 2.0))

                                 cv2.putText(frame_small, f"Calibrating... {len(calib_frames)}/{CALIB_N}", (10, FRAME_H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                                 
                                 if len(calib_frames) >= CALIB_N:
                                     yaw_base = float(np.median([y for y, _ in calib_frames]))
                                     pitch_base = float(np.median([p for _, p in calib_frames]))
                                     
                                     if calib_iris_frames:
                                         calib_iris_h_ratio = float(np.median([h for h, _ in calib_iris_frames]))
                                         calib_iris_v_ratio = float(np.median([v for _, v in calib_iris_frames]))

                                     calibrated = True
                                     print(f"Calibration done: yaw_base={yaw_base:.1f}, pitch_base={pitch_base:.1f}")
                                     print(f"Iris calib done: h_ratio={calib_iris_h_ratio:.2f}, v_ratio={calib_iris_v_ratio:.2f}")
                                     frame_alert_message = "STATUS: CALIBRATED. Monitoring..."
                                     frame_alert_status = "ok"
                         except cv2.error: pass 
            
            # (Priority 2: Multiple People - unchanged)
            elif num_faces > 1:
                frame_alert_message = "ALERT: MULTIPLE PEOPLE DETECTED!"
                frame_alert_status = "alert"
                last_seen_face_time = now
                away_start, distract_start, eye_closed_start, gaze_distract_start = None, None, None, None
            
            # (Priority 3: Away - unchanged)
            elif num_faces == 0:
                if away_start is None: away_start = now
                if last_seen_face_time and (now - last_seen_face_time) < AWAY_DWELL_S:
                    frame_alert_message = "STATUS: Searching for face..."
                    frame_alert_status = "ok"
                elif (now - away_start) >= AWAY_DWELL_S:
                    frame_alert_message = "ALERT: STUDENT AWAY!"
                    frame_alert_status = "alert"
                else: 
                    frame_alert_message = "STATUS: Searching for face..."
                    frame_alert_status = "ok"
                distract_start, eye_closed_start, gaze_distract_start = None, None, None
            
            # (Priority 4: Pose & Gaze Estimation - unchanged)
            else: # Exactly one face, and calibrated
                last_seen_face_time = now
                away_start = None
                lms = results.multi_face_landmarks[0]
                
                # --- Head Pose Logic ---
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
                            yaw, pitch = (yaw_raw - yaw_base), -(pitch_raw - pitch_base)
                            yaw_ema = (1 - EMA_ALPHA) * yaw_ema + EMA_ALPHA * yaw
                            pitch_ema = (1 - EMA_ALPHA) * pitch_ema + EMA_ALPHA * pitch

                            over_thresh = (yaw_ema < -YAW_THRESH or yaw_ema > YAW_THRESH or
                                           pitch_ema > PITCH_DOWN_THRESH or pitch_ema < PITCH_UP_THRESH)

                            if over_thresh:
                                if distract_start is None: distract_start = now
                                if (now - distract_start) >= DISTRACT_DWELL_S:
                                    if   yaw_ema < -YAW_THRESH: reason = f"Left {int(abs(yaw_ema))}°"
                                    elif yaw_ema > YAW_THRESH: reason = f"Right {int(abs(yaw_ema))}°"
                                    elif pitch_ema > PITCH_DOWN_THRESH: reason = f"Down {int(abs(pitch_ema))}°"
                                    else: reason = f"Up {int(abs(pitch_ema))}°"
                                    frame_alert_message, frame_alert_status = f"ALERT: DISTRACTED ({reason})", "alert"
                                else: 
                                    frame_alert_message, frame_alert_status = "STATUS: FOCUSED", "ok"
                                eye_closed_start, gaze_distract_start = None, None
                            
                            else: # Head pose is OK
                                distract_start = None
                                
                                # --- Gaze & Eye Closure Checks ---
                                gaze_alert = False # Flag
                                try:
                                    # (1) Eye Closure Check (EAR)
                                    left_ear = calculate_ear(lms, LEFT_EYE_INDICES, FRAME_W, FRAME_H)
                                    right_ear = calculate_ear(lms, RIGHT_EYE_INDICES, FRAME_W, FRAME_H)
                                    avg_ear = (left_ear + right_ear) / 2.0
                                    cv2.putText(frame_small, f"EAR: {avg_ear:.2f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                                    if avg_ear < EYE_CLOSED_THRESH:
                                        if eye_closed_start is None: eye_closed_start = now
                                        if (now - eye_closed_start) >= EYE_CLOSED_DWELL_S:
                                            frame_alert_message, frame_alert_status = "ALERT: EYES CLOSED", "alert"
                                            gaze_distract_start = None
                                        # else: dwell time
                                    
                                    else: # Eyes are OPEN
                                        eye_closed_start = None
                                        
                                        # (2) Iris Gaze Check
                                        l_h_ratio, l_v_ratio = calculate_iris_ratios(lms, LEFT_EYE_INDICES, LEFT_IRIS_INDICES, FRAME_W, FRAME_H)
                                        r_h_ratio, r_v_ratio = calculate_iris_ratios(lms, RIGHT_EYE_INDICES, RIGHT_IRIS_INDICES, FRAME_W, FRAME_H)

                                        if l_h_ratio is not None and r_h_ratio is not None:
                                            avg_h_ratio, avg_v_ratio = (l_h_ratio + r_h_ratio) / 2.0, (l_v_ratio + r_v_ratio) / 2.0
                                            h_delta, v_delta = avg_h_ratio - calib_iris_h_ratio, avg_v_ratio - calib_iris_v_ratio
                                            
                                            cv2.putText(frame_small, f"Gaze H: {h_delta:+.2f}", (10, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                            cv2.putText(frame_small, f"Gaze V: {v_delta:+.2f}", (10, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                                            if abs(h_delta) > GAZE_H_THRESH or abs(v_delta) > GAZE_V_THRESH:
                                                if gaze_distract_start is None: gaze_distract_start = now
                                                if (now - gaze_distract_start) >= GAZE_DWELL_S:
                                                    if abs(h_delta) > abs(v_delta): gaze_reason = "Left" if h_delta < 0 else "Right"
                                                    else: gaze_reason = "Up" if v_delta < 0 else "Down"
                                                    frame_alert_message, frame_alert_status = f"ALERT: GAZE DISTRACTION ({gaze_reason})", "alert"
                                                    gaze_alert = True 
                                                # else: dwell time
                                            else:
                                                gaze_distract_start = None
                                        else:
                                            gaze_distract_start = None 
                                
                                except Exception as e:
                                    print(f"Gaze/EAR calculation error: {e}")
                                    eye_closed_start, gaze_distract_start = None, None
                                
                                # Set "FOCUSED" status
                                if frame_alert_status != "alert" and not gaze_alert:
                                    # [MODIFIED] Check if we are recovering from a tab switch
                                    if "TAB_SWITCH" in current_alert_message:
                                        frame_alert_message, frame_alert_status = "STATUS: FOCUSED", "ok"
                                    # Only set to FOCUSED if not already in an alert state
                                    elif current_alert_status != 'alert':
                                         frame_alert_message, frame_alert_status = "STATUS: FOCUSED", "ok"


                            cv2.putText(frame_small, f"Yaw: {int(yaw_ema)}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            cv2.putText(frame_small, f"Pitch: {int(pitch_ema)}", (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        else: # Pose estimation failed
                            frame_alert_message, frame_alert_status = "STATUS: Pose Estimation Failed", "ok"
                            distract_start, eye_closed_start, gaze_distract_start = None, None, None
                    
                    except cv2.error as e:
                        print(f"solvePnP error: {e}")
                        frame_alert_message, frame_alert_status = "STATUS: Pose Calc Error", "ok"
                        distract_start, eye_closed_start, gaze_distract_start = None, None, None
                
                else: # Landmarks unclear
                    frame_alert_message, frame_alert_status = "STATUS: Landmarks Unclear", "ok"
                    distract_start, eye_closed_start, gaze_distract_start = None, None, None
            
            # (YOLO device detection logic is unchanged)
            if (now - last_yolo_time) * 1000.0 >= YOLO_MIN_MS:
                last_yolo_time = now
                try:
                    with torch.inference_mode():
                        results_yolo = yolo_model(frame_small, imgsz=YOLO_IMGSZ, conf=min(PHONE_CONF, OTHER_CONF), classes=DEVICE_CLASSES, device=DEVICE, verbose=False)
                    best_name, best_box, best_priority = None, None, 0.0
                    for res in results_yolo:
                        for box in res.boxes:
                            cls_id, conf = int(box.cls[0]), float(box.conf[0])
                            name = yolo_model.names.get(cls_id, str(cls_id))
                            thr = PHONE_CONF if cls_id == PHONE_CLASS else OTHER_CONF
                            if conf < thr: continue
                            priority = conf + (2.0 if cls_id == PHONE_CLASS else 1.0)
                            if priority > best_priority:
                                best_priority, best_name = priority, name
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                best_box = (x1, y1, x2, y2)
                    if best_name is not None:
                        last_device_time, last_device_name, last_device_box = now, best_name, best_box
                except Exception as e:
                    print(f"YOLO error: {e}")

            # (Applying device detection alert logic is unchanged)
            if last_device_name and (now - last_device_time) <= DEVICE_HOLD_S:
                if frame_alert_status == "ok":
                    if last_device_name != 'keyboard':
                        frame_alert_message, frame_alert_status = f"ALERT: {last_device_name.upper()} DETECTED!", "alert"
                    else: 
                        frame_alert_message, frame_alert_status = "STATUS: TYPING (Keyboard)", "ok"
                
                if last_device_box:
                    color = (0, 0, 255) if frame_alert_status == "alert" and last_device_name != 'keyboard' else (0, 255, 0)
                    cv2.rectangle(frame_small, (last_device_box[0], last_device_box[1]), (last_device_box[2], last_device_box[3]), color, 2)
                    cv2.putText(frame_small, f"{last_device_name}", (last_device_box[0], max(20, last_device_box[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # --- [NEW] Talking detection (short-lived) ---
            if talking_active and (now - last_speech_time) <= SPEECH_HOLD_S:
                if frame_alert_status == "ok":
                    frame_alert_message = "ALERT: TALKING DETECTED"
                    frame_alert_status = "alert"
            else:
                talking_active = False

            # --- [NEW] Liveness Challenge ---
            # Start challenge only when calibrated and one face is visible
            if calibrated and (num_faces == 1):
                # Trigger challenge at interval when not already active
                if (not liveness_active) and ((now - last_liveness_time) >= LIVENESS_INTERVAL_S):
                    liveness_active = True
                    liveness_prompt_time = now
                    liveness_pose_ref = (yaw_ema, pitch_ema)
                    hand_seen_start = None
                    # Show prompt if no other alert currently overrides
                    if frame_alert_status == "ok":
                        frame_alert_message = "CHALLENGE: RAISE HAND OR NOD"
                        frame_alert_status = "alert"

            # If challenge is active, look for response
            if liveness_active:
                # Accept if hand is present with dwell
                if hand_present:
                    if hand_seen_start is None:
                        hand_seen_start = now
                    elif (now - hand_seen_start) >= LIVENESS_HAND_DWELL_S:
                        # Success
                        liveness_active = False
                        last_liveness_time = now
                        if frame_alert_status != "alert":  # don't override another alert
                            frame_alert_message = "STATUS: LIVENESS VERIFIED"
                            frame_alert_status = "ok"
                else:
                    hand_seen_start = None

                # Or accept if sufficient head movement from reference
                dyaw = abs(yaw_ema - liveness_pose_ref[0])
                dpitch = abs(pitch_ema - liveness_pose_ref[1])
                if dyaw >= LIVENESS_MOVE_DEG or dpitch >= LIVENESS_MOVE_DEG:
                    liveness_active = False
                    last_liveness_time = now
                    if frame_alert_status != "alert":
                        frame_alert_message = "STATUS: LIVENESS VERIFIED"
                        frame_alert_status = "ok"

                # Timeout -> fail
                if liveness_active and ((now - liveness_prompt_time) >= LIVENESS_WINDOW_S):
                    liveness_active = False
                    liveness_fail_until = now + LIVENESS_FAIL_HOLD_S

            # Keep fail alert on screen (and allow scoring)
            if liveness_fail_until > now:
                if frame_alert_status == "ok":
                    frame_alert_message = "ALERT: LIVENESS FAILED"
                    frame_alert_status = "alert"

            # --- Emit Alert Status Update ---
            if (frame_alert_message != current_alert_message) or (frame_alert_status != current_alert_status):
                
                # [MODIFIED] Count the transition to a new alert type
                if frame_alert_status == "alert":
                    if "MULTIPLE" in frame_alert_message and "MULTIPLE" not in current_alert_message:
                         alert_counts["MULTIPLE"] += 1
                    elif "AWAY" in frame_alert_message and "AWAY" not in current_alert_message:
                         alert_counts["AWAY"] += 1
                    elif "DETECTED" in frame_alert_message and "DETECTED" not in current_alert_message:
                         alert_counts["DEVICE"] += 1
                    elif "DISTRACTED" in frame_alert_message and "DISTRACTED" not in current_alert_message:
                         alert_counts["DISTRACTED"] += 1
                    elif "EYES CLOSED" in frame_alert_message and "EYES CLOSED" not in current_alert_message:
                         alert_counts["EYES_CLOSED"] += 1
                    elif "GAZE DISTRACTION" in frame_alert_message and "GAZE DISTRACTION" not in current_alert_message:
                         alert_counts["GAZE_DISTRACTION"] += 1
                    elif "TALKING" in frame_alert_message and "TALKING" not in current_alert_message:
                         alert_counts["TALKING"] += 1
                    elif "TAB SWITCH" in frame_alert_message and "TAB SWITCH" not in current_alert_message:
                         alert_counts["TAB_SWITCH"] += 1    
                    # NOTE: TAB_SWITCH is counted in its own handler
                    # NOTE: CHALLENGE / LIVENESS FAILED are not counted

                current_alert_message, current_alert_status = frame_alert_message, frame_alert_status
                socketio.emit('proctor_alert', {
                    'message': current_alert_message,
                    'status': current_alert_status
                })
            
            # --- Time-Based Score Calculation ---
            new_penalty_reason = "FOCUSED"
            # [MODIFIED] Added TAB_SWITCH and liveness to penalty logic
            if current_alert_status == "alert":
                 if "CHALLENGE" in current_alert_message:
                     # No penalty during liveness challenge prompt
                     new_penalty_reason = "Liveness Challenge"
                 elif "MULTIPLE" in current_alert_message:
                     attention_score -= SCORE_PENALTY_MAJOR_PER_SEC * delta_time
                     new_penalty_reason = "Multiple People"
                 elif "AWAY" in current_alert_message:
                     attention_score -= SCORE_PENALTY_MAJOR_PER_SEC * delta_time
                     new_penalty_reason = "User Away"
                 elif "DETECTED" in current_alert_message:
                     attention_score -= SCORE_PENALTY_MAJOR_PER_SEC * delta_time
                     new_penalty_reason = "Device Detected"
                 elif "DISTRACTED" in current_alert_message:
                     attention_score -= SCORE_PENALTY_DISTRACT_PER_SEC * delta_time
                     new_penalty_reason = "Distracted"
                 elif "EYES CLOSED" in current_alert_message:
                     attention_score -= SCORE_PENALTY_DISTRACT_PER_SEC * delta_time
                     new_penalty_reason = "Eyes Closed"
                 elif "GAZE DISTRACTION" in current_alert_message:
                     attention_score -= SCORE_PENALTY_DISTRACT_PER_SEC * delta_time
                     new_penalty_reason = "Gaze Distraction"
                 elif "TAB_SWITCH" in current_alert_message:
                      attention_score -= SCORE_PENALTY_MAJOR_PER_SEC * delta_time
                      new_penalty_reason = "Switched Tabs"# The initial 10-point penalty was applied instantly.
                 elif "TALKING" in current_alert_message:
                     attention_score -= SCORE_PENALTY_MAJOR_PER_SEC * delta_time
                     new_penalty_reason = "Talking Detected"
                 elif "LIVENESS FAILED" in current_alert_message:
                     attention_score -= SCORE_PENALTY_MAJOR_PER_SEC * delta_time
                     new_penalty_reason = "Liveness Failed"
            
            elif current_alert_status == "ok":
                 attention_score += SCORE_RECOVERY_RATE_PER_SEC * delta_time
                 new_penalty_reason = "FOCUSED"
            
            attention_score = max(0.0, min(100.0, attention_score))
            
            if new_penalty_reason != current_penalty_reason:
                current_penalty_reason = new_penalty_reason
            
            socketio.emit('score_update', {
                'score': f"{attention_score:.2f}",
                'reason': current_penalty_reason
            })
            
            last_score_update_time = now

            # (Stream frame logic is unchanged)
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
        # --- [MODIFIED] Report Generation on Disconnect ---
        print("Client disconnected. Generating report...")
        try:
            report_filename = f"proctoring_report_{int(time.time())}.txt"
            final_score = attention_score
            
            report_content = f"""
Proctoring Session Report
=========================
Report Generated: {time.asctime()}
Final Trust Score: {final_score:.2f}

Alert Summary (Total Occurrences)
---------------------------------
Distractions (Pose): {alert_counts['DISTRACTED']}
Eyes Closed:         {alert_counts['EYES_CLOSED']}
Gaze Distraction:    {alert_counts['GAZE_DISTRACTION']}
User Away:           {alert_counts['AWAY']}
Multiple People:     {alert_counts['MULTIPLE']}
Talking Detected:    {alert_counts['TALKING']}
Tab Switches:        {alert_counts['TAB_SWITCH']}
"""

            with open(report_filename, "w") as f:
                f.write(report_content.strip())
            print(f"Successfully saved report to {report_filename}")
        except Exception as e:
            print(f"Could not write report: {e}")
        
        # [NEW] Stop background speech listener
        stop_speech_listener()

        print("Releasing camera.")
        try:
            hands.close()
        except Exception:
            pass
        cam.release()
        face_mesh.close()

# (video_feed route is unchanged)
@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

# (main run block is unchanged)
if __name__ == "__main__":
    print("Starting the AI Proctoring server...")
    print(f"Attempting to use camera source: {CAMERA_SOURCE}")
    print("Go to http://127.0.0.1:5000 in your browser.")
    socketio.run(app, debug=False, allow_unsafe_werkzeug=True, host='0.0.0.0', port=5000)
