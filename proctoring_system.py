import cv2
import time
import numpy as np
import sounddevice as sd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO

# ================= CONFIG =================
LIVES = 5
HEAD_VIOLATION_TIME = 5        # seconds
ENV_CHECK_INTERVAL = 10        # seconds

YAW_THRESHOLD = 30             # degrees (yaw only — left/right head turn)
# PITCH_THRESHOLD removed — intentionally yaw-only design

BRIGHTNESS_MIN = 40
NOISE_MAX = 0.03

FRAME_SKIP = 3
MODEL_PATH = "face_landmarker.task"

# ---------- MODULE 2 CONFIG ----------
YOLO_MODEL_PATH = "yolov8n.pt"   # nano model (CPU friendly)
YOLO_CHECK_INTERVAL = 5          # seconds
YOLO_CONFIRM_HITS = 2            # anti false-positive

# ---------- No-Face Config ----------
NO_FACE_VIOLATION_TIME = 5       # seconds before absent face triggers a violation

# ---------- MediaPipe Face Landmarker ----------
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_facial_transformation_matrixes=True,
    num_faces=1
)
landmarker = vision.FaceLandmarker.create_from_options(options)

# ---------- YOLO Object Detector ----------
yolo_model = YOLO(YOLO_MODEL_PATH)

# FIX 7: Initialize time-sensitive state AFTER models are fully loaded
# so the first YOLO check doesn't fire prematurely during startup lag
last_yolo_check = time.time()
person_hits = 0
phone_hits = 0


# ---------- Helpers ----------

def get_yaw(transform_matrix):
    """
    Extract yaw (left/right head turn) from a 4x4 facial transformation matrix.

    Correct ZYX Euler decomposition:
      sy  = sqrt(r[0,0]^2 + r[1,0]^2)
      yaw = arctan2(-r[2,0], sy)          <- correct yaw axis

    FIX (Critical): Original used arctan2(r[1,0], r[0,0]) for yaw —
    that is the roll formula. It produced wrong angle values and
    therefore wrong violation triggers. Corrected here.

    FIX: Added gimbal-lock guard (sy < 1e-6). When the face is nearly
    perpendicular to the camera, sy collapses and arctan2 becomes
    numerically unstable. Fallback returns 0.0 (no false violation).

    FIX: np.asarray used instead of np.array to avoid a redundant
    memory allocation on every processed frame.
    """
    r = np.asarray(transform_matrix).reshape(4, 4)[:3, :3]
    sy = np.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)

    if sy < 1e-6:
        # Gimbal-lock edge case — cannot reliably extract yaw, skip frame
        return 0.0

    yaw = np.degrees(np.arctan2(-r[2, 0], sy))
    return yaw


def check_environment(frame):
    """
    Checks frame brightness and ambient audio noise level.

    FIX: Audio recorded for 0.5s instead of 1.0s.
    Original blocked the entire main loop for a full second every
    ENV_CHECK_INTERVAL. 0.5s is sufficient for a reliable RMS reading
    and halves the dead-time gap in monitoring.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = gray.mean()

    # FIX: 0.5s sample — reliable RMS, half the blocking time
    audio = sd.rec(int(44100 * 0.5), samplerate=44100, channels=1, dtype='float32')
    sd.wait()
    noise = np.sqrt(np.mean(audio ** 2))

    if brightness < BRIGHTNESS_MIN:
        print("Warning: Low brightness")

    if noise > NOISE_MAX:
        print("Warning: High ambient noise")


# ---------- Main ----------
cap = cv2.VideoCapture(0)

lives_left = LIVES
violation_start = None
no_face_start = None           # FIX: track when face disappears from frame
last_env_check = time.time()
frame_count = 0

print("Exam monitoring started")

while lives_left > 0:
    ret, frame = cap.read()
    if not ret:
        # FIX: Log camera failure clearly — distinguishable from a normal quit
        print("❌ Camera read failed. Session terminated.")
        break

    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    suspicious = False

    # ---------- FEATURE 1: Head Pose Monitoring (Yaw Only) ----------
    if result.facial_transformation_matrixes:
        no_face_start = None   # face is present — reset absence timer

        yaw = get_yaw(result.facial_transformation_matrixes[0])
        suspicious = abs(yaw) > YAW_THRESHOLD

        if suspicious:
            if violation_start is None:
                violation_start = time.time()
            elif time.time() - violation_start > HEAD_VIOLATION_TIME:
                lives_left -= 1
                violation_start = None
                print(f"❌ Head pose violation. Lives left: {lives_left}")
        else:
            violation_start = None

    else:
        # FIX: No face detected — start or continue absence timer.
        # Original silently ignored this, allowing a student to look
        # fully away or cover the camera without any penalty.
        violation_start = None  # head-pose timer irrelevant when no face
        if no_face_start is None:
            no_face_start = time.time()
        elif time.time() - no_face_start > NO_FACE_VIOLATION_TIME:
            lives_left -= 1
            no_face_start = None
            print(f"❌ Face not detected (absent/covered). Lives left: {lives_left}")

    # ---------- FEATURE 2: Periodic Environment Check ----------
    if time.time() - last_env_check > ENV_CHECK_INTERVAL:
        check_environment(frame)
        last_env_check = time.time()

    # ---------- FEATURE 3: Multi-Person & Mobile Detection ----------
    if time.time() - last_yolo_check > YOLO_CHECK_INTERVAL:
        last_yolo_check = time.time()

        detections = yolo_model(
            frame,
            conf=0.45,          # FIX: raised from 0.25 — reduces false positives
                                 # (books/reflections mis-triggering phone/person alerts)
                                 # 0.45 still reliably catches genuine detections
            iou=0.5,
            classes=[0, 67],    # 0=person, 67=cell phone
            verbose=False
        )

        person_count = 0
        phone_detected = False

        for det in detections:
            if det.boxes is None:
                continue

            for cls in det.boxes.cls:
                cls_id = int(cls)
                if cls_id == 0:
                    person_count += 1
                elif cls_id == 67:
                    phone_detected = True

        # --- Violation Confirmation Logic ---
        violation = False

        if person_count > 1:
            person_hits += 1
        else:
            # FIX: Gradual decay instead of hard reset to 0.
            # Original: one clean frame wiped all confirmation progress.
            # A student could wave a phone in/out of frame across checks
            # and never accumulate enough hits to trigger a violation.
            person_hits = max(0, person_hits - 1)

        if phone_detected:
            phone_hits += 1
        else:
            phone_hits = max(0, phone_hits - 1)   # FIX: same gradual decay

        if person_hits >= YOLO_CONFIRM_HITS:
            violation = True
            print("❌ Multiple persons detected")
            person_hits = 0

        if phone_hits >= YOLO_CONFIRM_HITS:
            violation = True
            print("❌ Mobile phone detected")
            phone_hits = 0

        if violation:
            lives_left -= 1
            print(f"❌ Object violation. Lives left: {lives_left}")

    # FIX: imshow moved to after all processing — consistent frame/waitKey order
    cv2.imshow("Camera Debug View", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------- END ----------
if lives_left == 0:
    print("Cheating Confirmed")

cap.release()
cv2.destroyAllWindows()