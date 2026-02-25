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

YAW_THRESHOLD = 30             # degrees
PITCH_THRESHOLD = 20           # degrees

BRIGHTNESS_MIN = 40
NOISE_MAX = 0.03

FRAME_SKIP = 3
MODEL_PATH = "face_landmarker.task"

last_yolo_check = time.time()
person_hits = 0
phone_hits = 0

# ---------- MODULE 2 CONFIG ----------
YOLO_MODEL_PATH = "yolov8n.pt"   # nano model (CPU friendly)
YOLO_CHECK_INTERVAL = 5          # seconds
YOLO_CONFIRM_HITS = 2            # anti false-positive

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

# ---------- Helpers ----------
def get_pitch_yaw(transform_matrix):
    r = np.array(transform_matrix).reshape(4, 4)[:3, :3]
    sy = np.sqrt(r[0, 0]**2 + r[1, 0]**2)

    pitch = np.degrees(np.arctan2(-r[2, 0], sy))
    yaw = np.degrees(np.arctan2(r[1, 0], r[0, 0]))

    return pitch, yaw

def check_environment(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = gray.mean()

    audio = sd.rec(int(44100), samplerate=44100, channels=1)
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
last_env_check = time.time()
frame_count = 0

print("Exam monitoring started")

while lives_left > 0:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        continue
    cv2.imshow("Camera Debug View", frame)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)
    suspicious = False
    # ---------- FEATURE 1: Head Pose Monitoring ----------
    if result.facial_transformation_matrixes:
        pitch, yaw = get_pitch_yaw(
            result.facial_transformation_matrixes[0]
        )

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

    # ---------- FEATURE 2: Periodic Environment Check ----------
    if time.time() - last_env_check > ENV_CHECK_INTERVAL:
        check_environment(frame)
        last_env_check = time.time()

    # ---------- FEATURE 3: Multi-Person & Mobile Detection ----------
    if time.time() - last_yolo_check > YOLO_CHECK_INTERVAL:
        last_yolo_check = time.time()

        # YOLO expects BGR image directly
        detections = yolo_model(
            frame,
            conf=0.25,
            iou=0.5,
            classes=[0, 67],   # 0=person, 67=cell phone
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
            person_hits = 0

        if phone_detected:
            phone_hits += 1
        else:
            phone_hits = 0

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

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------- END ----------
if lives_left == 0:
    print("Cheating Confirmed")

cap.release()
cv2.destroyAllWindows()