# 🎓 AI Proctoring System

A real-time, industry-grade exam proctoring system powered by **MediaPipe Face Landmarker** and **YOLOv8**. It monitors students via webcam throughout an exam session and automatically flags suspicious behavior including head turns, phone usage, multiple persons in frame, and unexplained absence from the camera.

---

## 📋 Features

| Feature | Description |
|---|---|
| **Head Pose Monitoring** | Detects sustained left/right head turns using yaw angle from facial landmarks |
| **No-Face Detection** | Flags student absence when no face is detected for a sustained period |
| **Multi-Person Detection** | Identifies unauthorized additional individuals in the room |
| **Phone Detection** | Detects mobile devices via YOLOv8 object detection |
| **Environment Check** | Monitors ambient noise level and frame brightness periodically |

---

## 🛠️ Requirements

### Python Version
Python **3.8 – 3.11** recommended.

### Install Dependencies

```bash
pip install opencv-python mediapipe ultralytics sounddevice numpy
```

### Required Model Files

| File | Source |
|---|---|
| `face_landmarker.task` | [MediaPipe Models](https://developers.google.com/mediapipe/solutions/vision/face_landmarker#models) |
| `yolov8n.pt` | Auto-downloaded by `ultralytics` on first run |

Place `face_landmarker.task` in the **same directory** as the script.

---

## 🚀 Usage

```bash
python proctoring_system.py
```

- The webcam monitoring window opens automatically on launch.
- Press **`q`** to stop the session manually at any time.
- The system terminates automatically and prints **"Cheating Confirmed"** when all lives are exhausted.

---

## ⚙️ Configuration Reference

All tunable parameters are defined at the top of the script under `# CONFIG`.

| Parameter | Default | Description |
|---|---|---|
| `LIVES` | `5` | Total violations allowed before the session is terminated |
| `HEAD_VIOLATION_TIME` | `5s` | Sustained yaw deviation duration before a life is deducted |
| `YAW_THRESHOLD` | `30°` | Maximum left/right head rotation before it is considered suspicious |
| `NO_FACE_VIOLATION_TIME` | `5s` | Seconds of missing face before a life is deducted |
| `BRIGHTNESS_MIN` | `40` | Minimum acceptable frame brightness (0–255) |
| `NOISE_MAX` | `0.03` | Maximum ambient audio RMS before a noise warning is raised |
| `FRAME_SKIP` | `3` | Process every Nth frame — balances CPU load vs. responsiveness |
| `ENV_CHECK_INTERVAL` | `10s` | How often the environment check runs |
| `YOLO_CHECK_INTERVAL` | `5s` | How often YOLO object detection runs |
| `YOLO_CONFIRM_HITS` | `2` | Consecutive positive detections required before raising a violation |

---

## 🔍 How It Works

### Feature 1 — Head Pose Monitoring (Yaw Only)
MediaPipe's Face Landmarker returns a 4×4 facial transformation matrix per frame. The system extracts the **yaw** angle (left/right head turn) using a correct ZYX Euler decomposition with a gimbal-lock guard for numerical stability. A violation is only triggered if the student continuously exceeds `YAW_THRESHOLD` for the full `HEAD_VIOLATION_TIME` duration — momentary glances do not count.

### Feature 2 — No-Face Detection
If MediaPipe detects no face for longer than `NO_FACE_VIOLATION_TIME` seconds — whether the student steps away, looks fully downward, or deliberately covers the camera — a life is deducted. The absence timer resets immediately as soon as a face reappears.

### Feature 3 — Environment Check
Every `ENV_CHECK_INTERVAL` seconds the system records a 0.5-second audio sample and computes its RMS noise floor, and evaluates the mean pixel brightness of the current frame. Warnings are printed to the console for low brightness and excessive background noise. The check is intentionally lightweight and non-blocking beyond the short audio sample.

### Feature 4 — Multi-Person & Phone Detection
YOLOv8 nano runs every `YOLO_CHECK_INTERVAL` seconds targeting class `0` (person) and class `67` (cell phone) at a confidence threshold of `0.45`. A **gradual hit-counter decay** system is used for confirmation — a single clean frame between detections reduces the counter by 1 rather than wiping it entirely, preventing a student from evading detection by briefly moving a device out of frame between checks.

---

## 📁 Project Structure

```
proctoring_system.py     # Main monitoring script
face_landmarker.task     # MediaPipe face landmark model
yolov8n.pt               # YOLOv8 nano weights (auto-downloaded on first run)
requirements.txt         # Python dependencies
README.md                # This file
.gitignore               # Git ignore rules
```

---

## 📄 License

For authorized examination and proctoring use only. Ensure full compliance with applicable privacy and data protection laws before deploying in any live environment.