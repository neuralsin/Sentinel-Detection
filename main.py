

from ultralytics import YOLO
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort

def run_detection_tracking(video_source=0):
    model = YOLO('yolov8n.pt')  # YOLOv8 nano for speed
    tracker = DeepSort(max_age=30)

    cap = cv2.VideoCapture(video_source)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            if conf > 0.3:
                # Deep SORT wants [x, y, width, height]
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))

        tracks = tracker.update_tracks(detections, frame=frame)
        for track in tracks:
            if not track.is_confirmed():
                continue
            x, y, w, h = map(int, track.to_ltwh())
            track_id = track.track_id
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{track_id}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Sentinel - Detection & Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_detection_tracking()


import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

def get_pose_landmarks(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    if results.pose_landmarks:
        # Return list of (x,y,z) normalized landmarks
        return [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]
    return None

def run_behavior_detection(video_source=0):
    cap = cv2.VideoCapture(video_source)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        landmarks = get_pose_landmarks(frame)
        if landmarks:
            # Just drawing landmarks for now
            mp.solutions.drawing_utils.draw_landmarks(
                frame, pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).pose_landmarks, mp_pose.POSE_CONNECTIONS
            )
            # Here you’d collect sequences of landmarks over time for behavior analysis (LSTM/Transformer)
            # For now, just print a simple message
            cv2.putText(frame, "Pose detected - track this shit for anomalies", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        cv2.imshow("Sentinel - Behavior Anomaly Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_behavior_detection()


import numpy as np
import sounddevice as sd
import tensorflow as tf
import tensorflow_hub as hub
import queue
import time

# Load YAMNet model from TF Hub
yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)

# YAMNet class names (partial, you can extend)
class_map_path = tf.keras.utils.get_file('yamnet_class_map.csv',
                                         'https://raw.githubusercontent.com/auditory/yamnet/master/yamnet_class_map.csv')
class_names = []
with open(class_map_path) as f:
    for line in f.readlines()[1:]:
        class_names.append(line.strip().split(',')[2])

# Audio params
SAMPLE_RATE = 16000
BUFFER_DURATION = 1.0  # seconds
buffer_size = int(SAMPLE_RATE * BUFFER_DURATION)
audio_queue = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"Audio error: {status}")
    audio_queue.put(indata.copy())

def run_audio_detection():
    print("Starting audio event detection... Press Ctrl+C to stop")

    with sd.InputStream(channels=1, samplerate=SAMPLE_RATE, callback=audio_callback):
        try:
            while True:
                if audio_queue.qsize() > 0:
                    audio_chunk = audio_queue.get()
                    audio_chunk = np.squeeze(audio_chunk)

                    # Run YAMNet
                    scores, embeddings, spectrogram = yamnet_model(audio_chunk)
                    scores_np = scores.numpy()
                    mean_scores = np.mean(scores_np, axis=0)
                    top_index = np.argmax(mean_scores)
                    top_class = class_names[top_index]
                    top_score = mean_scores[top_index]

                    # Threshold to catch loud shit only
                    if top_score > 0.3 and top_class in ['Gunshot, gunfire',
                                                         'Siren',
                                                         'Glass breaking, smashing',
                                                         'Scream']:
                        print(f"[ALERT] Detected audio event: {top_class} with confidence {top_score:.2f}")

                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Stopping audio event detection...")

if __name__ == "__main__":
    run_audio_detection()


import threading
import time

# Dummy placeholders for detection flags - replace these with your actual detection modules' outputs
video_alert_flag = False
audio_alert_flag = False

# Simulated functions to update alert flags (in practice, connect these to your actual detection outputs)
def video_detection_simulator():
    global video_alert_flag
    while True:
        # Fake detection logic for demo: toggle alert every 10 sec
        video_alert_flag = not video_alert_flag
        print(f"[Video] Suspicious activity detected? {video_alert_flag}")
        time.sleep(10)

def audio_detection_simulator():
    global audio_alert_flag
    while True:
        # Fake detection logic for demo: toggle alert every 7 sec
        audio_alert_flag = not audio_alert_flag
        print(f"[Audio] Suspicious sound detected? {audio_alert_flag}")
        time.sleep(7)

def fusion_engine():
    while True:
        # Basic fusion logic:
        # Alert only if both video and audio detect suspicious shit simultaneously
        if video_alert_flag and audio_alert_flag:
            print("[FUSION ALERT] Suspicious activity CONFIRMED by video + audio.")
            # Here you’d call your alert system (sms/email/push)
        else:
            print("[FUSION] No combined alert.")

        time.sleep(2)

if __name__ == "__main__":
    # Run simulators and fusion engine concurrently
    threading.Thread(target=video_detection_simulator, daemon=True).start()
    threading.Thread(target=audio_detection_simulator, daemon=True).start()

    fusion_engine()


from plyer import notification
import time

def send_popup_alert(title, message):
    notification.notify(
        title=title,
        message=message,
        app_name="Sentinel",
        timeout=5  # seconds the popup stays visible
    )

if __name__ == "__main__":
    # Example: call this whenever you detect suspicious activity
    send_popup_alert("Sentinel Alert", "Suspicious behavior detected: Running in restricted area!")
    time.sleep(6)
    send_popup_alert("Sentinel Alert", "Audio alert: Glass breaking detected!")
