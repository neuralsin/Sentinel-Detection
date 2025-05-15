import cv2
import threading
import time
import numpy as np
import queue
import sounddevice as sd
import tensorflow as tf
import tensorflow_hub as hub
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import mediapipe as mp
from plyer import notification

# === GLOBAL FLAGS ===
video_alert_flag = False
audio_alert_flag = False

# === YOLO + DeepSORT Video Detection & Tracking ===
def run_detection_tracking(video_source=0):
    global video_alert_flag
    model = YOLO('yolov8n.pt')  # YOLOv8 nano - lightweight af
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
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))

        tracks = tracker.update_tracks(detections, frame=frame)
        suspicious_found = False
        for track in tracks:
            if not track.is_confirmed():
                continue
            x, y, w, h = map(int, track.to_ltwh())
            track_id = track.track_id
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{track_id}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            # Example suspicious logic: if detected class is person (cls==0), mark suspicious
            if cls == 0 and conf > 0.5:
                suspicious_found = True

        video_alert_flag = suspicious_found

        cv2.imshow("Sentinel - Detection & Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# === Mediapipe Pose Behavior Detection ===
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

def run_behavior_detection(video_source=0):
    cap = cv2.VideoCapture(video_source)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )
            # This can be expanded with anomaly detection based on landmarks sequences

        cv2.putText(frame, "Behavior Detection Running", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.imshow("Sentinel - Behavior Anomaly Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# === YAMNet Audio Event Detection ===
yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)

class_map_path = tf.keras.utils.get_file('yamnet_class_map.csv',
                                         'https://raw.githubusercontent.com/auditory/yamnet/master/yamnet_class_map.csv')
class_names = []
with open(class_map_path) as f:
    for line in f.readlines()[1:]:
        class_names.append(line.strip().split(',')[2])

SAMPLE_RATE = 16000
BUFFER_DURATION = 1.0
audio_queue = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"Audio error: {status}")
    audio_queue.put(indata.copy())

def run_audio_detection():
    global audio_alert_flag
    print("Starting audio event detection... Press Ctrl+C to stop")

    with sd.InputStream(channels=1, samplerate=SAMPLE_RATE, callback=audio_callback):
        try:
            while True:
                if audio_queue.qsize() > 0:
                    audio_chunk = audio_queue.get()
                    audio_chunk = np.squeeze(audio_chunk)

                    scores, embeddings, spectrogram = yamnet_model(audio_chunk)
                    scores_np = scores.numpy()
                    mean_scores = np.mean(scores_np, axis=0)
                    top_index = np.argmax(mean_scores)
                    top_class = class_names[top_index]
                    top_score = mean_scores[top_index]

                    alert_classes = ['Gunshot, gunfire',
                                     'Siren',
                                     'Glass breaking, smashing',
                                     'Scream']

                    if top_score > 0.3 and top_class in alert_classes:
                        print(f"[ALERT] Audio detected: {top_class} ({top_score:.2f})")
                        audio_alert_flag = True
                    else:
                        audio_alert_flag = False

                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Audio detection stopped.")

# === Fusion Logic & Popup Alert ===
def send_popup_alert(title, message):
    notification.notify(
        title=title,
        message=message,
        app_name="Sentinel",
        timeout=5
    )

def fusion_engine():
    already_alerted = False
    while True:
        if video_alert_flag and audio_alert_flag:
            if not already_alerted:
                print("[FUSION ALERT] Suspicious activity CONFIRMED by video + audio.")
                send_popup_alert("Sentinel Alert", "Suspicious behavior detected by video + audio!")
                already_alerted = True
        else:
            if already_alerted:
                print("[FUSION] Alert cleared.")
            already_alerted = False

        time.sleep(2)

# === MAIN THREADS SETUP ===
if __name__ == "__main__":
    # Run video detection and tracking in its own thread
    video_thread = threading.Thread(target=run_detection_tracking, args=(0,), daemon=True)
    video_thread.start()

    # Run audio detection in its own thread
    audio_thread = threading.Thread(target=run_audio_detection, daemon=True)
    audio_thread.start()

    # Run fusion alert engine in main thread
    fusion_engine()
