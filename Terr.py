import cv2
import numpy as np
import time
from collections import defaultdict, deque
from ultralytics import YOLO

# =========================
# CONFIG
# =========================
VIDEO_PATH = r"C:\Users\Qc\Desktop\Traffic Analysis System\input.mp4"

CONF_THRESHOLD = 0.4
VEHICLE_CLASSES = [2, 3, 5, 7]

EXPECTED_DIRECTION = "DOWN"
STOP_LINE_Y = 450

RED_TIME = 10
GREEN_TIME = 10

# =========================
# HOMOGRAPHY (CALIBRATE!)
# =========================
src_pts = np.float32([
    [200, 500],
    [1000, 500],
    [300, 800],
    [900, 800]
])

dst_pts = np.float32([
    [0, 0],
    [12, 0],
    [0, 40],
    [12, 40]
])

H, _ = cv2.findHomography(src_pts, dst_pts)

# =========================
# FUNCTIONS
# =========================
def to_world(pt):
    p = np.array([pt[0], pt[1], 1]).reshape(3, 1)
    w = H @ p
    w /= w[2]
    return w[0][0], w[1][0]


def real_speed(prev, curr, fps):
    if prev is None:
        return 0
    p1 = to_world(prev)
    p2 = to_world(curr)
    dist = np.linalg.norm(np.array(p1) - np.array(p2))
    return dist * fps * 3.6


def get_direction(hist):
    if len(hist) < 2:
        return "NONE"
    dx = hist[-1][0] - hist[0][0]
    dy = hist[-1][1] - hist[0][1]
    return "DOWN" if dy > 0 else "UP"


def wrong_way(direction):
    return direction != EXPECTED_DIRECTION


def traffic_light_state(start):
    t = int(time.time() - start)
    cycle = t % (RED_TIME + GREEN_TIME)
    return "RED" if cycle < RED_TIME else "GREEN"


def red_light_violation(curr, prev, light):
    if light != "RED" or prev is None:
        return False
    return prev[1] < STOP_LINE_Y and curr[1] >= STOP_LINE_Y


# =========================
# INIT
# =========================
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 25

track_history = defaultdict(lambda: deque(maxlen=10))

start_time = time.time()
frame_id = 0

# =========================
# MAIN LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        print("Video ended.")
        break

    frame_id += 1
    annotated = frame.copy()

    # Traffic light simulation
    light = traffic_light_state(start_time)

    light_color = (0, 0, 255) if light == "RED" else (0, 255, 0)
    cv2.circle(annotated, (50, 50), 20, light_color, -1)
    cv2.putText(annotated, light, (80, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, light_color, 2)

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=CONF_THRESHOLD,
        classes=VEHICLE_CLASSES,
        verbose=False
    )

    if results[0].boxes.id is not None:

        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int)

        for box, tid in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            prev = track_history[tid][-1] if track_history[tid] else None
            track_history[tid].append((cx, cy))

            # Real speed
            speed = real_speed(prev, (cx, cy), fps)

            # Direction
            direction = get_direction(track_history[tid])
            is_wrong = wrong_way(direction)

            # Red light violation
            is_red = red_light_violation((cx, cy), prev, light)

            # Color logic
            color = (0, 255, 0)

            if is_wrong:
                color = (255, 0, 255)

            if is_red:
                color = (0, 255, 255)

            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label = f"ID:{tid} {int(speed)}km/h {direction}"

            if is_wrong:
                label += " WRONG WAY"

            if is_red:
                label += " RED VIOLATION"

            cv2.putText(annotated, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Stop line
    cv2.line(annotated, (0, STOP_LINE_Y),
             (annotated.shape[1], STOP_LINE_Y), (0, 0, 255), 2)

    # FPS
    fps_live = frame_id / (time.time() - start_time)
    cv2.putText(annotated, f"FPS:{fps_live:.2f}",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2)

    cv2.imshow("Traffic System", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================
# CLEANUP
# =========================
cap.release()
cv2.destroyAllWindows()

print("Processing Complete ✅")