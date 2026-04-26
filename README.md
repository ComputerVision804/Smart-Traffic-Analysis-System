# Smart-Traffic-Analysis-System
🚦 Smart Traffic Analysis System

An end-to-end Computer Vision pipeline for real-time traffic monitoring that combines detection, tracking, and analytics into a single system.

📌 Features
🚗 Vehicle Detection using YOLOv8
🔁 Multi-Object Tracking with ByteTrack
📏 Real-World Speed Estimation using homography mapping
🔄 Wrong-Way Detection based on trajectory direction
🚨 Red-Light Violation Detection using stop-line crossing
🎥 Annotated Video Output with IDs, speed, and alerts
⚡ Performance Monitoring (FPS)
🧠 System Overview

This system processes a video input and performs:

Object detection (vehicles)
Object tracking with persistent IDs
Pixel → real-world coordinate transformation
Speed estimation in km/h
Behavior analysis (direction + violations)
🛠️ Tech Stack
Python
OpenCV
Ultralytics YOLOv8
ByteTrack
NumPy
📂 Project Structure
traffic-analysis/
│── main.py
│── input.mp4
│── yolov8n.pt
│── bytetrack.yaml
│── README.md
⚙️ Installation
pip install ultralytics opencv-python numpy
▶️ Usage
Place your video file:
input.mp4
Run the script:
python main.py
Press Q to exit.
📐 Homography Calibration (Important)

For accurate speed estimation, you must calibrate:

src_pts = np.float32([...])  # pixel points
dst_pts = np.float32([...])  # real-world meters

👉 Tips:

Measure lane width or road distance in meters
Match 4 points from image → real-world rectangle
🚨 Violation Detection Logic
🔴 Red Light Violation
Vehicle crosses stop line while signal = RED
🔄 Wrong Way Detection
Vehicle direction ≠ expected traffic flow
⚠️ Limitations
Speed depends heavily on homography accuracy
Traffic light is simulated (not detected yet)
Works best with stable, fixed camera
🚀 Future Improvements
Real traffic light detection (YOLO model)
Lane detection using deep learning
Live dashboard (Flask / Streamlit)
Database logging (violations + stats)
Multi-camera system

https://github.com/user-attachments/assets/72a24807-cb94-40e7-98f8-50802a111e78
👤 Author

Awais Shah
AI / ML | Computer Vision

⭐ If you like this project

Give it a ⭐ on GitHub and feel free to contribute!
