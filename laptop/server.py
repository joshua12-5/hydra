import cv2
import time
import os
import numpy as np
from flask import Flask, Response, render_template, jsonify, request
from flask_socketio import SocketIO
from ultralytics import YOLO
from threading import Thread, Lock

os.makedirs("recordings", exist_ok=True)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

WIDTH, HEIGHT = 640, 480
JPEG_QUALITY = 60
RECORD_FPS = 15
YOLO_INTERVAL = 0.25

model = YOLO("yolo11n.pt")
CLASSES = [0]

raw_frame = None
detections = []
gps_logs = []

recording = False
video_writer = None
lock = Lock()

@socketio.on("video")
def video_rx(data):
    global raw_frame
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return
    frame = cv2.resize(frame, (WIDTH, HEIGHT))
    with lock:
        raw_frame = frame.copy()
        if recording and video_writer:
            video_writer.write(frame)

def yolo_loop():
    global detections
    while True:
        with lock:
            if raw_frame is None:
                continue
            frame = raw_frame.copy()
        r = model.predict(frame, classes=CLASSES, verbose=False)
        boxes = []
        if r and r[0].boxes:
            for b in r[0].boxes:
                x1,y1,x2,y2 = map(int,b.xyxy[0])
                boxes.append({"bbox":(x1,y1,x2,y2),"cls":int(b.cls[0])})
        with lock:
            detections = boxes
        time.sleep(YOLO_INTERVAL)

Thread(target=yolo_loop, daemon=True).start()

def mjpeg(frame):
    ok, jpg = cv2.imencode(".jpg", frame,
        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return (b"--frame\r\nContent-Type:image/jpeg\r\n\r\n" +
            jpg.tobytes() + b"\r\n") if ok else None

@app.route("/")
def index(): return render_template("index.html")

@app.route("/raw")
def raw():
    def gen():
        while True:
            with lock:
                f = raw_frame.copy() if raw_frame is not None else None
            if f is not None:
                yield mjpeg(f)
            time.sleep(0.01)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/cv")
def cv():
    def gen():
        while True:
            with lock:
                if raw_frame is None: continue
                f = raw_frame.copy()
                b = detections.copy()
            for d in b:
                x1,y1,x2,y2 = d["bbox"]
                cv2.rectangle(f,(x1,y1),(x2,y2),(0,255,255),2)
            yield mjpeg(f)
            time.sleep(0.01)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/logs")
def logs():
    with lock:
        snapshot = list(gps_logs[-100:])
    return jsonify(snapshot)

@socketio.on("gps")
def gps_rx(d):
    with lock:
        gps_logs.append(d)
        del gps_logs[:-500]

@socketio.on("manual_gps")
def gps_manual():
    with lock:
        last = gps_logs[-1] if gps_logs else None
    gps_logs_entry = {
        "time": time.time(),
        "lat": last["lat"] if last else 0,
        "lon": last["lon"] if last else 0,
        "manual": True
    }
    with lock:
        gps_logs.append(gps_logs_entry)

@socketio.on("laptop_audio")
def laptop_audio(d):
    socketio.emit("laptop_audio", d, skip_sid=request.sid)

@socketio.on("pi_audio")
def pi_audio(d):
    socketio.emit("pi_audio", d, skip_sid=request.sid)

@socketio.on("toggle_record")
def record(state):
    global recording, video_writer
    if state and raw_frame is not None:
        fn = f"recordings/{int(time.time())}.avi"
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        video_writer = cv2.VideoWriter(fn, fourcc, RECORD_FPS, (WIDTH, HEIGHT))
        recording = video_writer.isOpened()
    else:
        recording = False
        if video_writer:
            video_writer.release()
            video_writer = None

socketio.run(app, host="0.0.0.0", port=5000)