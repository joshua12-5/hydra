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
        # FIX #3: was doing `continue` inside the lock when raw_frame was None,
        # causing a tight spin. Now exits the lock and sleeps before retrying.
        frame = None
        with lock:
            if raw_frame is not None:
                frame = raw_frame.copy()
        if frame is None:
            time.sleep(0.05)
            continue
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
                # FIX #5: guard against mjpeg() returning None on imencode failure
                chunk = mjpeg(f)
                if chunk is not None:
                    yield chunk
            time.sleep(0.01)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/cv")
def cv():
    def gen():
        while True:
            # FIX #4: was doing `continue` inside lock when raw_frame was None
            # — tight spin. Now releases lock and sleeps.
            with lock:
                if raw_frame is None:
                    f, b = None, None
                else:
                    f = raw_frame.copy()
                    b = detections.copy()
            if f is None:
                time.sleep(0.05)
                continue
            for d in b:
                x1,y1,x2,y2 = d["bbox"]
                cv2.rectangle(f,(x1,y1),(x2,y2),(0,255,255),2)
            # FIX #5: guard against mjpeg() returning None
            chunk = mjpeg(f)
            if chunk is not None:
                yield chunk
            time.sleep(0.01)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/logs")
def logs():
    # FIX #7: return all stored entries (up to 500), not just the last 100
    with lock:
        snapshot = list(gps_logs)
    return jsonify(snapshot)

@socketio.on("gps")
def gps_rx(d):
    with lock:
        gps_logs.append(d)
        del gps_logs[:-500]

@socketio.on("manual_gps")
def gps_manual():
    # FIX #6: was two separate lock acquisitions with a gap (TOCTOU).
    # Now the read and append happen in a single atomic lock acquisition.
    with lock:
        last = gps_logs[-1] if gps_logs else None
        gps_logs.append({
            "time": time.time(),
            "lat": last["lat"] if last else 0,
            "lon": last["lon"] if last else 0,
            "manual": True
        })

@socketio.on("laptop_audio")
def laptop_audio(d):
    socketio.emit("laptop_audio", d, skip_sid=request.sid)

@socketio.on("pi_audio")
def pi_audio(d):
    socketio.emit("pi_audio", d, skip_sid=request.sid)

# FIX #12 (server side): relay PTT state to Pi so it mutes its mic during TX
@socketio.on("ptt_start")
def ptt_start():
    socketio.emit("pi_mute", True, skip_sid=request.sid)

@socketio.on("ptt_stop")
def ptt_stop():
    socketio.emit("pi_mute", False, skip_sid=request.sid)

@socketio.on("toggle_record")
def record(state):
    global recording, video_writer
    # FIX #1 & #2: recording, video_writer, and raw_frame all accessed under
    # lock so video_rx can't write to a half-released writer during teardown.
    if state:
        with lock:
            frame_ready = raw_frame is not None
        if frame_ready:
            fn = f"recordings/{int(time.time())}.avi"
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            vw = cv2.VideoWriter(fn, fourcc, RECORD_FPS, (WIDTH, HEIGHT))
            with lock:
                if vw.isOpened():
                    video_writer = vw
                    recording = True
                else:
                    recording = False
                    video_writer = None
        else:
            recording = False
        # FIX #8: push the real server-side outcome back to the client
        socketio.emit("record_state", recording)
    else:
        with lock:
            recording = False
            vw = video_writer
            video_writer = None
        if vw:
            vw.release()
        socketio.emit("record_state", False)

socketio.run(app, host="0.0.0.0", port=5000)
