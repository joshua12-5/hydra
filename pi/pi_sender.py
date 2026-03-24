import cv2
import time
import socketio
import serial
import pynmea2
import sounddevice as sd
import numpy as np
from threading import Thread
from collections import deque

# ================= CONFIG =================

SERVER_IP = "172.20.10.2"
SERVER_PORT = 5000

VIDEO_SIZE = (320, 240)
JPEG_QUALITY = 60

NET_RATE = 16000      
AUDIO_BLOCK = 2048

MIC_DEVICE = 1
SPEAKER_DEVICE = None  # None = default ALSA device

sio = socketio.Client(reconnection=True)

# ================= GPS =================

def init_gps():
    try:
        return serial.Serial("/dev/serial0", 9600, timeout=1)
    except:
        return None

gps = init_gps()

def gps_loop():
    if not gps:
        print("GPS not found")
        return
    while True:
        try:
            line = gps.readline().decode(errors="ignore")
            if line.startswith(("$GPGGA", "$GPRMC")):
                msg = pynmea2.parse(line)
                if msg.latitude and msg.longitude and sio.connected:
                    sio.emit("gps", {
                        "time": time.time(),
                        "lat": msg.latitude,
                        "lon": msg.longitude,
                        "manual": False
                    })
        except:
            pass
        time.sleep(0.1)

# ================= VIDEO =================

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

def video_loop():
    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        # 🔄 Flip camera upside down (180° rotation)
        frame = cv2.rotate(frame, cv2.ROTATE_180)

        frame = cv2.resize(frame, VIDEO_SIZE)

        ok, jpg = cv2.imencode(".jpg", frame,
                               [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

        if ok and sio.connected:
            sio.emit("video", jpg.tobytes())

        time.sleep(0.02)

# ================= AUDIO SEND (MIC) =================

def mic_loop():
    def callback(indata, frames, time_info, status):
        if sio.connected:
            sio.emit("pi_audio", indata.copy().tobytes())

    with sd.InputStream(device=MIC_DEVICE,
                        channels=1,
                        samplerate=NET_RATE,
                        blocksize=AUDIO_BLOCK,
                        dtype='int16',
                        callback=callback):
        while True:
            time.sleep(1)

# ================= AUDIO RECEIVE (BLOCKING SPEAKER) =================

audio_queue = deque(maxlen=50)

def speaker_loop():
    print("Speaker thread started")
    while True:
        if audio_queue:
            mono = audio_queue.popleft()
            stereo = np.column_stack((mono, mono))

            sd.play(stereo, samplerate=NET_RATE)
            sd.wait()
        else:
            time.sleep(0.01)

@sio.on("laptop_audio")
def receive_laptop_audio(data):
    audio = np.frombuffer(data, dtype=np.int16)
    audio_queue.append(audio)
    print("AUDIO RX:", len(audio))

# ================= CONNECT =================

def connect_loop():
    while True:
        try:
            if not sio.connected:
                sio.connect(f"http://{SERVER_IP}:{SERVER_PORT}")
                print("Connected to server")
        except:
            pass
        time.sleep(3)

# ================= START =================

Thread(target=connect_loop, daemon=True).start()
Thread(target=video_loop, daemon=True).start()
Thread(target=gps_loop, daemon=True).start()
Thread(target=mic_loop, daemon=True).start()
Thread(target=speaker_loop, daemon=True).start()

while True:
    time.sleep(5)