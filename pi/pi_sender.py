import cv2
import time
import queue
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
    except Exception:
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
                # FIX #16: previous check `if msg.latitude and msg.longitude`
                # silently drops valid fixes at exactly 0.0 degrees latitude/longitude.
                if msg.latitude is not None and msg.longitude is not None and sio.connected:
                    sio.emit("gps", {
                        "time": time.time(),
                        "lat": msg.latitude,
                        "lon": msg.longitude,
                        "manual": False
                    })
        except Exception:
            # FIX #15: was bare `except:` which swallows KeyboardInterrupt/SystemExit
            pass
        time.sleep(0.1)

# ================= VIDEO =================

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

def video_loop():
    while True:
        ok, frame = cap.read()
        if not ok:
            # FIX #13: was spinning with no sleep on camera failure — pegs CPU
            time.sleep(0.1)
            continue

        frame = cv2.rotate(frame, cv2.ROTATE_180)
        frame = cv2.resize(frame, VIDEO_SIZE)

        ok2, jpg = cv2.imencode(".jpg", frame,
                               [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if ok2 and sio.connected:
            sio.emit("video", jpg.tobytes())

        time.sleep(0.02)

# ================= AUDIO SEND (MIC) =================

pi_muted = False  # FIX #12: controlled by server ptt_start/ptt_stop events

# FIX #14: mic callback was calling sio.emit() directly — unsafe from a
# real-time audio thread (involves locking + socket I/O). Instead, put raw
# bytes into a bounded queue and let a dedicated sender thread emit them.
mic_send_queue = queue.Queue(maxsize=10)

def mic_callback(indata, frames, time_info, status):
    if sio.connected and not pi_muted:
        try:
            mic_send_queue.put_nowait(indata.copy().tobytes())
        except queue.Full:
            pass  # drop frame rather than block the audio thread

def mic_sender_loop():
    while True:
        try:
            data = mic_send_queue.get(timeout=1)
            if sio.connected:
                sio.emit("pi_audio", data)
        except queue.Empty:
            pass
        except Exception:
            pass

def mic_loop():
    with sd.InputStream(device=MIC_DEVICE,
                        channels=1,
                        samplerate=NET_RATE,
                        blocksize=AUDIO_BLOCK,
                        dtype='int16',
                        callback=mic_callback):
        while True:
            time.sleep(1)

# ================= AUDIO RECEIVE (SPEAKER) =================

# FIX #9 & #10: query device channel count so we don't force stereo on a
# mono-only output, and pass SPEAKER_DEVICE to sd.OutputStream.
def get_speaker_channels():
    try:
        info = sd.query_devices(SPEAKER_DEVICE)
        return min(max(int(info['max_output_channels']), 1), 2)
    except Exception:
        return 1

SPEAKER_CHANNELS = get_speaker_channels()

audio_queue = deque(maxlen=50)

# FIX #11: replaced sd.play()+sd.wait() blocking pattern (produces audible
# gaps between 128ms chunks) with a continuous OutputStream callback that
# drains the queue frame-by-frame with no gaps.
_spk_buf = {'data': np.zeros(0, dtype=np.int16)}

def speaker_callback(outdata, frames, time_info, status):
    buf = _spk_buf['data']
    # Top up internal buffer from queue
    while len(buf) < frames and audio_queue:
        buf = np.concatenate([buf, audio_queue.popleft()])
    take = min(frames, len(buf))
    chunk = buf[:take]
    _spk_buf['data'] = buf[take:]
    # Write samples, pad with silence if underrun
    for ch in range(SPEAKER_CHANNELS):
        outdata[:take, ch] = chunk
        if take < frames:
            outdata[take:, ch] = 0

def speaker_loop():
    print(f"Speaker thread started ({SPEAKER_CHANNELS}ch)")
    with sd.OutputStream(device=SPEAKER_DEVICE,
                         channels=SPEAKER_CHANNELS,
                         samplerate=NET_RATE,
                         dtype='int16',
                         blocksize=AUDIO_BLOCK,
                         callback=speaker_callback):
        while True:
            time.sleep(1)

@sio.on("laptop_audio")
def receive_laptop_audio(data):
    audio = np.frombuffer(data, dtype=np.int16)
    audio_queue.append(audio)

# FIX #12: listen for PTT mute commands from server
@sio.on("pi_mute")
def on_pi_mute(state):
    global pi_muted
    pi_muted = bool(state)
    print("Pi mic muted:", pi_muted)

# ================= CONNECT =================

def connect_loop():
    while True:
        try:
            if not sio.connected:
                sio.connect(f"http://{SERVER_IP}:{SERVER_PORT}")
                print("Connected to server")
        except Exception:
            # FIX #15: was bare `except:` — now Exception only, lets
            # KeyboardInterrupt/SystemExit propagate for clean shutdown
            pass
        time.sleep(3)

# ================= START =================

Thread(target=connect_loop,  daemon=True).start()
Thread(target=video_loop,    daemon=True).start()
Thread(target=gps_loop,      daemon=True).start()
Thread(target=mic_loop,      daemon=True).start()
Thread(target=mic_sender_loop, daemon=True).start()
Thread(target=speaker_loop,  daemon=True).start()

try:
    while True:
        time.sleep(5)
except KeyboardInterrupt:
    print("Shutting down")
    sio.disconnect()
