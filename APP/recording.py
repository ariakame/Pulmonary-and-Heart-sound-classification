import serial
import wave
import time
import numpy as np
from scipy.signal import butter, lfilter, medfilt

# === Settings ===
PORT = "/dev/ttyACM0"       # Adjust to match your port
BAUD = 115200
DURATION = 5                # Recording duration in seconds
FILENAME = "cleaned_audio.wav"

# === Filter Functions ===
def butter_bandpass(lowcut, highcut, fs, order=6):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band')

def apply_bandpass_filter(data, lowcut=100, highcut=1500, fs=8000):
    b, a = butter_bandpass(lowcut, highcut, fs)
    return lfilter(b, a, data)

def trim_silence(audio, threshold=5):
    idx = np.where(np.abs(audio) > threshold)[0]
    if len(idx) == 0:
        return audio
    return audio[idx[0]:idx[-1]]

# === Serial Recording ===
print("[*] Connecting to Arduino...")
ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)  # Wait for Arduino to reset

print(f"[*] Recording {DURATION} seconds of audio...")
audio_bytes = bytearray()
start_time = time.time()

while time.time() - start_time < DURATION:
    if ser.in_waiting:
        audio_bytes += ser.read(ser.in_waiting)

ser.close()

# === Sample Rate Estimation ===
total_samples = len(audio_bytes)
actual_duration = time.time() - start_time
estimated_sample_rate = int(total_samples / actual_duration)
print(f"[+] Estimated Sample Rate: {estimated_sample_rate} Hz")
print(f"[+] Total Bytes Received: {total_samples}")

# === Preprocessing ===
raw = np.frombuffer(audio_bytes, dtype=np.uint8)
signal = raw.astype(np.int16) - 128                     # Convert to signed
signal = signal - int(np.mean(signal))                  # Remove DC offset safely

filtered = apply_bandpass_filter(signal, 100, 1500, estimated_sample_rate)
smoothed = medfilt(filtered, kernel_size=3)
trimmed = trim_silence(smoothed, threshold=4)

# === Normalize and Convert to 8-bit Unsigned
if np.max(np.abs(trimmed)) != 0:
    normalized = np.clip((trimmed / np.max(np.abs(trimmed))) * 127, -128, 127)
else:
    normalized = trimmed  # Avoid division by zero for silence

final = (normalized + 128).clip(0, 255).astype(np.uint8)

# === Save as WAV
with wave.open(FILENAME, 'wb') as wf:
    wf.setnchannels(1)                  # Mono
    wf.setsampwidth(1)                  # 8-bit
    wf.setframerate(estimated_sample_rate)
    wf.writeframes(final.tobytes())

print(f"[✓] Cleaned audio saved to '{FILENAME}'")

