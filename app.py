from flask import Flask, render_template, jsonify
import cv2
import numpy as np
import time
from mtcnn import MTCNN
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import os

app = Flask(__name__)



def extract_roi(frame, keypoints):
    left_eye = keypoints['left_eye']
    right_eye = keypoints['right_eye']

    x = int((left_eye[0] + right_eye[0]) / 2) - 30
    y = int((left_eye[1] + right_eye[1]) / 2) - 60
    w, h = 60, 30

    h_frame, w_frame, _ = frame.shape
    x = max(0, min(x, w_frame - w))
    y = max(0, min(y, h_frame - h))

    roi = frame[y:y+h, x:x+w]
    return roi if roi.size != 0 else None

def bandpass_filter(signal, lowcut=0.75, highcut=4, fs=30, order=3):
    nyquist = 0.5 * fs
    low, high = lowcut / nyquist, highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def estimate_bpm(signal, fs=30):
    N = len(signal)
    freqs = fftfreq(N, d=1/fs)
    fft_vals = np.abs(fft(signal))
    idx = np.where((freqs > 0.75) & (freqs < 4))
    freqs = freqs[idx]
    fft_vals = fft_vals[idx]

    peak_freq = freqs[np.argmax(fft_vals)]
    bpm = peak_freq * 60
    return bpm

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start')
def start_measurement():
    detector = MTCNN()
    cap = cv2.VideoCapture(0)
    g_values = []

    start_time = time.time()
    duration = 40

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = detector.detect_faces(frame)
        if results:
            result = results[0]
            keypoints = result['keypoints']
            roi = extract_roi(frame, keypoints)

            # Draw bounding box and keypoint
            x, y, w, h = result['box']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            forehead = keypoints.get('forehead')
            if forehead:
                cv2.circle(frame, tuple(map(int, forehead)), 5, (255, 0, 0), -1)

            if roi is not None:
                avg_color = np.mean(np.mean(roi, axis=0), axis=0)
                g_values.append(avg_color[1])

        # Show frame
        cv2.imshow('Face Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if time.time() - start_time > duration:
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(g_values) > 30:
        try:
            print(f"Collected Green Values: {g_values}")
            filtered_g = bandpass_filter(g_values)
            bpm = estimate_bpm(filtered_g)
            print(f"Estimated BPM: {bpm:.2f}")

            # Save pulse plot
            plt.figure(figsize=(10, 4))
            plt.plot(filtered_g, color='green')
            plt.title('Filtered Pulse Signal')
            plt.xlabel('Frame')
            plt.ylabel('Green Intensity')
            plt.tight_layout()

            if not os.path.exists('static'):
                os.makedirs('static')

            plot_path = os.path.join('static', 'pulse_plot.png')
            plt.savefig(plot_path)
            plt.close()

            return jsonify({'status': 'success', 'bpm': f"{bpm:.2f}", 'g_values': g_values})
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e), 'g_values': g_values})
    else:
        return jsonify({'status': 'error', 'message': 'Not enough data to estimate BPM.', 'g_values': g_values})

if __name__ == '__main__':
    app.run(debug=True)
