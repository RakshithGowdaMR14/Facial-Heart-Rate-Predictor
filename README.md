# 💓 Face Recognition Based Heartbeat Prediction

A Python-based AI system that estimates a person's heart rate (in BPM) using a standard webcam. It uses facial recognition and remote photoplethysmography (rPPG) to detect pulse from subtle color variations in the skin, without any physical contact.

---

## 📌 Features

- Real-time face detection using **MTCNN**
- Region of interest (ROI) extraction from the **forehead or cheeks**
- Heartbeat prediction using **RGB signal analysis**
- Signal preprocessing using **bandpass filtering**
- Frequency-based heart rate estimation using **FFT**
- Live heart rate display on webcam feed

---

## 🧠 How It Works

1. **Face Detection**: Detects and tracks the face using MTCNN.
2. **ROI Selection**: Extracts forehead or cheek region using facial landmarks.
3. **RGB Signal Extraction**: Captures average R, G, B values from ROI over time.
4. **Signal Preprocessing**: Cleans the signal using bandpass filtering and normalization.
5. **Heart Rate Estimation**: Uses FFT to find the dominant frequency and convert it to BPM.
6. **Display**: Shows real-time heart rate on screen.

---
# SNAPSHOTS

![Screenshot 2025-04-30 073110](https://github.com/user-attachments/assets/5be9bc01-26e6-42ec-9c20-854f95b34f12)
