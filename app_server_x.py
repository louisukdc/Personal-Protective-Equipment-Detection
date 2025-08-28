# app_server.py

from flask import Flask, request, jsonify
import base64
import numpy as np
import cv2
from ultralytics import YOLO
import time

app = Flask(__name__)

# --- Konfigurasi Model YOLO ---
MODEL_PATH = "bestnew.pt"

try:
    model = YOLO(MODEL_PATH, task='detect')
    labels = model.names
    print(f"[SERVER STARTUP] Model YOLO '{MODEL_PATH}' berhasil dimuat dengan {len(labels)} kelas.")
except Exception as e:
    print(f"[SERVER STARTUP ERROR] Gagal memuat model YOLO: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    if not request.is_json or 'image' not in request.json:
        return jsonify({"error": "Permintaan harus dalam format JSON dan menyertakan kunci 'image'."}), 400

    image_b64 = request.json['image']
    
    try:
        image_bytes = base64.b64decode(image_b64)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Gagal mendekode gambar. Pastikan format gambar valid (mis. JPEG, PNG)."}), 400

        inference_start_time = time.perf_counter()
        results = model(frame, verbose=False)
        inference_end_time = time.perf_counter()
        print(f"[INFERENCE] Inferensi selesai dalam {(inference_end_time - inference_start_time)*1000:.2f} ms")

        detections = results[0].boxes
        predictions = []
        for i in range(len(detections)):
            xyxy_tensor = detections[i].xyxy.cpu()
            xyxy = xyxy_tensor.numpy().squeeze()    
            
            # --- Perbaikan di sini ---
            xmin, ymin, xmax, ymax = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]) 

            class_id = int(detections[i].cls.item())
            class_name = labels[class_id]
            
            # --- Perbaikan di sini ---
            confidence = float(detections[i].conf.item()) 

            predictions.append({
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "class_name": class_name,
                "confidence": confidence
            })

        return jsonify({"predictions": predictions}), 200

    except base64.binascii.Error:
        print(f"[ERROR] String base64 tidak valid.")
        return jsonify({"error": "Format string base64 gambar tidak valid."}), 400
    except Exception as e:
        print(f"[ERROR] Terjadi kesalahan saat melakukan prediksi: {e}")
        return jsonify({"error": f"Kesalahan internal server: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)