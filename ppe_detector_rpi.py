# ppe_detector_rpi.py
# Script ini berjalan di Raspberry Pi.
# Ia menangkap frame dari kamera, mengirimkannya ke AWS VPS untuk inferensi,
# dan menampilkan hasil deteksi di layar lokal.

import os
import sys
import argparse
import glob
import time
import requests # Untuk mengirim permintaan HTTP ke VPS
import base64   # Untuk mengkodekan/mendekodekan gambar
import json     # Untuk bekerja dengan data JSON

import cv2
import numpy as np

# --- Argument Parser ---
# Definisikan dan parse argumen input pengguna
parser = argparse.ArgumentParser()
# Argumen `--model` tidak lagi diperlukan karena inferensi dilakukan di VPS.
# Menggantinya dengan `--server_url`
parser.add_argument('--server_url', help='URL endpoint inferensi AWS VPS (contoh: "http://YOUR_AWS_VPS_IP/predict")',
                     required=True)
parser.add_argument('--source', help='Sumber gambar: file gambar ("test.jpg"), \
                     folder gambar ("test_dir"), file video ("testvid.mp4"), \
                     indeks kamera USB ("usb0"), atau URL stream RTSP ("rtsp://user:pass@ip:port/path"), \
                     atau Picamera ("picamera0")',
                     required=True)
parser.add_argument('--thresh', help='Ambang batas kepercayaan minimum untuk menampilkan objek yang terdeteksi (contoh: "0.4")',
                     default=0.5, type=float)
parser.add_argument('--resolution', help='Resolusi dalam WxH untuk menampilkan hasil inferensi (contoh: "640x480"), \
                     jika tidak, cocokkan resolusi sumber',
                     default=None)
parser.add_argument('--record', help='Rekam hasil dari video atau webcam dan simpan sebagai "demo1.avi". Harus menentukan argumen --resolution untuk merekam.',
                     action='store_true')

args = parser.parse_args()


# --- Parse Input Pengguna ---
server_url = args.server_url
img_source = args.source
min_thresh = args.thresh
user_res = args.resolution
record = args.record

# Set warna bounding box (menggunakan skema warna Tableu 10)
# Ini harus konsisten dengan pemetaan kelas di sisi server jika ingin warna spesifik per kelas
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106),
               (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184),
               (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
               (255, 0, 255), (192, 192, 192), (128, 128, 128), (128, 0, 0), (128, 128, 0)] # Perluas jika ada banyak kelas

# --- Tentukan Tipe Sumber ---
img_ext_list = ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.bmp','.BMP']
vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext in img_ext_list:
        source_type = 'image'
    elif ext in vid_ext_list:
        source_type = 'video'
    else:
        print(f'Ekstensi file {ext} tidak didukung.')
        sys.exit(0)
elif img_source.startswith('usb'):
    source_type = 'usb'
    usb_idx = int(img_source[3:])
elif img_source.startswith('picamera'):
    source_type = 'picamera'
    picam_idx = int(img_source[8:])
elif img_source.startswith('rtsp://'):
    source_type = 'rtsp'
else:
    print(f'Input {img_source} tidak valid. Silakan coba lagi. Tipe yang didukung: file, folder, video, usbX, picameraX, rtsp://URL')
    sys.exit(0)

# Parse resolusi tampilan yang ditentukan pengguna
resize = False
if user_res:
    resize = True
    try:
        resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])
    except ValueError:
        print(f"Error: Format resolusi tidak valid '{user_res}'. Harap gunakan WxH (misalnya, 1280x720).")
        sys.exit(0)

# Periksa apakah perekaman valid dan siapkan perekaman
if record:
    if source_type not in ['video', 'usb', 'rtsp', 'picamera']:
        print('Perekaman hanya berfungsi untuk sumber video, kamera, atau RTSP. Silakan coba lagi.')
        sys.exit(0)
    if not user_res:
        print('Harap tentukan resolusi untuk merekam video.')
        sys.exit(0)

    record_name = 'demo1.avi'
    record_fps = 30
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW,resH))


# --- Muat atau Inisialisasi Sumber Gambar ---
if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = []
    filelist = glob.glob(os.path.join(img_source, '*'))
    for file in filelist:
        _, file_ext = os.path.splitext(file)
        if file_ext in img_ext_list:
            imgs_list.append(file)
elif source_type in ['video', 'usb', 'rtsp']:
    if source_type == 'video': cap_arg = img_source
    elif source_type == 'usb': cap_arg = usb_idx
    elif source_type == 'rtsp': cap_arg = img_source

    cap = cv2.VideoCapture(cap_arg)

    if not cap.isOpened():
        print(f"ERROR: Tidak dapat membuka sumber video {cap_arg}. Harap periksa path/indeks/URL sumber.")
        sys.exit(0)

    if user_res:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resW)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resH)

elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    if not user_res:
        print("ERROR: Resolusi harus ditentukan untuk sumber Picamera2.")
        sys.exit(0)
    cap.configure(cap.create_video_configuration(main={"format": 'XRGB8888', "size": (resW, resH)}))
    cap.start()

# Inisialisasi variabel kontrol dan status
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0

# --- Mulai Loop Inferensi ---
while True:
    t_start = time.perf_counter()

    frame = None
    if source_type == 'image' or source_type == 'folder':
        if img_count >= len(imgs_list):
            print('Semua gambar telah diproses. Keluar dari program.')
            sys.exit(0)
        img_filename = imgs_list[img_count]
        frame = cv2.imread(img_filename)
        img_count = img_count + 1

    elif source_type in ['video', 'usb', 'rtsp']:
        ret, frame = cap.read()
        if not ret or frame is None:
            print('Tidak dapat membaca frame dari sumber. Ini menunjukkan sumber terputus atau tidak berfungsi. Keluar dari program.')
            break

    elif source_type == 'picamera':
        frame_bgra = cap.capture_array()
        frame = cv2.cvtColor(np.copy(frame_bgra), cv2.COLOR_BGRA2BGR)
        if (frame is None):
            print('Tidak dapat membaca frame dari Picamera. Ini menunjukkan kamera terputus atau tidak berfungsi. Keluar dari program.')
            break

    if frame is None:
        continue

    # Buat salinan frame untuk ditampilkan nanti (akan ditimpa dengan hasil)
    display_frame = frame.copy()

    # Encode frame menjadi byte JPEG untuk dikirim ke VPS
    # Kualitas JPEG dapat disesuaikan (0-100), default 95. Kualitas lebih rendah = ukuran lebih kecil, tetapi kualitas gambar lebih buruk.
    # Menggunakan kualitas 80 untuk kompromi.
    _, img_encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    image_bytes_b64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')

    # Kirim frame ke VPS untuk inferensi
    predictions = []
    try:
        # Mengirim data gambar sebagai JSON
        response = requests.post(server_url, json={'image': image_bytes_b64}, timeout=15) # Tambahkan timeout
        response.raise_for_status() # Akan memunculkan HTTPError untuk respons 4xx/5xx

        results_data = response.json()
        predictions = results_data.get('predictions', []) # Ambil daftar prediksi

    except requests.exceptions.Timeout:
        print("Permintaan server timed out. Pastikan VPS Anda berjalan dan dapat diakses.")
    except requests.exceptions.ConnectionError as e:
        print(f"Kesalahan koneksi ke server: {e}. Pastikan URL server benar dan server berjalan.")
    except requests.exceptions.HTTPError as e:
        print(f"Kesalahan HTTP dari server: {e.response.status_code} - {e.response.text}")
    except json.JSONDecodeError:
        print("Gagal mendekode respons JSON dari server.")
    except Exception as e:
        print(f"Terjadi kesalahan tak terduga selama komunikasi server: {e}")

    object_count = 0
    # Proses dan gambar deteksi yang diterima dari VPS
    for detection in predictions:
        xmin = detection.get('xmin', 0)
        ymin = detection.get('ymin', 0)
        xmax = detection.get('xmax', 0)
        ymax = detection.get('ymax', 0)
        classname = detection.get('class_name', 'unknown')
        conf = detection.get('confidence', 0.0)

        # Gambar box jika ambang batas kepercayaan cukup tinggi
        if conf > min_thresh:
            # Hash sederhana untuk classname untuk mendapatkan warna yang konsisten
            # Gunakan modulo dengan panjang daftar warna Anda
            color_idx = hash(classname) % len(bbox_colors)
            color = bbox_colors[color_idx]
            cv2.rectangle(display_frame, (xmin,ymin), (xmax,ymax), color, 2)

            label = f'{classname}: {int(conf*100)}%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(display_frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED)
            cv2.putText(display_frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            object_count = object_count + 1

    # Ubah ukuran frame untuk tampilan jika diperlukan
    if resize:
        display_frame = cv2.resize(display_frame, (resW, resH))

    # Hitung dan gambar framerate
    if source_type in ['video', 'usb', 'picamera', 'rtsp']:
        cv2.putText(display_frame, f'FPS: {avg_frame_rate:0.2f}', (10,20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)

    # Tampilkan hasil deteksi
    cv2.putText(display_frame, f'Jumlah objek: {object_count}', (10,40), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)
    cv2.imshow('Hasil deteksi YOLO', display_frame)
    if record: recorder.write(display_frame)

    # Jika inferensi pada gambar individual, tunggu penekanan tombol pengguna.
    # Jika tidak, tunggu 5ms sebelum pindah ke frame berikutnya.
    if source_type == 'image' or source_type == 'folder':
        key = cv2.waitKey()
    elif source_type in ['video', 'usb', 'picamera', 'rtsp']:
        key = cv2.waitKey(5)

    if key == ord('q') or key == ord('Q'): # Tekan 'q' untuk keluar
        break
    elif key == ord('s') or key == ord('S'): # Tekan 's' untuk menjeda inferensi
        cv2.waitKey()
    elif key == ord('p') or key == ord('P'): # Tekan 'p' untuk menyimpan gambar hasil pada frame ini
        cv2.imwrite('capture.png', display_frame)

    # Hitung FPS untuk frame ini
    t_stop = time.perf_counter()
    frame_rate_calc = float(1/(t_stop - t_start))

    # Tambahkan hasil FPS ke frame_rate_buffer (untuk menemukan FPS rata-rata selama beberapa frame)
    if len(frame_rate_buffer) >= fps_avg_len:
        temp = frame_rate_buffer.pop(0)
        frame_rate_buffer.append(frame_rate_calc)
    else:
        frame_rate_buffer.append(frame_rate_calc)

    # Hitung FPS rata-rata untuk frame-frame sebelumnya
    avg_frame_rate = np.mean(frame_rate_buffer)


# --- Bersihkan ---
print(f'Rata-rata FPS pipeline: {avg_frame_rate:.2f}')
if source_type in ['video', 'usb', 'rtsp']:
    if 'cap' in locals() and cap.isOpened(): # Pastikan cap objek ada dan terbuka sebelum release
        cap.release()
elif source_type == 'picamera':
    if 'cap' in locals():
        cap.stop()
if record: 
    if 'recorder' in locals() and recorder.isOpened(): # Pastikan recorder objek ada dan terbuka
        recorder.release()
cv2.destroyAllWindows()