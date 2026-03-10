# 🤚 Hand Tracking dengan Gesture Control

Aplikasi Python untuk mengontrol komputer menggunakan gestur tangan melalui kamera.

## ✨ Fitur

| Gesture | Aksi |
|---------|------|
| 🖱️ Gerakkan telunjuk | Menggerakkan kursor mouse |
| 👆 Jempol + Telunjuk (pinch) | Klik kiri (Left Click) |
| ✌️ Jempol + Jari tengah (pinch) | Klik kanan (Right Click) |
| ⬆️ 2 Jari (atas/bawah) | Scroll halaman |

## 📋 Prasyarat

- Python 3.8+
- Webcam/camera
- macOS (sudah tested)

## 🔧 Package yang Diperlukan

```bash
pip3 install mediapipe opencv-python pyautogui
```

> **Note:** MediaPipe dan OpenCV sudah terinstall di sistem ini.

## 🚀 Cara Menjalankan

```bash
cd hand_tracking
python3 hand_tracking.py
```

Tekan **Q** untuk keluar dari aplikasi.

## ⚙️ Konfigurasi

Edit class `Config` di dalam file `hand_tracking.py` untuk menyesuaikan:

```python
class Config:
    SMOOTHING = 5          # Semakin besar = gerakan lebih halus
    DEAD_ZONE = 5          # Zona diam untuk mengurangi getaran
    PINCH_THRESHOLD = 35   # Sensitivitas pinch (klik)
    SCROLL_SPEED = 50      # Kecepatan scroll
```

## 📝 Tips Penggunaan

1. **Posisi tangan**: Pastikan tangan terlihat jelas oleh kamera
2. **Pencahayaan**: Gunakan pencahayaan yang cukup untuk deteksi lebih baik
3. **Jarak**: Jarak ideal 30-60 cm dari kamera
4. **Background**: Background yang kontras dengan kulit lebih baik

## 🛡️ Safety

- **FAILSAFE**: PyAutoGUI failsafe diaktifkan - gerakkan mouse ke pojok kiri atas untuk emergency stop
- Tekan `Q` di window kamera untuk keluar dengan aman

## 📁 Struktur File

```
hand_tracking/
├── hand_tracking.py    # Kode utama
└── README.md          # Dokumentasi ini
```

## 🐛 Troubleshooting

| Masalah | Solusi |
|---------|--------|
| Kamera tidak terbuka | Cek permission kamera di System Preferences > Security & Privacy |
| Mouse tidak bergerak | Pastikan hanya 1 tangan yang terdeteksi |
| Klik tidak responsif | Sesuaikan `PINCH_THRESHOLD` di konfigurasi |
| Gerakan patah-patah | Naikkan nilai `SMOOTHING` |

---

Dibuat dengan ❤️ menggunakan Python + MediaPipe + OpenCV
