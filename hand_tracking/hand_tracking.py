"""
Hand Tracking with Gesture Control
===================================
Aplikasi hand tracking untuk kontrol komputer dengan gestur tangan.

Fitur:
- Virtual Mouse: Gerakkan kursor dengan jari telunjuk
- Click: Pinch (jempol + telunjuk) untuk klik kiri
- Right Click: Pinch (jempol + jari tengah) untuk klik kanan
- Scroll: Gestur 2 jari (telunjuk + tengah) geser atas/bawah

Cara pakai:
1. Pastikan kamera menyala
2. Tunjukkan tangan ke kamera
3. Gerakkan telunjuk untuk menggerakkan mouse
4. Satukan jempol & telunjuk untuk klik
"""

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import math
from collections import deque

# Konfigurasi
class Config:
    # Resolusi default window (bisa diubah dengan fullscreen)
    CAM_WIDTH = 960
    CAM_HEIGHT = 540
    
    # Smoothing untuk gerakan mouse (semakin besar semakin halus)
    SMOOTHING = 6
    
    # Dead zone untuk mengurangi getaran (pixel)
    DEAD_ZONE = 6
    
    # Threshold untuk pinch detection (semakin BESAR = semakin ketat)
    # Range: 30-60 (30=sangat sensitif, 60=harus benar-benar menyentuh)
    PINCH_THRESHOLD = 45
    
    # Threshold untuk release klik (harus lebih besar dari PINCH_THRESHOLD)
    PINCH_RELEASE_THRESHOLD = 60
    
    # Area operasi mouse (untuk menghindari tepi layar)
    SCREEN_MARGIN = 50  # Pixel dari tepi
    
    # Kecepatan scroll (semakin KECIL = semakin lambat)
    SCROLL_SPEED = 15
    
    # Delay antar scroll (frame) - semakin BESAR = semakin jarang scroll
    SCROLL_DELAY = 8
    
    # Smoothing scroll (semakin besar = semakin halus tapi lebih lambat respon)
    SCROLL_SMOOTHING = 10
    
    # Threshold zona scroll (semakin kecil = zona semakin sempit)
    SCROLL_ZONE_ENTER = 0.35  # 35% dari atas/bawah
    SCROLL_ZONE_EXIT = 0.45   # 45% - hysteresis

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Utility functions
def get_distance(p1, p2):
    """Hitung jarak antara 2 titik"""
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def get_midpoint(p1, p2):
    """Dapatkan titik tengah antara 2 titik"""
    return ((p1.x + p2.x) / 2, (p1.y + p2.y) / 2)

def map_value(value, in_min, in_max, out_min, out_max):
    """Map nilai dari range satu ke range lainnya"""
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

class MouseController:
    """Kontrol mouse dengan smoothing"""
    
    def __init__(self):
        self.screen_width, self.screen_height = pyautogui.size()
        self.position_history = deque(maxlen=Config.SMOOTHING)
        self.prev_cursor_pos = None
        self.click_cooldown = 0
        self.right_click_cooldown = 0
        self.scroll_cooldown = 0
        self.is_clicking = False
        self.is_right_clicking = False
        
        # Scroll smoothing
        self.scroll_y_history = deque(maxlen=Config.SCROLL_SMOOTHING)
        self.scroll_zone = 'MIDDLE'
        self.scroll_accumulator = 0  # Akumulasi untuk smooth scroll
        
    def update(self, hand_landmarks, frame_width, frame_height):
        """Update posisi mouse berdasarkan landmark tangan"""
        
        # Ambil posisi jari telunjuk (index finger tip)
        index_tip = hand_landmarks.landmark[8]
        thumb_tip = hand_landmarks.landmark[4]
        middle_tip = hand_landmarks.landmark[12]
        
        # Convert ke koordinat layar dengan margin (x sudah mirror karena gambar diflip)
        margin = Config.SCREEN_MARGIN
        x = int(index_tip.x * (self.screen_width - 2 * margin)) + margin
        y = int(index_tip.y * (self.screen_height - 2 * margin)) + margin
        
        # Smoothing dengan moving average
        self.position_history.append((x, y))
        avg_x = int(sum(p[0] for p in self.position_history) / len(self.position_history))
        avg_y = int(sum(p[1] for p in self.position_history) / len(self.position_history))
        
        # Apply dead zone
        if self.prev_cursor_pos:
            dx = avg_x - self.prev_cursor_pos[0]
            dy = avg_y - self.prev_cursor_pos[1]
            if abs(dx) < Config.DEAD_ZONE:
                avg_x = self.prev_cursor_pos[0]
            if abs(dy) < Config.DEAD_ZONE:
                avg_y = self.prev_cursor_pos[1]
        
        # Gerakkan mouse
        pyautogui.moveTo(avg_x, avg_y)
        self.prev_cursor_pos = (avg_x, avg_y)
        
        # Deteksi gestures
        self._detect_gestures(hand_landmarks)
        
        # Return pinch distance untuk visual feedback
        pinch_dist = get_distance(thumb_tip, index_tip) * 1000
        return pinch_dist
    
    def _detect_gestures(self, hand_landmarks):
        """Deteksi berbagai gesture"""
        
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        middle_tip = hand_landmarks.landmark[12]
        ring_tip = hand_landmarks.landmark[16]
        pinky_tip = hand_landmarks.landmark[20]
        
        # ===== LEFT CLICK: Pinch jempol & telunjuk =====
        # Hitung jarak (normalized 0-1) lalu scale ke 0-1000
        pinch_distance = get_distance(thumb_tip, index_tip) * 1000
        
        # DEBUG: Print jarak ke terminal
        print(f"Pinch distance: {pinch_distance:.1f} | is_clicking: {self.is_clicking} | cooldown: {self.click_cooldown}")
        
        # Klik hanya saat pinch terdeteksi dan sebelumnya tidak sedang pinch
        if pinch_distance < Config.PINCH_THRESHOLD:
            if not self.is_clicking and self.click_cooldown == 0:
                print(">>> LEFT CLICK! <<<")
                pyautogui.click()
                self.is_clicking = True
                self.click_cooldown = 10
        else:
            # Reset click state ketika jari terpisah
            if pinch_distance > Config.PINCH_RELEASE_THRESHOLD:
                if self.is_clicking:
                    print("--- Released ---")
                self.is_clicking = False
            
        # ===== RIGHT CLICK: Pinch jempol & jari tengah =====
        pinch_middle_distance = get_distance(thumb_tip, middle_tip) * 1000
        
        # Pastikan telunjuk tidak ikut pinch (bedakan dengan left click)
        if (pinch_middle_distance < Config.PINCH_THRESHOLD and 
            pinch_distance > Config.PINCH_THRESHOLD + 20):  # Telunjuk harus jauh
            
            if not self.is_right_clicking and self.right_click_cooldown == 0:
                pyautogui.rightClick()
                self.is_right_clicking = True
                self.right_click_cooldown = 20
        else:
            if pinch_middle_distance > Config.PINCH_RELEASE_THRESHOLD:
                self.is_right_clicking = False
        
        # ===== SCROLL: 2 jari naik/turun =====
        # Scroll dengan geser 2 jari ke zona atas/bawah layar
        
        # Ambil posisi PIP (sendi) untuk referensi
        index_pip = hand_landmarks.landmark[6]
        middle_pip = hand_landmarks.landmark[10]
        ring_pip = hand_landmarks.landmark[14]
        pinky_pip = hand_landmarks.landmark[18]
        
        # Cek apakah telunjuk dan tengah TERANGKAT
        index_extended = index_tip.y < index_pip.y
        middle_extended = middle_tip.y < middle_pip.y
        
        # Cek apakah ring dan pinky LIPAT (tidak ikut scroll)
        ring_folded = ring_tip.y > ring_pip.y
        pinky_folded = pinky_tip.y > pinky_pip.y
        
        # Scroll valid hanya jika: 2 jari atas + 2 jari bawah lipat
        scroll_valid = index_extended and middle_extended and ring_folded and pinky_folded
        
        # Hitung posisi Y rata-rata 2 jari
        avg_y = (index_tip.y + middle_tip.y) / 2
        
        # Simpan history untuk smoothing
        self.scroll_y_history.append(avg_y)
        
        # Smooth Y position
        smooth_y = sum(self.scroll_y_history) / len(self.scroll_y_history)
        
        # Tentukan zona dengan hysteresis
        enter_threshold = Config.SCROLL_ZONE_ENTER  # 0.35
        exit_threshold = Config.SCROLL_ZONE_EXIT     # 0.45
        
        if smooth_y < enter_threshold:
            new_zone = 'UP'
        elif smooth_y > (1 - enter_threshold):
            new_zone = 'DOWN'
        elif smooth_y > exit_threshold and smooth_y < (1 - exit_threshold):
            new_zone = 'MIDDLE'
        else:
            new_zone = self.scroll_zone  # Tetap di zona sebelumnya
        
        self.scroll_zone = new_zone
        
        # Akumulasi untuk smooth scrolling
        if scroll_valid:
            if self.scroll_zone == 'UP':
                self.scroll_accumulator += 1
            elif self.scroll_zone == 'DOWN':
                self.scroll_accumulator -= 1
            else:
                self.scroll_accumulator = 0  # Reset di tengah
        else:
            self.scroll_accumulator = 0
        
        # DEBUG
        print(f"Scroll: y={smooth_y:.3f}, zone={self.scroll_zone}, accum={self.scroll_accumulator}, valid={scroll_valid}")
        
        # Scroll jika akumulasi cukup dan cooldown habis
        if self.scroll_cooldown == 0:
            if self.scroll_accumulator >= 3:  # 3 frame konsisten di atas
                print(">>> SCROLL UP! <<<")
                pyautogui.scroll(Config.SCROLL_SPEED)
                self.scroll_cooldown = Config.SCROLL_DELAY
                self.scroll_accumulator = 0
            elif self.scroll_accumulator <= -3:  # 3 frame konsisten di bawah
                print(">>> SCROLL DOWN! <<<")
                pyautogui.scroll(-Config.SCROLL_SPEED)
                self.scroll_cooldown = Config.SCROLL_DELAY
                self.scroll_accumulator = 0
        
        # Decrement cooldowns
        if self.click_cooldown > 0:
            self.click_cooldown -= 1
        if self.right_click_cooldown > 0:
            self.right_click_cooldown -= 1
        if self.scroll_cooldown > 0:
            self.scroll_cooldown -= 1
    
    def get_gesture_status(self, hand_landmarks):
        """Dapatkan status gesture untuk ditampilkan"""
        if not hand_landmarks:
            return "No hand detected"
        
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        middle_tip = hand_landmarks.landmark[12]
        
        pinch_distance = get_distance(thumb_tip, index_tip) * 1000
        pinch_middle = get_distance(thumb_tip, middle_tip) * 1000
        
        if self.is_clicking:
            return "LEFT CLICK (HOLD)"
        elif self.is_right_clicking:
            return "RIGHT CLICK"
        elif pinch_distance < Config.PINCH_THRESHOLD:
            return "LEFT CLICK DETECT"
        elif pinch_middle < Config.PINCH_THRESHOLD:
            return "RIGHT CLICK DETECT"
        else:
            # Cek apakah 2 jari terangkat (untuk indikasi bisa scroll)
            index_pip = hand_landmarks.landmark[6]
            middle_pip = hand_landmarks.landmark[10]
            index_extended = hand_landmarks.landmark[8].y < index_pip.y
            middle_extended = hand_landmarks.landmark[12].y < middle_pip.y
            
            # Cek ring & pinky lipat untuk status
            ring_folded = hand_landmarks.landmark[16].y > hand_landmarks.landmark[14].y
            pinky_folded = hand_landmarks.landmark[20].y > hand_landmarks.landmark[18].y
            
            if index_extended and middle_extended and ring_folded and pinky_folded:
                avg_y = (hand_landmarks.landmark[8].y + hand_landmarks.landmark[12].y) / 2
                if avg_y < Config.SCROLL_ZONE_ENTER:
                    return "SCROLL UP (move up)"
                elif avg_y > (1 - Config.SCROLL_ZONE_ENTER):
                    return "SCROLL DOWN (move down)"
                else:
                    return "2 FINGER - MOVE UP/DOWN"
            return "MOVE CURSOR"

def draw_hand_info(image, hand_landmarks, gesture_status, pinch_dist=None):
    """Gambar informasi tambahan di frame dengan visual feedback"""
    h, w, _ = image.shape
    
    # Background panel untuk info
    overlay = image.copy()
    cv2.rectangle(overlay, (10, 10), (380, 165), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
    
    # Teks info
    cv2.putText(image, "HAND TRACKING CONTROL", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(image, f"Status: {gesture_status}", (20, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Tampilkan pinch distance untuk debugging
    if pinch_dist:
        bar_width = 150
        bar_x = 20
        bar_y = 90
        # Background bar
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + 15), (50, 50, 50), -1)
        # Fill bar (max 100)
        fill_width = min(int(pinch_dist), bar_width)
        color = (0, 255, 0) if pinch_dist < Config.PINCH_THRESHOLD else (0, 165, 255)
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + fill_width, bar_y + 15), color, -1)
        cv2.putText(image, f"Pinch: {pinch_dist:.0f}", (bar_x + bar_width + 10, bar_y + 12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    cv2.putText(image, "Press 'Q' to quit", (20, 125), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Petunjuk scroll
    cv2.putText(image, "Scroll: 2 jari atas/bawah", (20, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    
    # Visual scroll zones (zona hijau transparan)
    zone_height = int(h * 0.25)
    overlay = image.copy()
    # Zona atas - scroll up
    cv2.rectangle(overlay, (0, 0), (w, zone_height), (0, 255, 0), -1)
    cv2.putText(overlay, "UP", (w//2 - 20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 0), 2)
    # Zona bawah - scroll down
    cv2.rectangle(overlay, (0, h - zone_height), (w, h), (0, 255, 0), -1)
    cv2.putText(overlay, "DOWN", (w//2 - 35, h - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 0), 2)
    cv2.addWeighted(overlay, 0.1, image, 0.9, 0, image)
    
    # Highlight jari yang aktif
    if hand_landmarks:
        # Index finger tip dengan warna berdasarkan status
        index_tip = hand_landmarks.landmark[8]
        cx, cy = int(index_tip.x * w), int(index_tip.y * h)
        color = (0, 0, 255) if "CLICK" in gesture_status else (0, 255, 255)
        cv2.circle(image, (cx, cy), 12, color, -1)
        cv2.circle(image, (cx, cy), 12, (0, 0, 0), 2)
        
        # Gambar garis antara jempol dan telunjuk
        thumb_tip = hand_landmarks.landmark[4]
        tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)
        line_color = (0, 255, 0) if "CLICK" in gesture_status else (255, 255, 255)
        cv2.line(image, (cx, cy), (tx, ty), line_color, 3)
    
    return image

def main():
    """Main function"""
    print("=" * 50)
    print("HAND TRACKING WITH GESTURE CONTROL")
    print("=" * 50)
    print("\nFitur:")
    print("  🖱️  Gerakkan telunjuk = Move cursor")
    print("  👆 Pinch jempol+telunjuk = Left click")
    print("  ✌️  Pinch jempol+tengah = Right click")
    print("  ⬆️  2 jari atas = Scroll UP")
    print("  ⬇️  2 jari bawah = Scroll DOWN")
    print("  🖥️  Tekan 'F' = Fullscreen ON/OFF")
    print("\nTekan 'Q' untuk keluar")
    print("=" * 50)
    
    # Inisialisasi kamera - resolusi tinggi untuk fullscreen
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Inisialisasi mouse controller
    mouse = MouseController()
    
    # Matikan pyautogui failsafe untuk pengalaman yang lebih baik
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.01
    
    # Buat window resizable dan fullscreen capable
    cv2.namedWindow('Hand Tracking Control', cv2.WINDOW_NORMAL)
    
    # State fullscreen
    is_fullscreen = False
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Gagal membaca frame dari kamera")
            break
        
        # Resize ke fullscreen jika aktif (sebelum proses untuk performa lebih baik)
        if is_fullscreen:
            screen_w, screen_h = pyautogui.size()
            image = cv2.resize(image, (screen_w, screen_h), interpolation=cv2.INTER_LINEAR)
        
        # Flip untuk mirror effect (seperti kaca), lalu convert ke RGB
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Proses dengan MediaPipe
        results = hands.process(image_rgb)
        
        gesture_status = "No hand detected"
        pinch_dist = None
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Gambar landmark tangan
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Update mouse control (gunakan ukuran image yang mungkin sudah resized)
                h, w, _ = image.shape
                pinch_dist = mouse.update(hand_landmarks, w, h)
                
                # Get status
                gesture_status = mouse.get_gesture_status(hand_landmarks)
        
        # Gambar informasi
        image = draw_hand_info(image, 
                              results.multi_hand_landmarks[0] if results.multi_hand_landmarks else None,
                              gesture_status,
                              pinch_dist)
        
        # Tampilkan frame
        cv2.imshow('Hand Tracking Control', image)
        
        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('f'):
            # Toggle fullscreen
            is_fullscreen = not is_fullscreen
            if is_fullscreen:
                cv2.setWindowProperty('Hand Tracking Control', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                print("🖥️  Fullscreen ON")
            else:
                cv2.setWindowProperty('Hand Tracking Control', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Hand Tracking Control', Config.CAM_WIDTH, Config.CAM_HEIGHT)
                print("🖥️  Fullscreen OFF")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("\nProgram ditutup. Sampai jumpa!")

if __name__ == "__main__":
    main()
