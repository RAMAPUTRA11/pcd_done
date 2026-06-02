import sys
import cv2
import numpy as np
import mediapipe as mp
import os
import time
from pygame import mixer  # Menggunakan pygame untuk play/stop audio yang stabil
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt5.uic import loadUi

# Konfigurasi Path Dataset Lokal hasil ekstrak curl
DATASET_PATH = "dataset/mrl"

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    change_roi_orig_signal = pyqtSignal(np.ndarray)
    change_roi_left_orig_signal = pyqtSignal(np.ndarray)
    # change_roi_bin_signal dihapus sesuai permintaan
    change_roi_mouth_orig = pyqtSignal(np.ndarray)
    # change_roi_mouth_bin dihapus sesuai permintaan
    
    # Isinya: EAR, PixelCount, MAR, MouthPixel, StatusText
    update_metrics_signal = pyqtSignal(float, int, float, int, str)

    def __init__(self, mode='REG'):
        super().__init__()
        self._run_flag = True
        self.mode = mode 
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Mapping Indeks Landmark
        self.LANDMARKS = {
            "mata_kanan": [33, 160, 158, 133, 153, 144],
            "mata_kiri": [362, 385, 387, 263, 373, 380],
            "alis_kanan": [70, 63, 105, 66, 107],
            "alis_kiri": [336, 296, 334, 293, 300],
            "mulut": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 415, 310, 311, 312, 13, 82, 81, 80],
            "rahang": [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454]
        }

        # Initialize Pygame Mixer untuk Audio Alarm
        mixer.init()
        self.alarm_path = "alarm.mp3"

        self.counter_mengantuk = 0
        self.mata_tertutup_start_time = None
        self.alarm_active = False
        self.alarm_start_time = None
        
        self.sedang_menguap = False 
        self.terakhir_menguap_time = 0 # Mengunci jeda waktu uapan

    def run(self):
        cap = cv2.VideoCapture(1)
        
        while self._run_flag:
            ret, frame = cap.read()
            if not ret: 
                continue
            
            frame = cv2.resize(frame, (640, 420))
            current_time = time.time()
            
            # --- LOGIKA MATIKAN ALARM OTOMATIS SETELAH 10 DETIK ---
            if self.alarm_active and self.alarm_start_time:
                if current_time - self.alarm_start_time >= 10.0:
                    mixer.music.stop()
                    self.alarm_active = False
                    self.alarm_start_time = None

            # --- PREPROCESSING CITRA DIGITAL (PCD): LAB + CLAHE ---
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            limg = cv2.merge((clahe.apply(l), a, b))
            enhanced_frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            
            if self.mode == 'REG':
                cv2.putText(enhanced_frame, "POSISIKAN WAJAH LALU KLIK REGISTRASI", (30, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                self.change_pixmap_signal.emit(enhanced_frame)
                
            elif self.mode == 'MONITOR':
                h_f, w_f, _ = enhanced_frame.shape
                rgb_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
                results = self.mp_face_mesh.process(rgb_frame)
                
                ear, pixel_count, mar, mouth_pixel = 0.0, 0, 0.0, 0
                status_driver = "NORMAL"
                
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        coords = {}
                        for organ, indices in self.LANDMARKS.items():
                            coords[organ] = []
                            for idx in indices:
                                pt = face_landmarks.landmark[idx]
                                px_x = int(pt.x * w_f)
                                px_y = int(pt.y * h_f)
                                coords[organ].append((px_x, px_y))
                                cv2.circle(enhanced_frame, (px_x, px_y), 1, (0, 255, 0), -1)

                        # Hitung EAR (Mata Kanan + Mata Kiri)
                        mk = coords["mata_kanan"]
                        m_kanan_v1 = np.linalg.norm(np.array(mk[1]) - np.array(mk[5]))
                        m_kanan_v2 = np.linalg.norm(np.array(mk[2]) - np.array(mk[4]))
                        m_kanan_h  = np.linalg.norm(np.array(mk[0]) - np.array(mk[3]))
                        ear_kanan  = (m_kanan_v1 + m_kanan_v2) / (2.0 * m_kanan_h)

                        ml = coords["mata_kiri"]
                        m_kiri_v1 = np.linalg.norm(np.array(ml[1]) - np.array(ml[5]))
                        m_kiri_v2 = np.linalg.norm(np.array(ml[2]) - np.array(ml[4]))
                        m_kiri_h  = np.linalg.norm(np.array(ml[0]) - np.array(ml[3]))
                        ear_kiri  = (m_kiri_v1 + m_kiri_v2) / (2.0 * m_kiri_h)
                        
                        ear = (ear_kanan + ear_kiri) / 2.0

                        # Hitung MAR (Mouth Open Ratio)
                        mulut_titik = coords["mulut"]
                        mar_v = np.linalg.norm(np.array(mulut_titik[16]) - np.array(mulut_titik[5])) 
                        mar_h = np.linalg.norm(np.array(mulut_titik[0]) - np.array(mulut_titik[10]))  
                        mar = mar_v / mar_h

                        # --- EKSTRAKSI ROI KOTAK PCD BAWAH ---
                        x_pts_r = [p[0] for p in mk]
                        y_pts_r = [p[1] for p in mk]
                        roi_r = enhanced_frame[max(0, min(y_pts_r)-10):min(h_f, max(y_pts_r)+10), max(0, min(x_pts_r)-10):min(w_f, max(x_pts_r)+10)]

                        x_pts_l = [p[0] for p in ml]
                        y_pts_l = [p[1] for p in ml]
                        roi_l = enhanced_frame[max(0, min(y_pts_l)-10):min(h_f, max(y_pts_l)+10), max(0, min(x_pts_l)-10):min(w_f, max(x_pts_l)+10)]

                        x_pts_m = [p[0] for p in mulut_titik]
                        y_pts_m = [p[1] for p in mulut_titik]
                        roi_mouth = enhanced_frame[max(0, min(y_pts_m)-12):min(h_f, max(y_pts_m)+12), max(0, min(x_pts_m)-12):min(w_f, max(x_pts_m)+12)]

                        # --- SEGMENTASI CITRA REAL-TIME (Kalkulasi Background Tetap Jalan Aktif) ---
                        if roi_r.size > 0:
                            gray_eye = cv2.cvtColor(roi_r, cv2.COLOR_BGR2GRAY)
                            blur_eye = cv2.GaussianBlur(gray_eye, (3, 3), 0)
                            _, bin_eye = cv2.threshold(blur_eye, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                            pixel_count = cv2.countNonZero(bin_eye)

                        if roi_mouth.size > 0:
                            gray_mouth = cv2.cvtColor(roi_mouth, cv2.COLOR_BGR2GRAY)
                            _, bin_mouth = cv2.threshold(gray_mouth, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                            bin_mouth_close = cv2.morphologyEx(bin_mouth, cv2.MORPH_CLOSE, kernel)
                            mouth_pixel = cv2.countNonZero(bin_mouth_close)

                        # =================================================================
                        # LOGIKA SAKLAR + COOLDOWN ANTI-DOUBLE COUNTING
                        # =================================================================
                        if not hasattr(self, 'mulut_sudah_hitung'): self.mulut_sudah_hitung = False

                        # 1. Deteksi Menguap (Membuka, Menutup, + Jeda Cooldown 3 Detik)
                        if mar >= 0.65:
                            if current_time - self.terakhir_menguap_time > 3.0:
                                self.sedang_menguap = True
                        else:
                            if self.sedang_menguap:
                                self.counter_mengantuk += 1  
                                self.sedang_menguap = False  
                                self.terakhir_menguap_time = current_time 
                                print(f"Menguap selesai terhitung! Total Kejadian: {self.counter_mengantuk}/4")

                        # 2. Cek Kondisi Mata Merem
                        if ear < 0.23:
                            if self.mata_tertutup_start_time is None:
                                self.mata_tertutup_start_time = current_time 
                            else:
                                durasi_merem = current_time - self.mata_tertutup_start_time
                                if durasi_merem >= 2.5:
                                    self.counter_mengantuk += 1
                                    self.mata_tertutup_start_time = None 
                                    print(f"Kantuk terhitung! Total Kejadian: {self.counter_mengantuk}/4")
                        else:
                            self.mata_tertutup_start_time = None 

                        # =================================================================
                        # 3. Akumulasi Pemicu Alarm (PENGUNCI 4/4 DAN RESET OTOMATIS SINKRON)
                        # =================================================================
                        if self.counter_mengantuk >= 4:
                            self.counter_mengantuk = 4
                            
                        if self.counter_mengantuk == 4 or self.alarm_active:
                            status_driver = "KRITIS: ALARM AKTIF!"
                            
                            if not self.alarm_active:
                                try:
                                    if os.path.exists(self.alarm_path):
                                        mixer.music.load(self.alarm_path)
                                        mixer.music.play(-1) 
                                        self.alarm_active = True
                                        self.alarm_start_time = current_time
                                except Exception as e:
                                    print(f"Gagal memutar audio: {e}")
                                    
                            if self.alarm_active and (current_time - self.alarm_start_time >= 10.0):
                                mixer.music.stop()         
                                self.alarm_active = False 
                                self.alarm_start_time = None
                                self.counter_mengantuk = 0 
                                print("Alarm selesai berbunyi 10 detik. Counter otomatis kembali ke 0.")
                                
                        else:
                            if ear < 0.23:
                                s_durasi = int(current_time - self.mata_tertutup_start_time) if self.mata_tertutup_start_time else 0
                                status_driver = f"MATA MEREM ({s_durasi}s)"
                            elif mar >= 0.65 or self.sedang_menguap or (current_time - self.terakhir_menguap_time <= 3.0 and self.terakhir_menguap_time != 0):
                                status_driver = "MENGUAP DETECTED"
                            else:
                                status_driver = "NORMAL"    

                        # Tampilkan info Counter di pojok video monitoring
                        cv2.putText(enhanced_frame, f"Kantuk: {self.counter_mengantuk}/4", (480, 40), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                        # Emit output gambar ke box PCD (Hanya mengirim gambar ROI Asli)
                        if roi_r.size > 0: self.change_roi_orig_signal.emit(roi_r)
                        if roi_l.size > 0: self.change_roi_left_orig_signal.emit(roi_l)
                        if roi_mouth.size > 0: self.change_roi_mouth_orig.emit(roi_mouth)
                        # Emit binerisasi dan morfologi telah dihapus dari antrean pipeline
                else:
                    status_driver = "WAJAH TIDAK TERDETEKSI"
                    self.mata_tertutup_start_time = None
                
                self.update_metrics_signal.emit(ear, pixel_count, mar, mouth_pixel, status_driver)
                self.change_pixmap_signal.emit(enhanced_frame)
                
        cap.release()
        mixer.quit()

    def stop(self):
        self._run_flag = False
        mixer.music.stop()
        self.wait()


# =====================================================================
# CLASS DASHBOARD APP (DIBERSIHKAN DARI KONEKSI BINER)
# =====================================================================
class DashboardApp(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi("dashboard.ui", self)
        
        if not os.path.exists(DATASET_PATH):
            QMessageBox.warning(self, "Dataset Tidak Ditemukan", f"Folder '{DATASET_PATH}' tidak terdeteksi.")
        
        self.btnStart.setEnabled(False)
        self.is_registered = False
        
        self.btnRegistrasi.clicked.connect(self.proses_registrasi)
        self.btnStart.clicked.connect(self.mulai_monitoring)
        self.btnStop.clicked.connect(self.hentikan_kamera)
        
        self.buka_thread_registrasi()

    def buka_thread_registrasi(self):
        self.thread = VideoThread(mode='REG')
        self.thread.change_pixmap_signal.connect(self.render_kamera_utama)
        self.thread.start()

    def proses_registrasi(self):
        self.is_registered = True
        self.lblStatus.setText("WAJAH TERDAFTAR! SILAKAN KLIK MULAI")
        self.lblStatus.setStyleSheet("color: #0d9488; background-color: #f0fdfa; font-size: 15px; font-weight: bold; border-radius: 6px; border: 1px solid #99f6e4;")
        self.btnStart.setEnabled(True)

    def mulai_monitoring(self):
        if not self.is_registered: return
        if hasattr(self, 'thread') and self.thread.isRunning():
            self.thread.stop()
            
        self.lblStatus.setText("MONITORING BERJALAN")
        self.lblStatus.setStyleSheet("color: #2563eb; background-color: #eff6ff; font-size: 15px; font-weight: bold; border-radius: 6px; border: 1px solid #bfdbfe;")
        
        self.thread = VideoThread(mode='MONITOR')
        self.thread.change_pixmap_signal.connect(self.render_kamera_utama)
        
        # Menghubungkan komponen ROI Asli ke UI
        self.thread.change_roi_orig_signal.connect(lambda img: self.render_pcd_box(self.lblRoiOriginal, img))
        self.thread.change_roi_left_orig_signal.connect(lambda img: self.render_pcd_box(self.lblRoiLeftOriginal, img))
        self.thread.change_roi_mouth_orig.connect(lambda img: self.render_pcd_box(self.lblRoiMouthOrig, img))
        
        # Pengikatan ke self.lblRoiBinary dan self.lblRoiMouthBin telah dihapus sepenuhnya
        
        self.thread.update_metrics_signal.connect(self.refresh_metrik_angka)
        self.thread.start()

    @pyqtSlot(np.ndarray)
    def render_kamera_utama(self, cv_img):
        self.lblVideo.setPixmap(self.konversi_gambar(cv_img, 640, 420))

    def render_pcd_box(self, label_target, cv_img):
        label_target.setPixmap(self.konversi_gambar(cv_img, 200, 120))

    @pyqtSlot(float, int, float, int, str)
    def refresh_metrik_angka(self, ear, pixel_count, mar, mouth_pixel, status_text):
        self.lblEar.setText(f"Eye Aspect Ratio (EAR) : {ear:.2f}")
        self.lblPixelCount.setText(f"Luas Piksel Mata (PCD) : {pixel_count} px")
        self.lblMar.setText(f"Mouth Open Ratio (MAR): {mar:.2f}")
        self.lblMouthPixel.setText(f"Luas Rongga Mulut(PCD): {mouth_pixel} px")
        
        if "KRITIS" in status_text or "MENGUAP" in status_text:
            self.lblStatus.setText(status_text)
            self.lblStatus.setStyleSheet("color: #dc2626; background-color: #fef2f2; font-size: 15px; font-weight: bold; border-radius: 6px; border: 1px solid #fca5a5;")
        elif status_text == "WAJAH TIDAK TERDETEKSI":
            self.lblStatus.setText(status_text)
            self.lblStatus.setStyleSheet("color: #64748b; background-color: #f8fafc; font-size: 15px; font-weight: bold; border-radius: 6px; border: 1px solid #cbd5e1;")
        else:
            self.lblStatus.setText(status_text)
            self.lblStatus.setStyleSheet("color: #10b981; background-color: #f0fdfa; font-size: 15px; font-weight: bold; border-radius: 6px; border: 1px solid #bbf7d0;")

    def konversi_gambar(self, img, w, h):
        img = np.require(img, np.uint8, 'C')
        h_img, w_img = img.shape[:2]
        if len(img.shape) == 2:
            bytes_per_line = w_img
            qimg = QImage(img.data, w_img, h_img, bytes_per_line, QImage.Format_Grayscale8)
        else:
            bytes_per_line = 3 * w_img
            qimg = QImage(img.data, w_img, h_img, bytes_per_line, QImage.Format_BGR888)
        return QPixmap.fromImage(qimg).scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    def hentikan_kamera(self):
        if hasattr(self, 'thread') and self.thread.isRunning():
            self.thread.stop()
        self.lblStatus.setText("SISTEM DIMATIKAN")
        self.lblStatus.setStyleSheet("color: #7f1d1d; background-color: #fef2f2; font-size: 15px; font-weight: bold; border-radius: 6px; border: 1px solid #f87171;")
        self.btnStart.setEnabled(False)
        self.is_registered = False

    def closeEvent(self, event):
        self.hentikan_kamera()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DashboardApp()
    window.show()
    sys.exit(app.exec_())