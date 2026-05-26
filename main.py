import sys
import cv2
import numpy as np
import mediapipe as mp
import os
import time
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.uic import loadUi

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    change_roi_orig_signal = pyqtSignal(np.ndarray)
    change_roi_left_orig_signal = pyqtSignal(np.ndarray)
    change_roi_bin_signal = pyqtSignal(np.ndarray)
    change_roi_mouth_orig = pyqtSignal(np.ndarray)
    change_roi_mouth_bin = pyqtSignal(np.ndarray)
    update_metrics_signal = pyqtSignal(float, int, float, int, str)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        pygame.mixer.init()
        self.alarm_path = "alarm.mp3"
        self.is_alarm_playing = False
        
        if os.path.exists(self.alarm_path):
            pygame.mixer.music.load(self.alarm_path)

    def run(self):
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        RIGHT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
        LEFT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]
        MOUTH_LANDMARKS = [78, 81, 13, 311, 308, 178, 14, 402]

        cap = cv2.VideoCapture(1)

        mata_tertutup_sejak = None   
        mulut_terbuka_sejak = None   
        jumlah_pelanggaran_kantuk = 0  
        sudah_hitung_kantuk_ini = False   
        sudah_hitung_menguap_ini = False  
        status_driver = "NORMAL / AMAN"
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        while self._run_flag:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.resize(frame, (640, 420))
            h, w, _ = frame.shape

            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            enhanced_frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

            rgb_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            eye_pixels = 0
            mouth_pixels = 0
            ear_score = 0.0
            mar_score = 0.0
            kondisi_pejam_frame_ini = False
            kondisi_menguap_frame_ini = False

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    
                    # 1. KOTAK 1 & 2: ROI MATA KANAN DAN KIRI
                    r_eye_coords = np.array([(int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h)) for idx in RIGHT_EYE_LANDMARKS])
                    rex, rey, rew, reh = cv2.boundingRect(r_eye_coords)
                    rex1, rey1 = max(0, rex - 10), max(0, rey - 10)
                    rex2, rey2 = min(w, rex + rew + 10), min(h, rey + reh + 10)
                    roi_right_eye = enhanced_frame[rey1:rey2, rex1:rex2]

                    l_eye_coords = np.array([(int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h)) for idx in LEFT_EYE_LANDMARKS])
                    lex, ley, lew, leh = cv2.boundingRect(l_eye_coords)
                    lex1, ley1 = max(0, lex - 10), max(0, ley - 10)
                    lex2, ley2 = min(w, lex + lew + 10), min(h, ley + leh + 10)
                    roi_left_eye = enhanced_frame[ley1:ley2, lex1:lex2]

                    if roi_right_eye.size > 0 and roi_right_eye.shape[0] > 2 and roi_right_eye.shape[1] > 2:
                        # 3. KOTAK 3: BINERISASI OTSU MATA
                        roi_gray = cv2.cvtColor(roi_right_eye, cv2.COLOR_BGR2GRAY)
                        roi_blur = cv2.GaussianBlur(roi_gray, (3, 3), 0)
                        _, roi_bin_pure = cv2.threshold(roi_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                        
                        roi_morph_eye = cv2.morphologyEx(roi_bin_pure, cv2.MORPH_CLOSE, kernel)
                        eye_pixels = cv2.countNonZero(roi_morph_eye)

                        p2_p6_r = np.linalg.norm(r_eye_coords[1] - r_eye_coords[5])
                        p3_p5_r = np.linalg.norm(r_eye_coords[2] - r_eye_coords[4])
                        p1_p4_r = np.linalg.norm(r_eye_coords[0] - r_eye_coords[3])
                        ear_right = (p2_p6_r + p3_p5_r) / (2.0 * p1_p4_r)

                        p2_p6_l = np.linalg.norm(l_eye_coords[1] - l_eye_coords[5])
                        p3_p5_l = np.linalg.norm(l_eye_coords[2] - l_eye_coords[4])
                        p1_p4_l = np.linalg.norm(l_eye_coords[0] - l_eye_coords[3])
                        ear_left = (p2_p6_l + p3_p5_l) / (2.0 * p1_p4_l)

                        ear_score = (ear_right + ear_left) / 2.0

                        self.change_roi_orig_signal.emit(roi_right_eye.copy())
                        self.change_roi_left_orig_signal.emit(roi_left_eye.copy())
                        self.change_roi_bin_signal.emit(roi_bin_pure.copy())

                    # 4. KOTAK 4: ROI MULUT
                    mouth_coords = np.array([(int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h)) for idx in MOUTH_LANDMARKS])
                    mx, my, mw, mh = cv2.boundingRect(mouth_coords)
                    mx1, my1 = max(0, mx - 12), max(0, my - 12)
                    mx2, my2 = min(w, mx + mw + 12), min(h, my + mh + 12)
                    
                    roi_mouth = enhanced_frame[my1:my2, mx1:mx2]

                    # PROTEKSI KETat: Pastikan ROI valid, punya area, dan ukurannya logis sebelum diolah OpenCV
                    if roi_mouth is not None and roi_mouth.size > 0 and roi_mouth.shape[0] > 5 and roi_mouth.shape[1] > 5:
                        try:
                            mouth_gray = cv2.cvtColor(roi_mouth, cv2.COLOR_BGR2GRAY)
                            
                            if mouth_gray is not None and mouth_gray.size > 0 and mouth_gray.shape[0] > 5 and mouth_gray.shape[1] > 5:
                                # 5. KOTAK 5: MORFOLOGI MULUT (Dibungkus try-except agar jika ada anomali citra tidak akan crash)
                                mouth_blur = cv2.GaussianBlur(mouth_gray, (5, 5), 0)
                                _, mouth_bin = cv2.threshold(mouth_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                                mouth_morph = cv2.morphologyEx(mouth_bin, cv2.MORPH_CLOSE, kernel)
                                mouth_pixels = cv2.countNonZero(mouth_morph)

                                m_vertical = np.linalg.norm(mouth_coords[2] - mouth_coords[6])
                                m_horizontal = np.linalg.norm(mouth_coords[0] - mouth_coords[4])
                                mar_score = m_vertical / m_horizontal if m_horizontal != 0 else 0

                                self.change_roi_mouth_orig.emit(roi_mouth.copy())
                                self.change_roi_mouth_bin.emit(mouth_morph.copy())
                        except cv2.error:
                            pass # Lewati frame jika terjadi anomali ukuran array C++

                    # --- LOGIKA MONITORING ---
                    if ear_score < 0.17 or eye_pixels < 100: 
                        if mata_tertutup_sejak is None:
                            mata_tertutup_sejak = time.time() 
                        if time.time() - mata_tertutup_sejak >= 5.0:
                            kondisi_pejam_frame_ini = True
                            if not sudah_hitung_kantuk_ini:
                                jumlah_pelanggaran_kantuk += 1
                                sudah_hitung_kantuk_ini = True 
                    else:
                        mata_tertutup_sejak = None   
                        sudah_hitung_kantuk_ini = False

                    if mar_score > 0.62 or mouth_pixels > 1250:
                        if mulut_terbuka_sejak is None:
                            mulut_terbuka_sejak = time.time()
                        if time.time() - mulut_terbuka_sejak >= 2.5: 
                            kondisi_menguap_frame_ini = True
                            if not sudah_hitung_menguap_ini:
                                jumlah_pelanggaran_kantuk += 1
                                sudah_hitung_menguap_ini = True
                    else:
                        mulut_terbuka_sejak = None 
                        sudah_hitung_menguap_ini = False 

                    if jumlah_pelanggaran_kantuk >= 3:
                        status_driver = f"🚨🚨 ALARM AKTIF! PELANGGARAN >3x ({jumlah_pelanggaran_kantuk}x)"
                        if not self.is_alarm_playing and os.path.exists(self.alarm_path):
                            pygame.mixer.music.play(-1)
                            self.is_alarm_playing = True
                    elif kondisi_pejam_frame_ini:
                        status_driver = f"🚨 KANTUK DETEKSI! (Total: {jumlah_pelanggaran_kantuk}/3)"
                    elif kondisi_menguap_frame_ini:
                        status_driver = f"⚠️ MENGUAP DETEKSI! (Total: {jumlah_pelanggaran_kantuk}/3)"
                    else:
                        status_driver = f"NORMAL / AMAN (Pelanggaran: {jumlah_pelanggaran_kantuk}/3)"

                    if jumlah_pelanggaran_kantuk < 3 and self.is_alarm_playing:
                        pygame.mixer.music.stop()
                        self.is_alarm_playing = False

                    box_color = (0, 0, 255) if jumlah_pelanggaran_kantuk >= 3 else ((0, 255, 255) if (kondisi_pejam_frame_ini or kondisi_menguap_frame_ini) else (46, 204, 113))
                    cv2.rectangle(frame, (rex1, rey1), (rex2, rey2), box_color, 2)
                    cv2.rectangle(frame, (lex1, ley1), (lex2, ley2), box_color, 2)
                    cv2.rectangle(frame, (mx1, my1), (mx2, my2), box_color, 2)

            self.change_pixmap_signal.emit(frame)
            self.update_metrics_signal.emit(ear_score, eye_pixels, mar_score, mouth_pixels, status_driver)

        pygame.mixer.music.stop()
        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()


class DashboardApp(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi("dashboard.ui", self)
        self.btnStart.clicked.connect(self.start_camera)
        self.btnStop.clicked.connect(self.stop_camera)
        self.thread = None

    def start_camera(self):
        if self.thread is None or not self.thread.isRunning():
            self.thread = VideoThread()
            self.thread.change_pixmap_signal.connect(self.update_main_image)
            self.thread.change_roi_orig_signal.connect(self.update_roi_orig)
            self.thread.change_roi_left_orig_signal.connect(self.update_roi_left_orig)
            self.thread.change_roi_bin_signal.connect(self.update_roi_bin)
            self.thread.change_roi_mouth_orig.connect(self.update_mouth_orig)
            self.thread.change_roi_mouth_bin.connect(self.update_mouth_bin)
            self.thread.update_metrics_signal.connect(self.update_metrics_display)
            self.thread.start()
            self.lblStatus.setText("MENYIAPKAN SISTEM...")

    def stop_camera(self):
        if self.thread is not None and self.thread.isRunning():
            self.thread.stop()
            self.lblVideo.setText("Kamera Standby - Hubungkan Perangkat")
            self.lblRoiOriginal.clear()
            self.lblRoiLeftOriginal.clear()
            self.lblRoiBinary.clear()
            self.lblRoiMouthOrig.clear()
            self.lblRoiMouthBin.clear()
            self.lblStatus.setText("SISTEM SIAP")

    def convert_cv_qt(self, cv_img, width, height):
        if len(cv_img.shape) == 3:
            h, w, ch = cv_img.shape
            bytes_per_line = ch * w
            convert_to_Qt_format = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format_BGR888)
        else:
            h, w = cv_img.shape
            bytes_per_line = w
            convert_to_Qt_format = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
        
        p = convert_to_Qt_format.scaled(width, height, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
        return QPixmap.fromImage(p)

    @pyqtSlot(np.ndarray)
    def update_main_image(self, cv_img):
        self.lblVideo.setPixmap(self.convert_cv_qt(cv_img, 640, 420))

    @pyqtSlot(np.ndarray)
    def update_roi_orig(self, cv_img):
        self.lblRoiOriginal.setPixmap(self.convert_cv_qt(cv_img, 200, 140))

    @pyqtSlot(np.ndarray)
    def update_roi_left_orig(self, cv_img):
        self.lblRoiLeftOriginal.setPixmap(self.convert_cv_qt(cv_img, 200, 140))

    @pyqtSlot(np.ndarray)
    def update_roi_bin(self, cv_img):
        self.lblRoiBinary.setPixmap(self.convert_cv_qt(cv_img, 200, 140))

    @pyqtSlot(np.ndarray)
    def update_mouth_orig(self, cv_img):
        self.lblRoiMouthOrig.setPixmap(self.convert_cv_qt(cv_img, 200, 140))

    @pyqtSlot(np.ndarray)
    def update_mouth_bin(self, cv_img):
        self.lblRoiMouthBin.setPixmap(self.convert_cv_qt(cv_img, 200, 140))

    @pyqtSlot(float, int, float, int, str)
    def update_metrics_display(self, ear, eye_pix, mar, mouth_pix, status):
        self.lblEar.setText(f"Eye Aspect Ratio (EAR) : {ear:.2f}")
        self.lblPixelCount.setText(f"Luas Piksel Mata (PCD) : {eye_pix} px")
        self.lblMar.setText(f"Mouth Open Ratio (MAR): {mar:.2f}")
        self.lblMouthPixel.setText(f"Luas Rongga Mulut(PCD): {mouth_pix} px")
        self.lblStatus.setText(status)

        if "ALARM AKTIF" in status:
            self.lblStatus.setStyleSheet("color: #ffffff; background-color: #dc2626; font-size: 15px; font-weight: bold; border-radius: 6px;")
        elif "🚨" in status or "⚠️" in status:
            self.lblStatus.setStyleSheet("color: #1f2937; background-color: #fbbf24; font-size: 15px; font-weight: bold; border-radius: 6px;")
        else:
            self.lblStatus.setStyleSheet("color: #10b981; background-color: #f9fafb; font-size: 15px; font-weight: bold; border-radius: 6px; border: 1px solid #e5e7eb;")

    def closeEvent(self, event):
        self.stop_camera()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DashboardApp()
    window.show()
    sys.exit(app.exec_())