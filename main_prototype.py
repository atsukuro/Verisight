# main_prototype.py (v3.2.2)
# 最終完成版：測定データをクラウドサーバーに送信する機能を搭載。

import cv2
import numpy as np
import mediapipe as mp
import pygame
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from collections import deque
import time
import sys
import traceback
import easygui
import csv
import os
import glob
import json
import requests

# --- 基本設定 ---
WEBCAM_WIDTH, WEBCAM_HEIGHT = 1280, 720
SMOOTHING_BUFFER_SIZE = 5
BLINK_THRESHOLD = 0.25
CALIBRATION_MIN_FRAMES = 2
SLIDE_DURATION_SECONDS = 5

class GazeTracker:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh; self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.L_EAR_INDICES = [362, 385, 387, 263, 373, 380]; self.R_EAR_INDICES = [33, 160, 158, 133, 153, 144]
        self.L_IRIS_POINTS = [474, 475, 476, 477]; self.R_IRIS_POINTS = [469, 470, 471, 472]
    def _get_eye_aspect_ratio(self, landmarks, eye_indices):
        p2_p6=landmarks[eye_indices[1]]-landmarks[eye_indices[5]]; p3_p5=landmarks[eye_indices[2]]-landmarks[eye_indices[4]]; p1_p4=landmarks[eye_indices[0]]-landmarks[eye_indices[3]]
        return (np.linalg.norm(p2_p6) + np.linalg.norm(p3_p5)) / (2.0 * np.linalg.norm(p1_p4)) if np.linalg.norm(p1_p4) > 1e-6 else 0.0
    def _track_one_eye(self, landmarks, iris_indices, eye_corner_indices):
        iris_center = np.mean([landmarks[i] for i in iris_indices], axis=0); pupil_radius = np.linalg.norm(landmarks[iris_indices[1]] - landmarks[iris_indices[3]]) / 2.0
        eye_left_corner, eye_right_corner = landmarks[eye_corner_indices[0]], landmarks[eye_corner_indices[3]]; eye_width = np.linalg.norm(eye_left_corner - eye_right_corner)
        if eye_width < 1e-6: return None
        norm_x = (iris_center[0] - eye_left_corner[0]) / eye_width
        eye_center_y = np.mean([landmarks[i] for i in [eye_corner_indices[1], eye_corner_indices[2], eye_corner_indices[4], eye_corner_indices[5]]], axis=0)[1]
        norm_y = (iris_center[1] - eye_center_y) / (eye_width / 2.0)
        return {"pupil_radius": pupil_radius, "normalized_gaze": (norm_x, norm_y)}
    def process_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); frame_rgb.flags.writeable = False; results = self.face_mesh.process(frame_rgb); frame_rgb.flags.writeable = True
        debug_info = {"landmarks": None, "active_eye": "None"}
        if not results.multi_face_landmarks: return None, debug_info
        landmarks = np.array([(lm.x * frame.shape[1], lm.y * frame.shape[0]) for lm in results.multi_face_landmarks[0].landmark]); debug_info["landmarks"] = landmarks
        if (self._get_eye_aspect_ratio(landmarks, self.L_EAR_INDICES) + self._get_eye_aspect_ratio(landmarks, self.R_EAR_INDICES)) / 2.0 < BLINK_THRESHOLD: return {"is_blinking": True}, debug_info
        for eye, i_pts, c_pts in [("Left", self.L_IRIS_POINTS, self.L_EAR_INDICES), ("Right", self.R_IRIS_POINTS, self.R_EAR_INDICES)]:
            gaze_data = self._track_one_eye(landmarks, i_pts, c_pts)
            if gaze_data: debug_info["active_eye"] = eye; gaze_data["is_blinking"] = False; return gaze_data, debug_info
        return None, debug_info

class GazeMapper:
    def __init__(self): self.poly=PolynomialFeatures(degree=2); self.model_x, self.model_y=LinearRegression(), LinearRegression(); self.is_trained=False
    def train(self, d):
        if len(d) < 9: print("キャリブレーション不足"); self.is_trained = False; return
        X, y = np.array([i[1] for i in d]), np.array([i[0] for i in d]); X_poly = self.poly.fit_transform(X)
        self.model_x.fit(X_poly, y[:, 0]); self.model_y.fit(X_poly, y[:, 1]); self.is_trained = True; print("✅ マッピング学習完了")
    def map_gaze(self, g):
        if not self.is_trained: return None
        gaze_poly = self.poly.transform(np.array([g])); return int(self.model_x.predict(gaze_poly)[0]), int(self.model_y.predict(gaze_poly)[0])

class MainApp:
    def __init__(self):
        self.W, self.H=0, 0; self.screen=None; self.font_l, self.font_s=None, None
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW); self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.tracker, self.mapper = GazeTracker(), GazeMapper(); self.smoother = deque(maxlen=5); self.recorded_data = []
    def init_pygame(self):
        pygame.init(); info = pygame.display.Info(); self.W, self.H = info.current_w, info.current_h
        self.screen = pygame.display.set_mode((self.W, self.H), pygame.FULLSCREEN); pygame.mouse.set_visible(False)
        self.font_l = pygame.font.SysFont("Meiryo", 36); self.font_s = pygame.font.SysFont("Meiryo", 24)
    def _draw_text(self, text, pos, font, color=(255,255,255), center=False):
        surf = font.render(text, True, color); rect = surf.get_rect(center=pos) if center else surf.get_rect(topleft=pos); self.screen.blit(surf, rect)
    def run_pre_check(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE: return False
                    if event.key == pygame.K_SPACE: return True
            ret, frame = self.cap.read();
            if not ret: continue
            frame = cv2.flip(frame, 1); gaze_data, d_info = self.tracker.process_frame(frame); self.screen.fill((30,30,30))
            p_h, p_w = frame.shape[:2]; preview_scale = 0.8; disp_w, disp_h = int(p_w * preview_scale), int(p_h * preview_scale)
            if d_info and d_info.get("landmarks") is not None:
                for idx in self.tracker.L_EAR_INDICES + self.tracker.R_EAR_INDICES + self.tracker.L_IRIS_POINTS + self.tracker.R_IRIS_POINTS:
                    if idx < len(d_info["landmarks"]): x, y = map(int, d_info["landmarks"][idx]); cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            prev_surf = pygame.surfarray.make_surface(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).swapaxes(0, 1))
            self.screen.blit(pygame.transform.smoothscale(prev_surf, (disp_w, disp_h)), ((self.W-disp_w)//2, (self.H-disp_h)//2))
            self._draw_text("キャリブレーション準備", (self.W//2, 50), self.font_l, center=True)
            self._draw_text("準備ができたら [SPACE] を押してください", (self.W//2, self.H - 100), self.font_s, center=True, color=(100,255,100))
            self._draw_text("（反応しない場合は、一度この画面をクリックしてください）", (self.W//2, self.H - 60), self.font_s, center=True, color=(200,200,200))
            status = "DETECTING" if gaze_data and not gaze_data.get("is_blinking") else "FAILING"
            color = (0,255,0) if status == "DETECTING" else (255,0,0)
            self._draw_text(f"認識ステータス: {status}", (self.W//2, self.H - 150), self.font_l, center=True, color=color)
            pygame.display.flip()
    def run_calibration(self):
        self.screen.fill((0,0,0)); self._draw_text("キャリブレーションを開始します", (self.W//2, self.H//2), self.font_l, center=True); pygame.display.flip(); pygame.time.wait(2000)
        points = [(p, q) for q in (0.1, 0.5, 0.9) for p in (0.1, 0.5, 0.9)]; calib_data = []
        for i, (px, py) in enumerate(points):
            sx, sy = int(self.W * px), int(self.H * py); self.screen.fill((0,0,0)); pygame.display.flip(); pygame.time.wait(250)
            start_time, collected_coords = time.time(), []
            while time.time() - start_time < 2.5:
                ret, frame = self.cap.read()
                if not ret: continue
                gaze_data, _ = self.tracker.process_frame(cv2.flip(frame, 1)); self.screen.fill((0,0,0))
                if int((time.time() - start_time) * 4) % 2 == 1: pygame.draw.circle(self.screen,(255,255,255),(sx,sy),20); pygame.draw.circle(self.screen,(0,0,0),(sx,sy),5)
                pygame.display.flip()
                if gaze_data and not gaze_data.get("is_blinking"): collected_coords.append(gaze_data["normalized_gaze"])
                for e in pygame.event.get(pygame.KEYDOWN):
                    if e.key == pygame.K_ESCAPE: print("中断"); return
            if len(collected_coords) >= CALIBRATION_MIN_FRAMES:
                calib_data.append(((sx, sy), np.median(np.array(collected_coords), axis=0))); print(f"Point {i+1}/9 OK")
            else: print(f"Point {i+1}/9 FAILED")
        self.mapper.train(calib_data)
    def run_calibration_validation(self):
        print("キャリブレーション精度を確認してください。"); pygame.mouse.set_visible(True)
        target_pos = [self.W // 2, self.H // 2]; running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE: pygame.mouse.set_visible(False); return "exit"
                    if event.key == pygame.K_SPACE: pygame.mouse.set_visible(False); return "continue"
                    if event.key == pygame.K_r: pygame.mouse.set_visible(False); return "retry"
            target_pos = pygame.mouse.get_pos()
            ret, frame = self.cap.read();
            if not ret: continue
            frame = cv2.flip(frame, 1); gaze_data, _ = self.tracker.process_frame(frame); self.screen.fill((0,0,0))
            self._draw_text("キャリブレーション精度確認", (self.W//2, 50), self.font_l, center=True)
            self._draw_text("マウスで的を動かし、視線（緑円）の追従を確認", (self.W//2, 100), self.font_s, center=True)
            self._draw_text("精度に満足なら [SPACE] で続行", (self.W//2, 130), self.font_s, center=True, color=(100,255,100))
            self._draw_text("やり直す場合は [R] を押してください", (self.W//2, 160), self.font_s, center=True, color=(255,255,100))
            pygame.draw.circle(self.screen,(255,255,255),target_pos,20); pygame.draw.circle(self.screen,(0,0,0),target_pos,5)
            if gaze_data and not gaze_data.get("is_blinking"):
                screen_pos = self.mapper.map_gaze(gaze_data["normalized_gaze"])
                if screen_pos:
                    self.smoother.append(screen_pos)
                    if self.smoother: s_pos = np.mean(self.smoother, axis=0); pygame.draw.circle(self.screen, (0, 255, 0), (int(s_pos[0]), int(s_pos[1])), 25, 3)
            pygame.display.flip()
        pygame.mouse.set_visible(False); return "exit"
    def run_slideshow_experiment(self, image_paths):
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0); self.cap.set(cv2.CAP_PROP_AUTO_WB, 0); print("カメラ自動調整を無効化。")
        try:
            self.screen.fill((128,128,128)); self._draw_text("ベースライン計測中...", (self.W//2, self.H//2 - 50), self.font_s, center=True)
            pupil_sizes = deque(maxlen=150); start_time = time.time()
            while time.time() - start_time < 5.0:
                self.screen.fill((128,128,128)); pygame.draw.circle(self.screen,(255,255,255),(self.W//2,self.H//2),20); pygame.draw.circle(self.screen,(0,0,0),(self.W//2,self.H//2),5); pygame.display.flip()
                ret, frame = self.cap.read()
                if not ret: continue
                gaze_data, _ = self.tracker.process_frame(cv2.flip(frame, 1))
                if gaze_data and not gaze_data.get("is_blinking"): pupil_sizes.append(gaze_data["pupil_radius"])
            baseline_pupil_size = np.mean(pupil_sizes) if pupil_sizes else 0
            print(f"✅ ベースライン計測完了。"); time.sleep(1)
            self.recorded_data = []
            overall_start_time = time.time()
            for slide_num, image_path in enumerate(image_paths, 1):
                try:
                    stimulus_img = pygame.image.load(image_path); img_rect = stimulus_img.get_rect(); scale = min(self.W/img_rect.width, self.H/img_rect.height)
                    scaled_w, scaled_h = int(img_rect.width*scale), int(img_rect.height*scale)
                    stimulus_img = pygame.transform.smoothscale(stimulus_img, (scaled_w, scaled_h)); stimulus_pos = ((self.W-scaled_w)//2, (self.H-scaled_h)//2)
                except pygame.error as e: print(f"スライド読込失敗: {e}"); continue
                slide_start_time = time.time(); running_slide = True
                while running_slide:
                    if time.time() - slide_start_time > SLIDE_DURATION_SECONDS: running_slide = False
                    for event in pygame.event.get():
                        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE: running_slide = False; return "exit_by_user"
                    ret, frame = self.cap.read()
                    if not ret: continue
                    frame = cv2.flip(frame, 1); self.screen.blit(stimulus_img, stimulus_pos); gaze_data, _ = self.tracker.process_frame(frame)
                    screen_pos = None
                    if self.mapper.is_trained and gaze_data and not gaze_data.get("is_blinking"):
                        screen_pos = self.mapper.map_gaze(gaze_data["normalized_gaze"])
                    if gaze_data and not gaze_data.get("is_blinking"):
                        self.recorded_data.append([time.time()-overall_start_time, slide_num, gaze_data['pupil_radius'], screen_pos[0] if screen_pos else -1, screen_pos[1] if screen_pos else -1])
                    pygame.display.flip(); pygame.time.Clock().tick(60)
            return "completed"
        finally:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1); self.cap.set(cv2.CAP_PROP_AUTO_WB, 1); print("カメラ自動調整を有効化。")
    def _send_data_to_cloud(self, num_slides, stimuli_folder_name):
        if not self.recorded_data: print("記録データがありません。"); return
        url = "http://127.0.0.1:5000/upload"
        payload = {"participant_id": "user_001", "timestamp": time.strftime('%Y%m%d_%H%M%S'), "slide_count": num_slides, "stimuli_folder_name": stimuli_folder_name, "gaze_data": self.recorded_data}
        try:
            print(f"\nデータをクラウドサーバー ({url}) に送信中...")
            response = requests.post(url, json=payload, timeout=15)
            if response.status_code == 200: print(f"✅ データ送信成功！ サーバーからの応答: {response.json()}")
            else: print(f"❌ データ送信失敗。ステータスコード: {response.status_code}, 内容: {response.text}")
        except requests.exceptions.RequestException as e: print(f"❌ サーバーへの接続に失敗しました: {e}")
    def run(self):
        folder_path = easygui.diropenbox(title="スライド画像フォルダを選択")
        if not folder_path: print("終了します。"); return
        folder_name = os.path.basename(folder_path)
        image_files = sorted(glob.glob(os.path.join(folder_path, '*.jpg')) + glob.glob(os.path.join(folder_path, '*.png')))
        if not image_files: easygui.msgbox("フォルダに画像がありません。"); return
        self.init_pygame()
        while True:
            if not self.run_pre_check(): break
            self.run_calibration()
            if not self.mapper.is_trained:
                if self.run_retry_screen("キャリブレーションに失敗しました。"): continue
                else: break
            validation_result = self.run_calibration_validation()
            if validation_result == "continue":
                experiment_result = self.run_slideshow_experiment(image_files)
                if experiment_result == "exit_by_user": print("ユーザーによって測定が中断されました。"); break
                self._send_data_to_cloud(num_slides=len(image_files), stimuli_folder_name=folder_name)
                self.screen.fill((0,0,0)); self._draw_text("測定終了", (self.W//2, self.H//2), self.font_l, center=True); pygame.display.flip(); time.sleep(3)
                break
            elif validation_result == "retry":
                print("キャリブレーションをやり直します...")
                continue
            else: # "exit"
                break
        print("プログラムを終了します。")
    def run_retry_screen(self, message):
        while True:
            self.screen.fill((0,0,0))
            self._draw_text(message, (self.W//2, self.H//2 - 50), self.font_l, center=True)
            self._draw_text("やり直しますか？ [Y] / [N]", (self.W//2, self.H//2 + 50), self.font_s, center=True)
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_y: return True
                    if event.key == pygame.K_n: return False

if __name__ == '__main__':
    app = None
    try: app = MainApp(); app.run()
    except Exception as e: print(f"予期せぬエラー: {e}"); traceback.print_exc()
    finally:
        if app and app.cap.isOpened(): app.cap.release()
        if pygame.get_init(): pygame.quit()
        sys.exit()