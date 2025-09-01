# client.py (v6.2.0 - 最終完成版)
# 最終的な洗練：
# 1. 統合キャリブレーションの導入により不要となった「分析レベル選択」画面を削除。
# 2. これにより、ユーザー体験がさらにシンプルで直感的になる。

import cv2
import numpy as np
import mediapipe as mp
import math
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from collections import deque
import time
import sys
import traceback
import easygui
import os
import glob
import json
import requests
from moviepy.editor import VideoFileClip

# --- Windows向け高DPI対応 ---
if sys.platform.startswith('win'):
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception as e:
        print(f"警告: 高DPI設定に失敗しました。({e})")

# --- 調整可能パラメータ ---
SLIDE_DURATION_SECONDS = 5
CALIBRATION_POINT_DURATION = 3.0

# --- 基本設定 ---
WEBCAM_WIDTH, WEBCAM_HEIGHT = 1280, 720
SMOOTHING_BUFFER_SIZE = 5
BLINK_THRESHOLD = 0.25
CALIBRATION_MIN_FRAMES = 2

class GazeTracker:
    # (変更なし)
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
        if not results.multi_face_landmarks: return None
        landmarks = np.array([(lm.x * frame.shape[1], lm.y * frame.shape[0]) for lm in results.multi_face_landmarks[0].landmark])
        if (self._get_eye_aspect_ratio(landmarks, self.L_EAR_INDICES) + self._get_eye_aspect_ratio(landmarks, self.R_EAR_INDICES)) / 2.0 < BLINK_THRESHOLD: return {"is_blinking": True}
        for eye, i_pts, c_pts in [("Left", self.L_IRIS_POINTS, self.L_EAR_INDICES), ("Right", self.R_IRIS_POINTS, self.R_EAR_INDICES)]:
            gaze_data = self._track_one_eye(landmarks, i_pts, c_pts)
            if gaze_data: gaze_data["is_blinking"] = False; return gaze_data
        return None

class GazeMapper:
    # (変更なし)
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
        self.W, self.H=0, 0; self.screen=None; self.font_xl, self.font_l, self.font_m, self.font_s=None, None, None, None
        self.cap = None; self.tracker, self.mapper = GazeTracker(), GazeMapper()
        self.smoother = deque(maxlen=5); self.calibration_data = []; self.gaze_data = []
        self.COLOR_BACKGROUND = (20, 30, 40); self.COLOR_ACCENT = (0, 150, 170); self.COLOR_ACCENT_HOVER = (0, 200, 220)
        self.COLOR_TEXT = (220, 230, 240); self.COLOR_TEXT_DIM = (150, 160, 170)
        self.COLOR_SUCCESS = (0, 220, 180); self.COLOR_FAIL = (250, 80, 100); self.COLOR_WARN = (255, 180, 0)

    def init_pygame_for_ui(self):
        # (変更なし)
        import pygame
        globals()['pygame'] = pygame
        pygame.init()
        info = pygame.display.Info()
        self.W, self.H = info.current_w, info.current_h
        self.screen = pygame.display.set_mode((self.W, self.H), pygame.FULLSCREEN)
        self.font_xl = pygame.font.SysFont("Meiryo", 80); self.font_l = pygame.font.SysFont("Meiryo", 52); self.font_m = pygame.font.SysFont("Meiryo", 36); self.font_s = pygame.font.SysFont("Meiryo", 24)

    def init_camera_and_pygame_for_exp(self):
        # (変更なし)
        import pygame
        globals()['pygame'] = pygame
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW); self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        pygame.init(); pygame.mixer.init(); info = pygame.display.Info(); self.W, self.H = info.current_w, info.current_h
        self.screen = pygame.display.set_mode((self.W, self.H), pygame.FULLSCREEN); pygame.mouse.set_visible(False)
        self.font_xl = pygame.font.SysFont("Meiryo", 80); self.font_l = pygame.font.SysFont("Meiryo", 52); self.font_m = pygame.font.SysFont("Meiryo", 36); self.font_s = pygame.font.SysFont("Meiryo", 24)

    def _fade_transition(self, direction='out', duration=300):
        # (変更なし)
        fade_surface = pygame.Surface((self.W, self.H))
        fade_surface.fill((0, 0, 0))
        for alpha in range(0, 255, 15):
            if direction == 'out': fade_surface.set_alpha(alpha)
            else: fade_surface.set_alpha(255 - alpha)
            self.screen.blit(fade_surface, (0, 0))
            pygame.display.flip()
            pygame.time.delay(int(duration / (255/15)))

    def _draw_text(self, text, pos, font, color=None, center=False):
        # (変更なし)
        if color is None: color = self.COLOR_TEXT
        surf = font.render(text, True, color); rect = surf.get_rect(center=pos) if center else surf.get_rect(topleft=pos); self.screen.blit(surf, rect)

    def run_pre_check(self):
        # (変更なし)
        self._fade_transition('in')
        while True:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE: self._fade_transition('out'); return False
                    if event.key == pygame.K_SPACE: self._fade_transition('out'); return True
            ret, frame = self.cap.read()
            if not ret: continue
            frame = cv2.flip(frame, 1);
            gaze_info = self.tracker.process_frame(frame)
            self.screen.fill(self.COLOR_BACKGROUND)
            
            p_h, p_w = frame.shape[:2]; preview_scale = 0.7; disp_w, disp_h = int(p_w * preview_scale), int(p_h * preview_scale)
            disp_x, disp_y = (self.W-disp_w)//2, (self.H-disp_h)//2
            frame_rect = pygame.Rect(disp_x - 5, disp_y - 5, disp_w + 10, disp_h + 10)
            pygame.draw.rect(self.screen, self.COLOR_ACCENT, frame_rect, 2, border_radius=5)
            
            prev_surf = pygame.surfarray.make_surface(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).swapaxes(0, 1))
            self.screen.blit(pygame.transform.smoothscale(prev_surf, (disp_w, disp_h)), (disp_x, disp_y))
            
            self._draw_text("カメラ確認", (self.W//2, 80), self.font_xl, center=True)
            self._draw_text("顔全体が枠内に収まっていることを確認してください", (self.W//2, 160), self.font_s, color=self.COLOR_TEXT_DIM, center=True)
            
            status = "認識中" if gaze_info and not gaze_info.get("is_blinking") else "認識できません"
            color = self.COLOR_SUCCESS if status == "認識中" else self.COLOR_FAIL
            self._draw_text(f"ステータス: {status}", (self.W//2, self.H - 140), self.font_l, center=True, color=color)
            self._draw_text("準備ができたら [SPACE] を押してください", (self.W//2, self.H - 80), self.font_m, center=True)
            pygame.display.flip()

    def run_unified_calibration(self):
        # (変更なし)
        self.screen.fill(self.COLOR_BACKGROUND)
        self._draw_text("統合キャリブレーション", (self.W//2, self.H//2 - 120), self.font_xl, center=True)
        self._draw_text("画面に表示される9つの点を、それぞれ見つめてください。", (self.W//2, self.H//2 + 150), self.font_m, center=True)
        self._draw_text("背景の明るさが滑らかに変化します。", (self.W//2, self.H//2 + 200), self.font_s, color=self.COLOR_TEXT_DIM, center=True)
        pygame.display.flip(); pygame.time.wait(4000)
        self._fade_transition('out'); pygame.time.wait(300)

        points = [(p, q) for q in (0.1, 0.5, 0.9) for p in (0.1, 0.5, 0.9)]
        gaze_mapping_data = []; self.calibration_data = []

        for i, (px, py) in enumerate(points):
            sx, sy = int(self.W * px), int(self.H * py); collected_coords = []
            start_time = time.time()
            while time.time() - start_time < CALIBRATION_POINT_DURATION:
                progress = (i + (time.time() - start_time) / CALIBRATION_POINT_DURATION) / (len(points) - 1)
                bg_lum = int(np.clip(255 * (1 - abs(1 - 2 * progress)), 0, 255))
                self.screen.fill((bg_lum, bg_lum, bg_lum))
                
                if int((time.time() - start_time) * 2.5) % 2 == 0:
                    if bg_lum > 128:
                        outline_color, ring_color, center_color = (220,220,220), (0,0,0), (220,220,220)
                    else:
                        outline_color, ring_color, center_color = (0,0,0), (220,220,220), (0,0,0)
                    pygame.draw.circle(self.screen, outline_color, (sx,sy), 13, 1)
                    pygame.draw.circle(self.screen, ring_color, (sx,sy), 12)
                    pygame.draw.circle(self.screen, center_color, (sx,sy), 5)

                pygame.display.flip()

                ret, frame = self.cap.read()
                if not ret: continue
                gaze_info = self.tracker.process_frame(cv2.flip(frame, 1))
                if gaze_info and not gaze_info.get("is_blinking"):
                    collected_coords.append(gaze_info["normalized_gaze"])
                    self.calibration_data.append([bg_lum, gaze_info["pupil_radius"]])
                for e in pygame.event.get(pygame.KEYDOWN):
                    if e.key == pygame.K_ESCAPE: print("中断"); return False
            
            if len(collected_coords) >= CALIBRATION_MIN_FRAMES:
                gaze_mapping_data.append(((sx, sy), np.median(np.array(collected_coords), axis=0))); print(f"Point {i+1}/9 OK")
            else:
                print(f"Point {i+1}/9 FAILED")
        
        self.mapper.train(gaze_mapping_data)
        return True

    def run_calibration_validation(self):
        # (変更なし)
        self._fade_transition('in'); print("キャリブレーション精度を確認してください。"); pygame.mouse.set_visible(True)
        target_pos = [self.W // 2, self.H // 2]; running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE: self._fade_transition('out'); pygame.mouse.set_visible(False); return "exit"
                    if event.key == pygame.K_SPACE: self._fade_transition('out'); pygame.mouse.set_visible(False); return "continue"
                    if event.key == pygame.K_r: self._fade_transition('out'); pygame.mouse.set_visible(False); return "retry"
            target_pos = pygame.mouse.get_pos(); ret, frame = self.cap.read()
            if not ret: continue
            gaze_info = self.tracker.process_frame(cv2.flip(frame, 1)); self.screen.fill(self.COLOR_BACKGROUND)
            self._draw_text("精度確認", (self.W//2, 80), self.font_xl, center=True); self._draw_text("マウスで的を動かし、視線（緑の円）の追従を確認してください", (self.W//2, 160), self.font_s, color=self.COLOR_TEXT_DIM, center=True)
            self._draw_text("[SPACE] 続行", (self.W//4, self.H - 80), self.font_m, color=self.COLOR_SUCCESS, center=True); self._draw_text("[R] やり直し", (self.W//2, self.H - 80), self.font_m, color=self.COLOR_WARN, center=True)
            self._draw_text("[ESC] 終了", (self.W*3//4, self.H - 80), self.font_m, color=self.COLOR_FAIL, center=True)
            pygame.draw.circle(self.screen, self.COLOR_TEXT, target_pos, 20, 2); pygame.draw.circle(self.screen,(0,0,0),target_pos,5)
            if gaze_info and not gaze_info.get("is_blinking"):
                screen_pos = self.mapper.map_gaze(gaze_info["normalized_gaze"])
                if screen_pos:
                    self.smoother.append(screen_pos)
                    if self.smoother: s_pos = np.mean(self.smoother, axis=0); pygame.draw.circle(self.screen, self.COLOR_SUCCESS, (int(s_pos[0]), int(s_pos[1])), 25, 3)
            pygame.display.flip()
        pygame.mouse.set_visible(False); return "exit"

    def run_slideshow_experiment(self, image_paths):
        # (変更なし)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0); self.cap.set(cv2.CAP_PROP_AUTO_WB, 0); print("カメラ自動調整を無効化。")
        try:
            self.gaze_data = []
            overall_start_time = time.time()
            for slide_num, image_path in enumerate(image_paths, 1):
                try:
                    stimulus_img = pygame.image.load(image_path)
                    stimulus_img = pygame.transform.smoothscale(stimulus_img, (self.W, self.H))
                except pygame.error as e: print(f"スライド読込失敗: {e}"); continue
                
                self.screen.blit(stimulus_img, (0,0)); pygame.display.flip()
                slide_start_time = time.time()
                while time.time() - slide_start_time < SLIDE_DURATION_SECONDS:
                    for event in pygame.event.get():
                        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE: return "exit_by_user"
                    ret, frame = self.cap.read()
                    if not ret: continue
                    gaze_info = self.tracker.process_frame(cv2.flip(frame, 1))
                    if gaze_info and not gaze_info.get("is_blinking"):
                        screen_pos = self.mapper.map_gaze(gaze_info["normalized_gaze"])
                        self.gaze_data.append([time.time()-overall_start_time, slide_num, gaze_info['pupil_radius'], screen_pos[0] if screen_pos else -1, screen_pos[1] if screen_pos else -1])
                    else:
                        self.gaze_data.append([time.time()-overall_start_time, slide_num, -1, -1, -1])
                    pygame.time.Clock().tick(60)
            return "completed"
        finally:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1); self.cap.set(cv2.CAP_PROP_AUTO_WB, 1); print("カメラ自動調整を有効化。")

    def run_video_experiment(self, video_path):
        # (変更なし)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0); self.cap.set(cv2.CAP_PROP_AUTO_WB, 0); print("カメラ自動調整を無効化。")
        temp_audio_path = "temp_audio.mp3"; video_clip = None
        try:
            video_clip = VideoFileClip(video_path)
            if video_clip.audio: video_clip.audio.write_audiofile(temp_audio_path, logger=None, codec='mp3')
            else: temp_audio_path = None
        except Exception as e:
            print(f"動画ファイルの読み込み/音声抽出に失敗: {e}")
            if video_clip: video_clip.close(); return "error"
        try:
            self.gaze_data = []
            overall_start_time = time.time()
            if temp_audio_path: pygame.mixer.music.load(temp_audio_path); pygame.mixer.music.play()
            clock = pygame.time.Clock(); video_start_time = time.time()
            running = True
            while running:
                elapsed_time = time.time() - video_start_time
                if elapsed_time >= video_clip.duration: running = False; continue
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE: running = False; return "exit_by_user"
                ret_cam, cam_frame = self.cap.read()
                if not ret_cam: continue
                video_frame = video_clip.get_frame(elapsed_time)
                pygame_surface = pygame.surfarray.make_surface(video_frame.swapaxes(0, 1))
                self.screen.blit(pygame.transform.smoothscale(pygame_surface, (self.W, self.H)), (0, 0))
                gaze_info = self.tracker.process_frame(cv2.flip(cam_frame, 1))
                item_identifier = int(elapsed_time * video_clip.fps)
                if gaze_info and not gaze_info.get("is_blinking"):
                    screen_pos = self.mapper.map_gaze(gaze_info["normalized_gaze"])
                    self.gaze_data.append([time.time()-overall_start_time, item_identifier, gaze_info['pupil_radius'], screen_pos[0] if screen_pos else -1, screen_pos[1] if screen_pos else -1])
                else:
                    self.gaze_data.append([time.time()-overall_start_time, item_identifier, -1, -1, -1])
                pygame.display.flip()
                clock.tick(video_clip.fps * 1.2)
            return "completed"
        finally:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1); self.cap.set(cv2.CAP_PROP_AUTO_WB, 1); print("カメラ自動調整を有効化。")
            if pygame.mixer.get_init(): pygame.mixer.music.stop()
            if video_clip: video_clip.close()
            if temp_audio_path and os.path.exists(temp_audio_path):
                try: os.remove(temp_audio_path)
                except (PermissionError, OSError) as e: print(f"警告: {temp_audio_path}の削除に失敗しました: {e}")

    def _send_data_to_cloud(self, item_name, is_video=False):
        # (変更なし)
        if not self.gaze_data: print("記録データがありません。"); return
        url = "https://verisight-365650487388.asia-northeast1.run.app/upload"
        payload = {"participant_id": "user_001", "timestamp": time.strftime('%Y%m%d_%H%M%S'), "item_name": item_name, "is_video": is_video, "calibration_data": self.calibration_data, "gaze_data": self.gaze_data}
        try:
            self.screen.fill((0,0,0)); self._draw_text("データを送信しています...", (self.W//2, self.H//2), self.font_m, center=True); pygame.display.flip()
            print(f"\nデータをクラウドサーバー ({url}) に送信中...")
            response = requests.post(url, json=payload, timeout=60)
            if response.status_code in [200, 202]: print(f"✅ データ送信成功！ サーバーからの応答: {response.json()}")
            else: print(f"❌ データ送信失敗。ステータスコード: {response.status_code}, 内容: {response.text}")
        except requests.exceptions.RequestException as e: print(f"❌ サーバーへの接続に失敗しました: {e}")

    def run_mode_selection(self):
        # (変更なし)
        self.init_pygame_for_ui()
        pygame.mouse.set_visible(True)
        button_width, button_height = 500, 120
        button_slides = pygame.Rect(self.W//2 - button_width//2, self.H//2 - 150, button_width, button_height)
        button_video = pygame.Rect(self.W//2 - button_width//2, self.H//2 + 30, button_width, button_height)
        while True:
            mouse_pos = pygame.mouse.get_pos()
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE): return None
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if button_slides.collidepoint(event.pos): self._fade_transition('out'); return "静止画"
                    if button_video.collidepoint(event.pos): self._fade_transition('out'); return "動画"
            
            self.screen.fill(self.COLOR_BACKGROUND)
            self._draw_text("Verisight", (self.W//2, 120), self.font_xl, center=True); self._draw_text("実行するモードを選択してください", (self.W//2, 200), self.font_s, color=self.COLOR_TEXT_DIM, center=True)
            slide_color = self.COLOR_ACCENT_HOVER if button_slides.collidepoint(mouse_pos) else self.COLOR_ACCENT
            pygame.draw.rect(self.screen, slide_color, button_slides, border_radius=20); self._draw_text("静止画アンケート", button_slides.center, self.font_m, center=True)
            video_color = self.COLOR_ACCENT_HOVER if button_video.collidepoint(mouse_pos) else self.COLOR_ACCENT
            pygame.draw.rect(self.screen, video_color, button_video, border_radius=20); self._draw_text("動画評価", button_video.center, self.font_m, center=True)
            pygame.display.flip(); pygame.time.Clock().tick(60)

    def run(self):
        # ★★★ 修正点：分析レベル選択を削除 ★★★
        mode = self.run_mode_selection()
        if not mode: print("モードが選択されませんでした。終了します。"); return
        if 'pygame' in globals(): pygame.quit()
        
        item_path, item_name, is_video = None, None, False
        if "静止画" in mode:
            folder_path = easygui.diropenbox(title="スライド画像フォルダを選択")
            if not folder_path: print("フォルダが選択されませんでした。終了します。"); return
            item_name = os.path.basename(folder_path); item_path = sorted(glob.glob(os.path.join(folder_path, '*.jpg')) + glob.glob(os.path.join(folder_path, '*.png')))
            if not item_path: easygui.msgbox("フォルダに画像がありません。"); return
        elif "動画" in mode:
            video_path = easygui.fileopenbox(title="評価する動画ファイルを選択", filetypes=["*.mp4", "*.mov", "*.avi"])
            if not video_path: print("動画ファイルが選択されませんでした。終了します。"); return
            item_name = os.path.basename(video_path); item_path = video_path; is_video = True
            
        print("システムを初期化しています..."); self.init_camera_and_pygame_for_exp()
        
        while True:
            if not self.run_pre_check(): break
            if not self.run_unified_calibration():
                if not self.run_retry_screen("キャリブレーションデータが不足しています。"): break
                else: continue
            if not self.mapper.is_trained:
                if self.run_retry_screen("キャリブレーションに失敗しました。"): continue
                else: break
            validation_result = self.run_calibration_validation()
            if validation_result == "continue":
                if is_video:
                    experiment_result = self.run_video_experiment(item_path)
                else:
                    experiment_result = self.run_slideshow_experiment(item_path)
                if experiment_result == "exit_by_user": print("ユーザーによって測定が中断されました。"); break
                
                self.screen.fill(self.COLOR_BACKGROUND); self._draw_text("測定終了", (self.W//2, self.H//2), self.font_l, center=True); pygame.display.flip(); time.sleep(1)
                self._send_data_to_cloud(item_name=item_name, is_video=is_video)
                self.screen.fill(self.COLOR_BACKGROUND); self._draw_text("処理完了", (self.W//2, self.H//2), self.font_l, center=True); pygame.display.flip(); time.sleep(3)
                break
            elif validation_result == "retry": print("キャリブレーションをやり直します..."); continue
            else: break
        print("プログラムを終了します。")

    def run_retry_screen(self, message):
        # (変更なし)
        self.screen.fill(self.COLOR_BACKGROUND)
        self._draw_text(message, (self.W//2, self.H//2 - 80), self.font_l, center=True, color=self.COLOR_FAIL)
        self._draw_text("やり直しますか？", (self.W//2, self.H//2), self.font_m, center=True)
        self._draw_text("[Y] はい / [N] いいえ", (self.W//2, self.H//2 + 60), self.font_m, center=True)
        pygame.display.flip()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_y: return True
                    if event.key == pygame.K_n: return False

if __name__ == '__main__':
    app = None
    try:
        app = MainApp()
        app.run()
    except Exception as e:
        print(f"予期せぬエラー: {e}"); traceback.print_exc()
    finally:
        if 'pygame' in globals() and pygame.get_init(): pygame.quit()
        if app and app.cap and app.cap.isOpened(): app.cap.release()
        sys.exit()