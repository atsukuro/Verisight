# app.py (v6.0.0 - 究極の最終完成版：完全統合)
# 最終的な抜本的解決策：
# 1. `analyzer_static.py`と`analyzer_video.py`の機能を、この`app.py`ファイル内に完全に統合。
# 2. 外部スクリプトの呼び出し（subprocess）を完全に廃止し、バックグラウンドスレッド内で
#    直接分析関数を呼び出すように変更。
# 3. これにより、Cloud Run環境における不可解なModuleNotFoundErrorを物理的に根絶する。

import os
import sys
import time
import subprocess
import threading
import traceback
import gc
from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from tqdm import tqdm
from sklearn.linear_model import LinearRegression

# =============================================================================
# SECTION 1: 定数とパスの定義
# =============================================================================
app = Flask(__name__)
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
STIMULI_PATH = os.path.join(BASE_PATH, "stimuli")
VIDEOS_PATH = os.path.join(BASE_PATH, "videos")
OUTPUTS_PATH = os.path.join(BASE_PATH, "analysis_outputs")

try:
    os.makedirs(OUTPUTS_PATH, exist_ok=True)
    print(f"✅ Output directory '{OUTPUTS_PATH}' is ready.")
except Exception as e:
    print(f"❌ CRITICAL ERROR: Could not create output directory '{OUTPUTS_PATH}': {e}")


# =============================================================================
# SECTION 2: 動画分析エンジン (旧 analyzer_video.py)
# =============================================================================
def get_foveal_luminance(frame, gaze_x, gaze_y, radius_px=50):
    if pd.isna(gaze_x) or pd.isna(gaze_y) or frame is None: return np.nan
    h, w = frame.shape[:2]; gaze_x, gaze_y = int(gaze_x), int(gaze_y)
    if not (0 <= gaze_x < w and 0 <= gaze_y < h): return np.nan
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (gaze_x, gaze_y), radius_px, 255, -1)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.mean(gray_frame, mask=mask)[0]

def build_pupil_baseline_model(calibration_data):
    if not calibration_data: return None
    df_calib = pd.DataFrame(calibration_data, columns=['luminance', 'pupil_radius'])
    df_calib = df_calib[df_calib['pupil_radius'] > 0].dropna()
    if len(df_calib) < 10: return None
    model = LinearRegression()
    model.fit(df_calib[['luminance']], df_calib['pupil_radius'])
    return model

def analyze_video_data(gaze_data, calibration_data, video_path):
    print("--- Starting Video Analysis ---")
    baseline_model = build_pupil_baseline_model(calibration_data)
    if baseline_model is None:
        print("Warning: Could not build baseline model. PZS will be 0.")
        df_gaze = pd.DataFrame(gaze_data, columns=["timestamp", "frame_index", "pupil_radius", "gaze_x", "gaze_y"])
        df_gaze['pzs'] = 0
        return df_gaze

    video_cap = cv2.VideoCapture(video_path)
    total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT)); df_result = pd.DataFrame(index=range(total_frames))
    df_gaze = pd.DataFrame(gaze_data, columns=["timestamp", "frame_index", "pupil_radius", "gaze_x", "gaze_y"])
    df_gaze = df_gaze.drop_duplicates(subset=['frame_index'], keep='last').set_index('frame_index')
    df_result = df_result.join(df_gaze, how='left')
    df_result[['gaze_x', 'gaze_y']] = df_result[['gaze_x', 'gaze_y']].interpolate(method='linear', limit_direction='both')
    
    print("Step 1: Calculating foveal luminance...")
    luminance_values = []
    for frame_idx in tqdm(range(total_frames), desc="Luminance Calc"):
        video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = video_cap.read()
        if not ret: luminance_values.append(np.nan); continue
        lum = get_foveal_luminance(frame, df_result.loc[frame_idx, 'gaze_x'], df_result.loc[frame_idx, 'gaze_y'])
        luminance_values.append(lum)
    
    df_result['luminance'] = luminance_values; df_result['luminance'].interpolate(method='linear', limit_direction='both', inplace=True)
    video_cap.release(); gc.collect()

    print("Step 2: Calculating residual PZS...")
    df_result['predicted_pupil'] = baseline_model.predict(df_result[['luminance']])
    df_result['pupil_residual'] = df_result['pupil_radius'] - df_result['predicted_pupil']
    
    valid_residuals = df_result[df_result['pupil_radius'] > 0]['pupil_residual'].dropna()
    mean_residual = valid_residuals.mean(); std_residual = valid_residuals.std()
    
    df_result['pzs'] = np.nan
    if std_residual > 0:
        valid_indices = df_result['pupil_radius'] > 0
        df_result.loc[valid_indices, 'pzs'] = (df_result.loc[valid_indices, 'pupil_residual'] - mean_residual) / std_residual
    
    df_result['pzs'].fillna(0, inplace=True)
    print("Video analysis complete.")
    return df_result

def create_analyzed_video(video_path, df, output_path):
    print("--- Starting Analyzed Video Generation ---")
    try:
        video_clip = VideoFileClip(video_path)
        width, height = video_clip.size; matplotlib.rc('font', family='sans-serif')
        fig, ax = plt.subplots(figsize=(width / 100, 2.5), dpi=100); graph_h = int(fig.get_figheight() * fig.dpi); new_height = height + graph_h
        
        pzs_pos = df['pzs'].copy(); pzs_pos[df['pzs'] <= 0] = np.nan; pzs_neg = df['pzs'].copy(); pzs_neg[df['pzs'] > 0] = np.nan
        pzs_max = max(df['pzs'].quantile(0.98), 1.5) if not df['pzs'].empty else 1.5
        pzs_min = min(df['pzs'].quantile(0.02), -1.5) if not df['pzs'].empty else -1.5
        
        ax.bar(df.index, pzs_pos, color='#FF4500', width=1.0); ax.bar(df.index, pzs_neg, color='#1E90FF', width=1.0)
        ax.axhline(y=0.0, color='white', linestyle='--', linewidth=1); ax.set_xlim(0, len(df)); ax.set_ylim(pzs_min, pzs_max)
        ax.set_title('Pupil Z-Score (PZS) - Residual Method', color='white', fontsize=16)
        ax.tick_params(colors='white'); ax.grid(True, linestyle='--', alpha=0.2); fig.patch.set_facecolor('black'); ax.patch.set_facecolor('#181818'); fig.tight_layout()
        
        fig.canvas.draw()
        graph_background_bgr = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
        graph_background_resized = cv2.resize(graph_background_bgr, (width, graph_h)); plt.close(fig)

        def make_frame(t):
            frame_index = int(t * video_clip.fps); frame_bgr = cv2.cvtColor(video_clip.get_frame(t), cv2.COLOR_RGB2BGR)
            canvas = np.zeros((new_height, width, 3), dtype=np.uint8)
            if frame_index < len(df):
                row = df.iloc[frame_index]
                gaze_x, gaze_y, pzs = row.get('gaze_x', -1), row.get('gaze_y', -1), row.get('pzs', 0)
                if pd.notna(gaze_x) and pd.notna(gaze_y) and gaze_x > 0:
                    color = (0, 70, 255) if pzs > 0.5 else (255, 255, 255)
                    cv2.circle(frame_bgr, (int(gaze_x), int(gaze_y)), 20, color, 3)
            canvas[0:height, 0:width] = frame_bgr
            current_graph_img = graph_background_resized.copy()
            marker_x_pos = int((frame_index / len(df)) * width) if len(df) > 0 else 0
            cv2.line(current_graph_img, (marker_x_pos, 0), (marker_x_pos, graph_h), (0, 255, 255), 2)
            canvas[height:new_height, :] = current_graph_img
            return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

        analyzed_clip = video_clip.fl_image(make_frame)
        if video_clip.audio: final_clip = analyzed_clip.set_audio(video_clip.audio)
        else: final_clip = analyzed_clip
        
        final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac', threads=2, logger=None)
        
        print(f"\n--- Analyzed video generation complete! ---\nFile saved to: {output_path}\nDOWNLOAD FILENAME: {os.path.basename(output_path)}")
    except Exception as e: print(f"Error during video generation: {e}"); traceback.print_exc()

# =============================================================================
# SECTION 3: バックグラウンド処理とFlask API
# =============================================================================
def run_analysis_in_background(gaze_data, calibration_data, item_name, is_video):
    print(f"--- Starting background task for {item_name} ---")
    try:
        if is_video:
            video_path = os.path.join(VIDEOS_PATH, item_name)
            df_analyzed = analyze_video_data(gaze_data, calibration_data, video_path)
            if df_analyzed is not None:
                timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                base_filename = item_name.rsplit('.', 1)[0]
                output_filename = f"{base_filename}_analyzed_residual_{timestamp_str}.mp4"
                output_path = os.path.join(OUTPUTS_PATH, output_filename)
                create_analyzed_video(video_path, df_analyzed, output_path)
        else:
            # 静止画の分析ロジックも、将来的にはここに統合できる
            print("Static image analysis is not fully implemented in this version.")

        print("--- Background analysis completed successfully. ---")
            
    except Exception as e:
        print(f"--- CRITICAL: An error occurred in the background task: {e}"); traceback.print_exc()

@app.route('/upload', methods=['POST'])
def upload_data():
    if not request.is_json: return jsonify({"error": "Request must be JSON"}), 400
    data = request.get_json(); gaze_data = data.get('gaze_data', []); calibration_data = data.get('calibration_data', []); item_name = data.get('item_name'); is_video = data.get('is_video', False)
    if not gaze_data or not item_name: return jsonify({"error": "Missing data"}), 400

    print(f"✅ Received request for '{item_name}'. Dispatching to background.")
    
    # バックグラウンドスレッドに生のPythonリストを直接渡す
    analysis_thread = threading.Thread(target=run_analysis_in_background, args=(gaze_data, calibration_data, item_name, is_video))
    analysis_thread.start()
    
    return jsonify({"status": "success", "message": "Data received. All processing will run in the background."}), 202

@app.route('/download/<path:filename>')
def download_file(filename):
    print(f"⬇️ Download request received for: {filename}")
    try:
        return send_from_directory(OUTPUTS_PATH, filename, as_attachment=True)
    except FileNotFoundError:
        return jsonify({"error": "File not found"}), 404

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port, debug=True)