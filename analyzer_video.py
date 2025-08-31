# analyzer_video.py (v2.5.0 - 自己修復機能付き・最終完成版)
# 修正点：
# 1. スクリプト実行の冒頭で、必要なライブラリを強制的にインストールする「自己修復機能」を追加。
#    これにより、Google Cloud Runの不可解な環境不整合問題を完全に解決する。

import os
import sys
import subprocess
import traceback

# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# ★★★ ここからが最後の、そして最も重要な自己修復機能 ★★★
try:
    print("✅ [Self-healing] Verifying critical libraries...")
    from moviepy.editor import VideoFileClip
    from tqdm import tqdm
    print("✅ [Self-healing] All libraries are already available.")
except ImportError:
    print("⚠️ [Self-healing] A required library is missing. Attempting to install...")
    try:
        # sys.executableは、現在このスクリプトを実行しているPythonのパス
        subprocess.check_call([sys.executable, "-m", "pip", "install", "moviepy", "tqdm", "imageio-ffmpeg", "pandas", "opencv-python-headless", "matplotlib"])
        print("✅ [Self-healing] Installation successful. Re-importing libraries...")
        # インストール後に再インポート
        from moviepy.editor import VideoFileClip
        from tqdm import tqdm
        print("✅ [Self-healing] Re-import successful.")
    except Exception as e:
        print(f"❌ [Self-healing] CRITICAL ERROR: Failed to install required libraries: {e}")
        traceback.print_exc()
        sys.exit(1) # インストールに失敗した場合は、ここで処理を終了
# ★★★ ここまでが自己修復機能 ★★★
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★


import cv2
import numpy as np
import datetime
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import gc

# (以降のコードは、v2.4.0から変更ありません)

def get_foveal_luminance(frame, gaze_x, gaze_y, radius_px=50):
    if pd.isna(gaze_x) or pd.isna(gaze_y) or frame is None: return np.nan
    h, w = frame.shape[:2]; gaze_x, gaze_y = int(gaze_x), int(gaze_y)
    if not (0 <= gaze_x < w and 0 <= gaze_y < h): return np.nan
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (gaze_x, gaze_y), radius_px, 255, -1)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.mean(gray_frame, mask=mask)[0]

def determine_adaptation_state(luminance_series, look_back_frames=15):
    diff = luminance_series.diff(periods=look_back_frames)
    state = pd.Series('steady', index=luminance_series.index)
    state[diff > 10] = 'brightening'
    state[diff < -10] = 'darkening'
    return state

def analyze_advanced_z_scores(df_gaze, video_path, level_params):
    print(f"--- Starting Advanced Z-Score Analysis (Level: {level_params['name']}) ---")
    
    LUMINANCE_BINS = level_params['LUMINANCE_BINS']
    SAMPLING_RATE = level_params['SAMPLING_RATE']
    
    video_cap = cv2.VideoCapture(video_path)
    if not video_cap.isOpened():
        print("Error: Could not open video file."); return None
    
    total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    df_result = pd.DataFrame(index=range(total_frames))
    df_gaze = df_gaze.drop_duplicates(subset=['frame_index'], keep='last').set_index('frame_index')
    df_result = df_result.join(df_gaze, how='left')

    print(f"Step 1: Calculating luminance (sampling 1/{SAMPLING_RATE} frames)...")
    luminance_values = []
    gaze_to_interpolate = df_result[['gaze_x', 'gaze_y']].copy()
    gaze_to_interpolate[gaze_to_interpolate < 0] = np.nan
    gaze_to_interpolate.interpolate(method='linear', limit_direction='both', inplace=True)

    for frame_idx in tqdm(range(0, total_frames, SAMPLING_RATE), desc="Luminance Calc"):
        video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = video_cap.read()
        if not ret: 
            df_result.loc[frame_idx, 'luminance'] = np.nan
            continue
        row = gaze_to_interpolate.loc[frame_idx]
        lum = get_foveal_luminance(frame, row['gaze_x'], row['gaze_y'])
        df_result.loc[frame_idx, 'luminance'] = lum
    
    video_cap.release(); gc.collect()

    print("Step 2: Interpolating luminance and determining adaptation state...")
    df_result['luminance'].interpolate(method='linear', limit_direction='both', inplace=True)
    df_result['adaptation'] = determine_adaptation_state(df_result['luminance'])

    print(f"Step 3: Calculating PZS (bins: {LUMINANCE_BINS})...")
    df_result['pzs'] = np.nan
    df_result['lum_bin'] = pd.qcut(df_result['luminance'], LUMINANCE_BINS, labels=False, duplicates='drop')
    valid_pupil_df = df_result[df_result['pupil_radius'] > 0].copy()

    for (lum_bin, adaptation), group in tqdm(valid_pupil_df.groupby(['lum_bin', 'adaptation']), desc="PZS Group Calc"):
        if len(group) < 5: continue
        mean_pupil = group['pupil_radius'].mean(); std_pupil = group['pupil_radius'].std()
        if std_pupil > 0:
            df_result.loc[group.index, 'pzs'] = (group['pupil_radius'] - mean_pupil) / std_pupil
    
    df_result['pzs'].fillna(0, inplace=True)
    print("Advanced Z-Score calculation complete.")
    return df_result

def create_analyzed_video(video_path, df, output_path):
    print("--- Starting Analyzed Video Generation ---")
    try:
        video_clip = VideoFileClip(video_path)
        width, height = video_clip.size; matplotlib.rc('font', family='sans-serif')
        fig, ax = plt.subplots(figsize=(width / 100, 2.5), dpi=100); graph_h = int(fig.get_figheight() * fig.dpi); new_height = height + graph_h
        
        pzs_pos = df['pzs'].copy(); pzs_pos[df['pzs'] <= 0] = np.nan
        pzs_neg = df['pzs'].copy(); pzs_neg[df['pzs'] > 0] = np.nan
        pzs_max = max(df['pzs'].quantile(0.98), 1.5) if not df['pzs'].empty else 1.5
        pzs_min = min(df['pzs'].quantile(0.02), -1.5) if not df['pzs'].empty else -1.5
        
        ax.bar(df.index, pzs_pos, color='#FF4500', width=1.0); ax.bar(df.index, pzs_neg, color='#1E90FF', width=1.0)
        ax.axhline(y=0.0, color='white', linestyle='--', linewidth=1); ax.set_xlim(0, len(df)); ax.set_ylim(pzs_min, pzs_max)
        ax.set_title('Pupil Z-Score (PZS) - Advanced', color='white', fontsize=16)
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
            marker_x_pos = int((frame_index / len(df)) * width)
            cv2.line(current_graph_img, (marker_x_pos, 0), (marker_x_pos, graph_h), (0, 255, 255), 2)
            canvas[height:new_height, :] = current_graph_img
            return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

        analyzed_clip = video_clip.fl_image(make_frame)
        if video_clip.audio: final_clip = analyzed_clip.set_audio(video_clip.audio)
        else: final_clip = analyzed_clip
        
        try:
            print("Attempting to encode with NVIDIA GPU (h264_nvenc)...")
            final_clip.write_videofile(output_path, codec='h264_nvenc', audio_codec='aac', threads=4, logger=None)
            print("Successfully encoded with GPU.")
        except Exception as e:
            print(f"GPU encoding failed: {e}. Falling back to CPU (libx264)...")
            final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac', threads=2, logger=None)
            print("Successfully encoded with CPU.")

        print(f"\n--- Analyzed video generation complete! ---\nFile saved to: {output_path}\nDOWNLOAD FILENAME: {os.path.basename(output_path)}")
    except Exception as e: print(f"Error during video generation: {e}"); traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Verisight Video Analysis Tool (Server)")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--level", default='standard', choices=['high', 'standard', 'fast'])
    args = parser.parse_args()

    ANALYSIS_LEVELS = {
        'high': {'LUMINANCE_BINS': 12, 'SAMPLING_RATE': 1},
        'standard': {'LUMINANCE_BINS': 10, 'SAMPLING_RATE': 5},
        'fast': {'LUMINANCE_BINS': 8, 'SAMPLING_RATE': 15},
    }
    level_params = ANALYSIS_LEVELS.get(args.level, ANALYSIS_LEVELS['standard'])
    level_params['name'] = args.level

    base_dir = "analysis_outputs"
    os.makedirs(base_dir, exist_ok=True)
    
    try:
        df_gaze = pd.read_csv(args.csv)
    except Exception as e: print(f"Error loading gaze data CSV: {e}"); return

    df_analyzed = analyze_advanced_z_scores(df_gaze, args.video, level_params)
    
    if df_analyzed is not None:
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = os.path.basename(args.video).rsplit('.', 1)[0]
        output_filename = f"{base_filename}_analyzed_{args.level}_{timestamp_str}.mp4"
        output_path = os.path.join(base_dir, output_filename)
        create_analyzed_video(args.video, df_analyzed, output_path)
    
    print("--- Video analysis script finished. ---")

if __name__ == '__main__':
    main()