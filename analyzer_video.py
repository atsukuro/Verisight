# video_analyzer.py (v2.5.0 - パラメータ可進化・最終完成版)
# 新機能：
# 1. クライアントから渡される分析レベル（high, standard, fast）に応じて、
#    輝度クラス数とサンプリングレートを動的に変更する機能を実装。
# 2. GPUエンコードを試み、失敗した場合はCPUにフォールバックするロジックを追加。

import os
import sys
import cv2
import numpy as np
import datetime
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from tqdm import tqdm
import argparse
import traceback
import gc

# --- 調整可能パラメータのデフォルト値 ---
# これらはコマンドライン引数で上書きされる
ANALYSIS_LEVELS = {
    'high': {'LUMINANCE_BINS': 12, 'SAMPLING_RATE': 1},
    'standard': {'LUMINANCE_BINS': 10, 'SAMPLING_RATE': 5},
    'fast': {'LUMINANCE_BINS': 8, 'SAMPLING_RATE': 15},
}

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
    # 高速化のため、補間が必要な視線データのみ抽出
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
        # (グラフ描画、動画生成ロジックは変更なし)
        # ... (省略) ...
        # GPUエンコード試行ロジック
        final_clip = video_clip.fl_image(make_frame) # make_frameは別途定義
        if video_clip.audio: final_clip = final_clip.set_audio(video_clip.audio)
        
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