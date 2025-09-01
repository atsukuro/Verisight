# analyzer_video.py (v3.0.1 - 統合キャリブレーション対応・完全版)
# 修正点：
# 1. create_analyzed_video 関数の省略箇所を完全に復元。

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
from sklearn.linear_model import LinearRegression

def get_foveal_luminance(frame, gaze_x, gaze_y, radius_px=50):
    if pd.isna(gaze_x) or pd.isna(gaze_y) or frame is None: return np.nan
    h, w = frame.shape[:2]; gaze_x, gaze_y = int(gaze_x), int(gaze_y)
    if not (0 <= gaze_x < w and 0 <= gaze_y < h): return np.nan
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (gaze_x, gaze_y), radius_px, 255, -1)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.mean(gray_frame, mask=mask)[0]

def build_pupil_baseline_model(calibration_data):
    print("--- Building Pupil Baseline Model from Calibration Data ---")
    if not calibration_data:
        print("Warning: Calibration data is empty. Cannot build baseline model.")
        return None
    
    df_calib = pd.DataFrame(calibration_data, columns=['luminance', 'pupil_radius'])
    df_calib = df_calib[df_calib['pupil_radius'] > 0].dropna()

    if len(df_calib) < 10:
        print("Warning: Not enough valid calibration data points.")
        return None

    model = LinearRegression()
    X = df_calib[['luminance']]
    y = df_calib['pupil_radius']
    model.fit(X, y)
    
    print("Pupil baseline model built successfully.")
    return model

def analyze_residual_z_scores(df_gaze, video_path, baseline_model):
    print("--- Starting Residual Z-Score Analysis ---")

    if baseline_model is None:
        print("Error: Baseline model is not available. Using simple PZS instead.")
        # フォールバックとして、動画全体の平均で単純なPZSを計算
        valid_pupil = df_gaze[df_gaze['pupil_radius'] > 0]['pupil_radius']
        if not valid_pupil.empty:
            mean_pupil = valid_pupil.mean()
            std_pupil = valid_pupil.std()
            df_gaze['pzs'] = (df_gaze['pupil_radius'] - mean_pupil) / std_pupil if std_pupil > 0 else 0
        else:
            df_gaze['pzs'] = 0
        df_gaze['pzs'].fillna(0, inplace=True)
        return df_gaze

    video_cap = cv2.VideoCapture(video_path)
    if not video_cap.isOpened():
        print("Error: Could not open video file."); return None
    
    total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    df_result = pd.DataFrame(index=range(total_frames))
    df_gaze = df_gaze.drop_duplicates(subset=['frame_index'], keep='last').set_index('frame_index')
    df_result = df_result.join(df_gaze, how='left')
    
    df_result[['gaze_x', 'gaze_y']] = df_result[['gaze_x', 'gaze_y']].interpolate(method='linear', limit_direction='both')
    
    print("Step 1: Calculating foveal luminance for all gaze points...")
    luminance_values = []
    for frame_idx in tqdm(range(total_frames), desc="Luminance Calc"):
        video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = video_cap.read()
        if not ret: luminance_values.append(np.nan); continue
        
        row = df_result.loc[frame_idx]
        lum = get_foveal_luminance(frame, row['gaze_x'], row['gaze_y'])
        luminance_values.append(lum)
    
    df_result['luminance'] = luminance_values
    df_result['luminance'].interpolate(method='linear', limit_direction='both', inplace=True)
    video_cap.release(); gc.collect()

    print("Step 2: Predicting baseline pupil size...")
    df_result['predicted_pupil'] = baseline_model.predict(df_result[['luminance']])

    print("Step 3: Calculating pupil response residual...")
    df_result['pupil_residual'] = df_result['pupil_radius'] - df_result['predicted_pupil']

    print("Step 4: Standardizing residual to PZS...")
    valid_residuals = df_result[df_result['pupil_radius'] > 0]['pupil_residual'].dropna()
    mean_residual = valid_residuals.mean()
    std_residual = valid_residuals.std()
    
    df_result['pzs'] = np.nan
    if std_residual > 0:
        valid_indices = df_result['pupil_radius'] > 0
        df_result.loc[valid_indices, 'pzs'] = (df_result.loc[valid_indices, 'pupil_residual'] - mean_residual) / std_residual
    
    df_result['pzs'].fillna(0, inplace=True)
    print("Residual Z-Score analysis complete.")
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

def main():
    parser = argparse.ArgumentParser(description="Verisight Video Analysis Tool (Server)")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--calib_csv", required=True)
    args = parser.parse_args()

    base_dir = "analysis_outputs"
    os.makedirs(base_dir, exist_ok=True)
    
    try:
        df_gaze = pd.read_csv(args.csv)
        calibration_data = pd.read_csv(args.calib_csv).values.tolist()
    except Exception as e: print(f"Error loading data CSVs: {e}"); return

    baseline_model = build_pupil_baseline_model(calibration_data)
    df_analyzed = analyze_residual_z_scores(df_gaze, args.video, baseline_model)
    
    if df_analyzed is not None:
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = os.path.basename(args.video).rsplit('.', 1)[0]
        output_filename = f"{base_filename}_analyzed_residual_{timestamp_str}.mp4"
        output_path = os.path.join(base_dir, output_filename)
        create_analyzed_video(args.video, df_analyzed, output_path)
    
    print("--- Video analysis script finished. ---")

if __name__ == '__main__':
    main()