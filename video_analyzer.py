# video_analyzer.py (v2.0.0 - 高度分析最終版)
# 新機能：
# 1. 「視野中心の輝度」と「順応プロセス」を考慮した、高度なPZS計算アルゴリズムを実装。
#    - 視線周辺の輝度を計算し、物理的な光刺激をより正確に定量化。
#    - 直前の輝度変化から「明順応」「暗順応」「定常」状態を判定。
#    - 輝度レベルと順応状態でデータをグループ化し、各グループ内でPZSを計算。
# 2. 上記の処理に必要なヘルパー関数（輝度計算、順応判定など）を追加。
# 3. OpenCVによるビデオフレームへの高速アクセスのため、フレームキャッシュ機能を追加。

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

# --- フレームキャッシュ ---
# 同じフレームを何度も読み込まないようにするためのシンプルなキャッシュ
FRAME_CACHE = {}
LAST_FRAME_IDX = -1
LAST_FRAME_IMG = None

def get_frame_from_video(video_cap, frame_index):
    """OpenCVのVideoCaptureオブジェクトから、指定されたフレームを効率的に取得する"""
    global LAST_FRAME_IDX, LAST_FRAME_IMG
    
    if frame_index in FRAME_CACHE:
        return FRAME_CACHE[frame_index]
    
    # 連続したフレームアクセスの場合は、set()を使わずにread()だけ行うことで高速化
    if frame_index != LAST_FRAME_IDX + 1:
        video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        
    ret, frame = video_cap.read()
    LAST_FRAME_IDX = frame_index
    
    if ret:
        LAST_FRAME_IMG = frame
        if len(FRAME_CACHE) < 100: # キャッシュサイズの制限
            FRAME_CACHE[frame_index] = frame
        return frame
    else:
        # 読み込みに失敗した場合は、最後に成功したフレームを返す
        return LAST_FRAME_IMG

def get_foveal_luminance(frame, gaze_x, gaze_y, radius_px=50):
    """指定された視点中心の円形領域の平均輝度を計算する"""
    if pd.isna(gaze_x) or pd.isna(gaze_y) or frame is None:
        return np.nan
    
    h, w = frame.shape[:2]
    gaze_x, gaze_y = int(gaze_x), int(gaze_y)
    
    # 視点がフレーム外の場合は計算しない
    if not (0 <= gaze_x < w and 0 <= gaze_y < h):
        return np.nan
        
    # 円形ROIのマスクを作成
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (gaze_x, gaze_y), radius_px, 255, -1)
    
    # 輝度（グレースケール）画像に変換
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # マスクされた領域の平均輝度を計算
    mean_luminance = cv2.mean(gray_frame, mask=mask)[0]
    return mean_luminance

def determine_adaptation_state(luminance_series, look_back_frames=15):
    """輝度の時系列データから、各点の順応状態を判定する"""
    diff = luminance_series.diff(periods=look_back_frames)
    
    state = pd.Series('steady', index=luminance_series.index)
    state[diff > 10] = 'brightening' # 輝度が10以上増加したら明順応
    state[diff < -10] = 'darkening'   # 輝度が10以上減少したら暗順応
    
    return state

def analyze_advanced_z_scores(df, video_path):
    """【新機能】輝度と順応プロセスを考慮したPZSを計算する"""
    print("--- Starting Advanced Z-Score Analysis ---")
    
    video_cap = cv2.VideoCapture(video_path)
    if not video_cap.isOpened():
        print("Error: Could not open video file.")
        df['pzs'] = 0
        return df

    # --- ステップ1: 各視点での中心窩輝度を計算 ---
    print("Step 1: Calculating foveal luminance for each gaze point...")
    tqdm.pandas(desc="Luminance Calculation")
    df['luminance'] = df.progress_apply(
        lambda row: get_foveal_luminance(
            get_frame_from_video(video_cap, int(row.name)), # indexがフレーム番号
            row['gaze_x'],
            row['gaze_y']
        ), axis=1
    )
    FRAME_CACHE.clear() # キャッシュをクリア
    
    # --- ステップ2: 順応状態を判定 ---
    print("Step 2: Determining adaptation state...")
    df['adaptation'] = determine_adaptation_state(df['luminance'])

    # --- ステップ3: 輝度レベルと順応状態でグループ化し、PZSを計算 ---
    print("Step 3: Calculating PZS within luminance and adaptation groups...")
    df['pzs'] = np.nan
    
    # 輝度を25.6ごとの10段階にビン化（0-25.5 -> 0, 25.6-51.1 -> 1, ...）
    df['lum_bin'] = pd.cut(df['luminance'], bins=np.arange(0, 257, 25.6), labels=False, right=False)

    # 有効な瞳孔データのみを対象
    valid_pupil_df = df[df['pupil_radius'] > 0].copy()

    for (lum_bin, adaptation), group in tqdm(valid_pupil_df.groupby(['lum_bin', 'adaptation']), desc="PZS Group Calculation"):
        if len(group) < 10: continue # データが少なすぎるグループは無視

        mean_pupil = group['pupil_radius'].mean()
        std_pupil = group['pupil_radius'].std()
        
        if std_pupil > 0:
            pzs_scores = (group['pupil_radius'] - mean_pupil) / std_pupil
            df.loc[group.index, 'pzs'] = pzs_scores

    video_cap.release()
    df['pzs'].fillna(0, inplace=True)
    print("Advanced Z-Score calculation complete.")
    return df

def create_analyzed_video(video_path, df, output_path):
    # (変更なし)
    print("--- Starting Analyzed Video Generation ---")
    try:
        video_clip = VideoFileClip(video_path)
        width, height = video_clip.size
        
        matplotlib.rc('font', family='sans-serif')
        fig, ax = plt.subplots(figsize=(width / 100, 2.5), dpi=100)
        graph_h = int(fig.get_figheight() * fig.dpi)
        new_height = height + graph_h

        pzs_pos = df['pzs'].copy(); pzs_pos[df['pzs'] <= 0] = np.nan
        pzs_neg = df['pzs'].copy(); pzs_neg[df['pzs'] > 0] = np.nan
        pzs_max = max(df['pzs'].quantile(0.98), 1.5) if not df['pzs'].empty else 1.5
        pzs_min = min(df['pzs'].quantile(0.02), -1.5) if not df['pzs'].empty else -1.5
        
        ax.bar(df.index, pzs_pos, color='#FF4500', width=1.0)
        ax.bar(df.index, pzs_neg, color='#1E90FF', width=1.0)
        
        ax.axhline(y=0.0, color='white', linestyle='--', linewidth=1)
        ax.set_xlim(0, len(df)); ax.set_ylim(pzs_min, pzs_max)
        ax.set_title('Pupil Z-Score (PZS) - Advanced', color='white', fontsize=16)
        ax.tick_params(colors='white'); ax.grid(True, linestyle='--', alpha=0.2)
        fig.patch.set_facecolor('black'); ax.patch.set_facecolor('#181818')
        fig.tight_layout()
        
        fig.canvas.draw()
        graph_background_bgr = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
        graph_background_resized = cv2.resize(graph_background_bgr, (width, graph_h))
        plt.close(fig)

        def make_frame(t):
            frame_index = int(t * video_clip.fps)
            frame_bgr = cv2.cvtColor(video_clip.get_frame(t), cv2.COLOR_RGB2BGR)
            
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

        final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac', threads=4, logger=None)
        print(f"Analyzed video saved to {output_path}")

    except Exception as e:
        print(f"Error during video generation: {e}")
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Verisight Video Analysis Tool (Server)")
    parser.add_argument("--csv", required=True, help="Path to the gaze data CSV file.")
    parser.add_argument("--video", required=True, help="Path to the source video file.")
    args = parser.parse_args()

    base_dir = "analysis_results"
    os.makedirs(base_dir, exist_ok=True)
    
    try:
        df_gaze = pd.read_csv(args.csv)
        video_cap = cv2.VideoCapture(args.video)
        total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_cap.release()
        
        df_full = pd.DataFrame(index=range(total_frames))
        df_gaze = df_gaze.drop_duplicates(subset=['frame_index'], keep='last').set_index('frame_index')
        df_merged = df_full.join(df_gaze, how='left')
        df_merged['gaze_x'] = df_merged['gaze_x'].interpolate(method='linear', limit_direction='both', limit=15)
        df_merged['gaze_y'] = df_merged['gaze_y'].interpolate(method='linear', limit_direction='both', limit=15)

    except Exception as e:
        print(f"Error during data loading: {e}"); return

    # ★★★ ここで新しい分析関数を呼び出す ★★★
    df_analyzed = analyze_advanced_z_scores(df_merged, args.video)
    
    if df_analyzed is not None:
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = os.path.basename(args.video).rsplit('.', 1)[0]
        output_filename = f"{base_filename}_analyzed_{timestamp_str}.mp4"
        output_path = os.path.join(base_dir, output_filename)
        
        create_analyzed_video(args.video, df_analyzed, output_path)
    
    print("--- Video analysis script finished. ---")

if __name__ == '__main__':
    main()