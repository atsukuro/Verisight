# video_analyzer.py (v1.1.0 - 高度分析準備版)
# 変更点：
# 1. 新しいZスコア分析ロジックを組み込むための準備として、全体の構造を整理。
# 2. MoviePyに加えてOpenCVも活用し、フレームごとの輝度計算を効率的に行えるように変更。
# 3. Zスコア計算部分を analyze_z_scores 関数として分離。

import os
import sys
import cv2
import numpy as np
import datetime
import pandas as pd
import matplotlib
matplotlib.use('Agg') # 画面がないサーバー環境用の設定
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from tqdm import tqdm
import argparse
import traceback

# 日本語フォントの設定（サーバーにフォントがない場合を考慮）
def set_japanese_font():
    # Renderサーバーには日本語フォントがないため、デフォルトのsans-serifを使用する
    matplotlib.rc('font', family='sans-serif')
    print("Font set to sans-serif for server environment.")

def analyze_z_scores(df, video_path):
    """
    視線データとビデオから瞳孔のZスコアを計算する。
    【今後、この関数を高度なロジックに置き換えていきます】
    """
    print("--- Starting Z-Score Analysis ---")
    
    # === 現在のシンプルなロジック（動画全体での標準化） ===
    if 'pupil_radius' in df.columns and not df['pupil_radius'].isnull().all():
        # データ欠損値(-1)を計算から除外する
        valid_pupil_data = df[df['pupil_radius'] > 0]['pupil_radius']
        
        if not valid_pupil_data.empty:
            mean_pupil = valid_pupil_data.mean()
            std_pupil = valid_pupil_data.std()
            
            # 元のデータフレームに対してPZSを計算（欠損値はnp.nanのまま）
            df['pzs'] = np.nan
            valid_indices = df['pupil_radius'] > 0
            if std_pupil > 0:
                df.loc[valid_indices, 'pzs'] = (df.loc[valid_indices, 'pupil_radius'] - mean_pupil) / std_pupil
            else:
                df.loc[valid_indices, 'pzs'] = 0
        else:
            df['pzs'] = 0 # 有効なデータが一つもなかった場合
    else:
        df['pzs'] = 0 # pupil_radius列が存在しない場合

    # 欠損していたPZSを0で埋める（グラフ表示のため）
    df['pzs'].fillna(0, inplace=True)
    
    print("Z-Score calculation complete (using simple method).")
    return df

def create_analyzed_video(video_path, df, output_path):
    print("--- Starting Analyzed Video Generation ---")
    try:
        video_clip = VideoFileClip(video_path)
        width, height = video_clip.size
        
        set_japanese_font()
        fig, ax = plt.subplots(figsize=(width / 100, 2.5), dpi=100)
        graph_h = int(fig.get_figheight() * fig.dpi)
        new_height = height + graph_h

        # PZSグラフの事前描画
        pzs_pos = df['pzs'].copy(); pzs_pos[df['pzs'] <= 0] = np.nan
        pzs_neg = df['pzs'].copy(); pzs_neg[df['pzs'] > 0] = np.nan
        pzs_max = max(df['pzs'].quantile(0.98), 1.5) if not df['pzs'].empty else 1.5
        pzs_min = min(df['pzs'].quantile(0.02), -1.5) if not df['pzs'].empty else -1.5
        
        ax.bar(df.index, pzs_pos, color='#FF4500', width=1.0)
        ax.bar(df.index, pzs_neg, color='#1E90FF', width=1.0)
        
        ax.axhline(y=0.0, color='white', linestyle='--', linewidth=1)
        ax.set_xlim(0, len(df)); ax.set_ylim(pzs_min, pzs_max)
        ax.set_title('Pupil Z-Score (PZS)', color='white', fontsize=16)
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
            
            # 視線カーソルを描画
            if frame_index < len(df):
                row = df.iloc[frame_index]
                gaze_x, gaze_y, pzs = row.get('gaze_x', -1), row.get('gaze_y', -1), row.get('pzs', 0)
                if pd.notna(gaze_x) and pd.notna(gaze_y) and gaze_x > 0:
                    color = (0, 70, 255) if pzs > 0 else (255, 255, 255) # BGR
                    cv2.circle(frame_bgr, (int(gaze_x), int(gaze_y)), 20, color, 3)
            
            canvas[0:height, 0:width] = frame_bgr
            
            current_graph_img = graph_background_resized.copy()
            marker_x_pos = int((frame_index / len(df)) * width)
            cv2.line(current_graph_img, (marker_x_pos, 0), (marker_x_pos, graph_h), (0, 255, 255), 2) # BGR: Yellow
            canvas[height:new_height, :] = current_graph_img
            
            return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB) # MoviePyはRGBを期待

        analyzed_clip = video_clip.fl_image(make_frame)
        if video_clip.audio:
            final_clip = analyzed_clip.set_audio(video_clip.audio)
        else:
            final_clip = analyzed_clip

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
        
        # 全フレームに対応する空のデータフレームを作成
        df_full = pd.DataFrame(index=range(total_frames))
        # 視線データをフレームインデックスをキーにして結合
        df_gaze = df_gaze.drop_duplicates(subset=['frame_index'], keep='last').set_index('frame_index')
        df_merged = df_full.join(df_gaze, how='left')

    except Exception as e:
        print(f"Error during data loading and merging: {e}")
        return

    # Zスコア分析を実行
    df_analyzed = analyze_z_scores(df_merged, args.video)
    
    if df_analyzed is not None:
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = os.path.basename(args.video).rsplit('.', 1)[0]
        output_filename = f"{base_filename}_analyzed_{timestamp_str}.mp4"
        output_path = os.path.join(base_dir, output_filename)
        
        create_analyzed_video(args.video, df_analyzed, output_path)
    
    print("--- Video analysis script finished. ---")

if __name__ == '__main__':
    main()