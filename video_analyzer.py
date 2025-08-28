# video_analyzer.py (v1.0.0 - サーバー版)
import os
import sys
import cv2
import numpy as np
import datetime
import pandas as pd
import matplotlib
matplotlib.use('Agg') # 画面がないサーバー環境用の設定
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip, AudioFileClip
from tqdm import tqdm
import argparse
import traceback

# 日本語フォントの設定（サーバーにフォントがない場合を考慮）
def set_japanese_font():
    try:
        font_path = 'NotoSansJP-Regular.ttf'
        if os.path.exists(font_path):
            matplotlib.rc('font', family='Noto Sans JP')
            print("Font 'Noto Sans JP' set.")
        else:
            matplotlib.rc('font', family='sans-serif')
            print("Warning: Japanese font not found. Using sans-serif.")
    except Exception as e:
        print(f"Font setting error: {e}")

def analyze_and_get_dataframe(video_path, gaze_csv_path):
    print("--- Starting Data Analysis ---")
    try:
        df = pd.read_csv(gaze_csv_path)
        if df.empty:
            print("Error: Gaze data CSV is empty.")
            return None
        
        video_cap = cv2.VideoCapture(video_path)
        video_fps = video_cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_cap.release()

        # PZS (Pupil Z-Score) を計算
        if 'pupil_radius' in df.columns and not df['pupil_radius'].isnull().all():
            mean_pupil = df['pupil_radius'].mean()
            std_pupil = df['pupil_radius'].std()
            if std_pupil > 0:
                df['pzs'] = (df['pupil_radius'] - mean_pupil) / std_pupil
            else:
                df['pzs'] = 0
        else:
            df['pzs'] = 0
        
        print("PZS calculation complete.")
        return df

    except Exception as e:
        print(f"Error during data analysis: {e}")
        traceback.print_exc()
        return None

def create_analyzed_video(video_path, df, output_path):
    print("--- Starting Analyzed Video Generation ---")
    try:
        video_clip = VideoFileClip(video_path)
        width, height = video_clip.size
        total_frames_video = int(video_clip.duration * video_clip.fps)
        
        set_japanese_font()
        fig, ax = plt.subplots(figsize=(width / 100, 2.5), dpi=100)
        graph_h = int(fig.get_figheight() * fig.dpi)
        new_height = height + graph_h

        # PZSグラフの事前描画
        pzs_pos = df['pzs'].copy(); pzs_pos[df['pzs'] <= 0] = np.nan
        pzs_neg = df['pzs'].copy(); pzs_neg[df['pzs'] > 0] = np.nan
        pzs_max = max(df['pzs'].quantile(0.98, interpolation='higher'), 1.5)
        pzs_min = min(df['pzs'].quantile(0.02, interpolation='lower'), -1.5)
        
        ax.bar(df.index, pzs_pos, color='#FF4500', width=1.0) # オレンジレッド
        ax.bar(df.index, pzs_neg, color='#1E90FF', width=1.0) # ドジャーブルー
        
        ax.axhline(y=0.0, color='white', linestyle='--', linewidth=1)
        ax.set_xlim(0, len(df)); ax.set_ylim(pzs_min, pzs_max)
        ax.set_title('PZS (Pupil Z-Score)', color='white', fontsize=16)
        ax.tick_params(colors='white'); ax.grid(True, linestyle='--', alpha=0.2)
        fig.patch.set_facecolor('black'); ax.patch.set_facecolor('#181818')
        fig.tight_layout()
        
        fig.canvas.draw()
        graph_background_bgr = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
        graph_background_resized = cv2.resize(graph_background_bgr, (width, graph_h))
        plt.close(fig)

        def make_frame(t):
            frame_index = int(t * video_clip.fps)
            frame = video_clip.get_frame(t) # RGB
            
            # グラフ部分を作成
            canvas = np.zeros((new_height, width, 3), dtype=np.uint8)
            canvas[0:height, 0:width] = frame
            
            current_graph_img = graph_background_resized.copy()
            marker_x_pos = int((frame_index / len(df)) * width)
            cv2.line(current_graph_img, (marker_x_pos, 0), (marker_x_pos, graph_h), (255, 255, 0), 2)
            canvas[height:new_height, :] = current_graph_img

            # 視線カーソルを描画
            if frame_index < len(df):
                row = df.iloc[frame_index]
                gaze_x, gaze_y, pzs = row.get('gaze_x', -1), row.get('gaze_y', -1), row.get('pzs', 0)
                if pd.notna(gaze_x) and pd.notna(gaze_y) and gaze_x > 0:
                    color = (255, 70, 0) if pzs > 0 else (255, 255, 255)
                    cv2.circle(canvas, (int(gaze_x), int(gaze_y)), 20, color, 3)
            
            return canvas

        # MoviePyで動画を生成
        analyzed_clip = video_clip.fl_image(make_frame)
        
        # 音声がある場合は合成
        if video_clip.audio:
            final_clip = analyzed_clip.set_audio(video_clip.audio)
        else:
            final_clip = analyzed_clip

        # GPUが使えるかもしれないRender環境を考慮
        try:
            final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac', threads=4, logger=None)
        except Exception: # フォールバック
             final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac', logger=None)

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
    
    df = analyze_and_get_dataframe(args.video, args.csv)
    
    if df is not None:
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = os.path.basename(args.video).rsplit('.', 1)[0]
        output_filename = f"{base_filename}_analyzed_{timestamp_str}.mp4"
        output_path = os.path.join(base_dir, output_filename)
        
        create_analyzed_video(args.video, df, output_path)
    
    print("--- Video analysis script finished. ---")

if __name__ == '__main__':
    main()