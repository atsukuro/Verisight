# analyzer.py (v2.0.0 - サーバー最終版)
# 修正：サーバー環境で不要かつエラーの原因となるeasyguiのimportを削除。

import pandas as pd
import cv2
import numpy as np
import os
import glob
import time
import matplotlib
matplotlib.use('Agg') # サーバー環境用の設定
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import argparse

# --- 日本語フォント設定 ---
def set_japanese_font():
    font_path = 'NotoSansJP-Regular.ttf'
    if os.path.exists(font_path):
        return FontProperties(fname=font_path)
    else:
        print(f"警告: 日本語フォントファイル '{font_path}' が見つかりません。")
        return None

jp_font = set_japanese_font()

def generate_heatmap(image, gaze_points, kernel_size=151, sigma=30):
    h, w = image.shape[:2]; heatmap = np.zeros((h, w), dtype=np.float32)
    for x, y in gaze_points:
        if 0 <= x < w and 0 <= y < h:
            x_min, x_max = max(0, x - kernel_size//2), min(w, x + kernel_size//2)
            y_min, y_max = max(0, y - kernel_size//2), min(h, y + kernel_size//2)
            if x_min >= x_max or y_min >= y_max: continue
            roi_w, roi_h = x_max - x_min, y_max - y_min; center_x, center_y = x - x_min, y - y_min
            Y, X = np.ogrid[:roi_h, :roi_w]; dist_sq = (X - center_x)**2 + (Y - center_y)**2
            gauss = np.exp(-dist_sq / (2 * sigma**2)); heatmap[y_min:y_max, x_min:x_max] += gauss
    if np.max(heatmap) > 0: heatmap = (heatmap / np.max(heatmap) * 255).astype(np.uint8)
    else: heatmap = heatmap.astype(np.uint8)
    colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(image, 0.6, colored_heatmap, 0.4, 0)

def get_color_from_pzs(pzs):
    if pzs <= 0.2: return (255, 255, 255);
    if pzs >= 1.5: return (0, 0, 255)
    ratio = (pzs - 0.2) / (1.3)
    yellow = np.array([0, 255, 255]); red = np.array([0, 0, 255])
    color = yellow * (1 - ratio) + red * ratio
    return tuple(map(int, color))

def generate_gaze_plot(image, slide_df):
    plot_image = image.copy(); prev_point = None
    # 負荷軽減のため、プロットする点の間隔を調整
    for i, (_, row) in enumerate(slide_df.iloc[::15, :].iterrows()):
        center = (int(row['gaze_x']), int(row['gaze_y']))
        if 0 <= center[0] < image.shape[1] and 0 <= center[1] < image.shape[0]:
            if prev_point: cv2.line(plot_image, prev_point, center, (255, 255, 255), 1, cv2.LINE_AA)
            color = get_color_from_pzs(row['pzs'])
            if i == 0:
                cv2.circle(plot_image, center, 15, (0, 255, 0), -1)
                cv2.putText(plot_image, "1", (center[0]-7, center[1]+7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
            else:
                cv2.circle(plot_image, center, 8, color, -1); cv2.circle(plot_image, center, 8, (0,0,0), 1)
            prev_point = center
    return plot_image

def main():
    parser = argparse.ArgumentParser(description="Verisight Static Image Analysis Tool (Server)")
    parser.add_argument("--csv", required=True, help="Path to the gaze data CSV file.")
    parser.add_argument("--stimuli", required=True, help="Path to the stimuli image folder.")
    args = parser.parse_args()
    
    print("--- Verisight Static Analyzer v2.0.0 ---")
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_folder = f"analysis_results_{timestamp}"
    os.makedirs(output_folder, exist_ok=True)

    try:
        df = pd.read_csv(args.csv)
        df.columns = ["timestamp", "slide_number", "pupil_radius", "gaze_x", "gaze_y"] # 列名を再設定
    except Exception as e:
        print(f"CSV read failed: {e}"); return

    # PZS計算
    if 'pupil_radius' in df.columns and not df['pupil_radius'].isnull().all():
        mean_pupil = df['pupil_radius'].mean()
        std_pupil = df['pupil_radius'].std()
        df['pzs'] = (df['pupil_radius'] - mean_pupil) / std_pupil if std_pupil > 0 else 0
    else:
        df['pzs'] = 0

    image_files = sorted(glob.glob(os.path.join(args.stimuli, '*.jpg')) + glob.glob(os.path.join(args.stimuli, '*.png')))
    
    for slide_num in df['slide_number'].unique():
        slide_df = df[df['slide_number'] == slide_num]
        if slide_df.empty or int(slide_num-1) >= len(image_files):
            continue
        
        print(f"Analyzing slide {slide_num}...")
        image_path = image_files[int(slide_num-1)]
        image = cv2.imdecode(np.fromfile(image_path, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            print(f"  - Failed to load image: {image_path}"); continue
            
        gaze_points = list(zip(slide_df['gaze_x'].astype(int), slide_df['gaze_y'].astype(int)))

        # ヒートマップ生成
        heatmap_img = generate_heatmap(image.copy(), gaze_points)
        cv2.imwrite(os.path.join(output_folder, f"heatmap_slide_{slide_num}.jpg"), heatmap_img)
        
        # ゲイズプロット生成
        gazeplot_img = generate_gaze_plot(image.copy(), slide_df)
        cv2.imwrite(os.path.join(output_folder, f"gazeplot_slide_{slide_num}.jpg"), gazeplot_img)

    print("--- Analysis complete. ---")

if __name__ == '__main__':
    main()