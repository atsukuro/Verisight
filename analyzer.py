# analyzer.py (v1.9.2)
# バグ修正：
# 1. Windowsのコンソール(cp932)で文字化け・エラーの原因となる絵文字を
#    print文からすべて削除。

import pandas as pd
import cv2
import numpy as np
import easygui
import os
import glob
import time
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import argparse

# --- 日本語フォント設定 ---
FONT_PATH = 'NotoSansJP-Regular.ttf'
if os.path.exists(FONT_PATH):
    jp_font = FontProperties(fname=FONT_PATH)
else:
    print(f"警告: 日本語フォントファイル '{FONT_PATH}' が見つかりません。")
    jp_font = None

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

def generate_gaze_plot(image, slide_df, aoi_rects, aoi_metrics):
    plot_image = image.copy(); prev_point = None
    for i, (_, row) in enumerate(slide_df.iloc[::5, :].iterrows()):
        center = (int(row['gaze_x']), int(row['gaze_y']))
        if prev_point: cv2.line(plot_image, prev_point, center, (255, 255, 255), 1, cv2.LINE_AA)
        color = get_color_from_pzs(row['pzs'])
        if i == 0:
            cv2.circle(plot_image, center, 15, (0, 255, 0), -1)
            cv2.putText(plot_image, "1", (center[0]-7, center[1]+7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
        else:
            cv2.circle(plot_image, center, 8, color, -1); cv2.circle(plot_image, center, 8, (0,0,0), 1)
        prev_point = center
    for aoi_name, rect in aoi_rects.items():
        ttff = aoi_metrics.get(f'ttff_{aoi_name}', -1)
        if ttff >= 0:
            text = f"TTFF: {ttff:.2f}s"; pos = (rect[0] + 5, rect[1] + 20)
            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(plot_image, (pos[0]-5, pos[1]-h-5), (pos[0]+w+5, pos[1]+5), (0,0,0), -1)
            cv2.putText(plot_image, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    return plot_image

def analyze_aoi_metrics(slide_df, aoi_rects, fps=60):
    metrics = {}; slide_start_time = slide_df['timestamp'].min()
    for aoi_name, rect in aoi_rects.items():
        x, y, w, h = rect
        aoi_df = slide_df[(slide_df['gaze_x']>=x) & (slide_df['gaze_x']<x+w) & (slide_df['gaze_y']>=y) & (slide_df['gaze_y']<y+h)]
        if not aoi_df.empty:
            is_in_aoi = (slide_df['gaze_x']>=x)&(slide_df['gaze_x']<x+w)&(slide_df['gaze_y']>=y)&(slide_df['gaze_y']<y+h)
            revisit_count = (is_in_aoi.diff() & is_in_aoi).sum()
        else: revisit_count = 0
        metrics[f'dwell_time_{aoi_name}'] = len(aoi_df) / fps
        metrics[f'ttff_{aoi_name}'] = (aoi_df['timestamp'].min() - slide_start_time) if not aoi_df.empty else -1
        metrics[f'avg_pupil_{aoi_name}'] = aoi_df['pupil_radius'].mean() if not aoi_df.empty else 0
        metrics[f'revisit_count_{aoi_name}'] = revisit_count
    return metrics

def generate_decision_matrix_plot(matrix_df):
    fig, ax = plt.subplots(figsize=(12, 10)); median_x = matrix_df['x_engagement'].median()
    xlim = (min(matrix_df['x_engagement'].min()*1.1, -0.5), max(matrix_df['x_engagement'].max()*1.1, 0.5))
    ylim = (min(matrix_df['y_bias'].min()*1.1, -0.5), max(matrix_df['y_bias'].max()*1.1, 0.5))
    ax.fill_between([median_x, xlim[1]], ylim[1], color='red', alpha=0.1, zorder=1)
    ax.fill_between([xlim[0], median_x], ylim[1], color='green', alpha=0.1, zorder=1)
    ax.fill_between([xlim[0], median_x], ylim[0], color='blue', alpha=0.1, zorder=1)
    ax.fill_between([median_x, xlim[1]], ylim[0], color='purple', alpha=0.1, zorder=1)
    ax.scatter(matrix_df['x_engagement'], matrix_df['y_bias'], s=150, alpha=0.8, zorder=10, c='black')
    for i, row in matrix_df.iterrows():
        ax.text(row['x_engagement'], row['y_bias'] + (ylim[1]*0.02), f"Q{row['question_number']}", fontsize=14, ha='center', fontproperties=jp_font, zorder=11)
    ax.axhline(0, color='grey', linestyle='--'); ax.axvline(median_x, color='grey', linestyle='--')
    ax.set_title('ディシジョン・マトリクス', fontsize=20, fontproperties=jp_font, pad=20)
    ax.set_xlabel('質問への関心度', fontsize=14, fontproperties=jp_font, labelpad=15)
    ax.set_ylabel('ポジティブ回答へのバイアス', fontsize=14, fontproperties=jp_font, labelpad=15)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.grid(True, linestyle='--', alpha=0.5)
    ax.text(xlim[1], ylim[1], '情熱ゾーン', ha='right', va='top', fontsize=16, color='red', alpha=0.7, fontproperties=jp_font)
    ax.text(xlim[0], ylim[1], '直感ゾーン', ha='left', va='top', fontsize=16, color='green', alpha=0.7, fontproperties=jp_font)
    ax.text(xlim[0], ylim[0], '無関心ゾーン', ha='left', va='bottom', fontsize=16, color='blue', alpha=0.7, fontproperties=jp_font)
    ax.text(xlim[1], ylim[0], '葛藤ゾーン', ha='right', va='bottom', fontsize=16, color='purple', alpha=0.7, fontproperties=jp_font)
    ax.set_xlim(xlim); ax.set_ylim(ylim); fig.tight_layout()
    return fig

def generate_cognitive_load_graph(slide_df, slide_num):
    df = slide_df.copy()
    df['elapsed_time'] = df['timestamp'] - df['timestamp'].min()
    df['gaze_x_smooth'] = df['gaze_x'].rolling(window=5, center=True, min_periods=1).mean()
    df['gaze_y_smooth'] = df['gaze_y'].rolling(window=5, center=True, min_periods=1).mean()
    df['gaze_velocity'] = np.sqrt(df['gaze_x_smooth'].diff()**2 + df['gaze_y_smooth'].diff()**2).fillna(0)
    window_size = 15
    df['pzs_smooth'] = df['pzs'].rolling(window=window_size, center=True, min_periods=1).mean()
    df['velocity_smooth'] = df['gaze_velocity'].rolling(window=window_size, center=True, min_periods=1).mean()
    fig, ax1 = plt.subplots(figsize=(12, 6))
    color1 = 'tab:blue'
    ax1.set_xlabel('時間 (秒)', fontproperties=jp_font); ax1.set_ylabel('瞳孔反応 (PZS)', color=color1, fontproperties=jp_font)
    ax1.plot(df['elapsed_time'], df['pzs_smooth'], color=color1, label='瞳孔反応 (PZS)')
    ax1.tick_params(axis='y', labelcolor=color1); ax1.axhline(0, color='grey', linestyle='--', alpha=0.5)
    ax2 = ax1.twinx(); color2 = 'tab:red'
    ax2.set_ylabel('視線速度 (pixels/frame)', color=color2, fontproperties=jp_font)
    ax2.plot(df['elapsed_time'], df['velocity_smooth'], color=color2, label='視線速度')
    ax2.tick_params(axis='y', labelcolor=color2)
    high_load_condition = (df['pzs_smooth'] > df['pzs_smooth'].mean()) & (df['velocity_smooth'] < df['velocity_smooth'].mean())
    ax1.fill_between(df['elapsed_time'], ax1.get_ylim()[0], ax1.get_ylim()[1], where=high_load_condition, facecolor='purple', alpha=0.2, label='高負荷推定区間')
    fig.suptitle(f'スライド {slide_num} の認知的負荷', fontsize=16, fontproperties=jp_font)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    lines1, labels1 = ax1.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', prop=jp_font)
    return fig

def main():
    parser = argparse.ArgumentParser(description="Verisight Analysis Tool")
    parser.add_argument("--csv", required=True, help="Path to the gaze data CSV file.")
    parser.add_argument("--stimuli", required=True, help="Path to the stimuli image folder.")
    args = parser.parse_args()
    csv_path = args.csv; stimuli_folder = args.stimuli

    print(f"--- Verisight アナライザー v1.9.2 (Auto Run Mode) ---")
    timestamp = time.strftime('%Y%m%d_%H%M%S'); output_folder = f"analysis_results_{timestamp}"; os.makedirs(output_folder, exist_ok=True)
    try: df = pd.read_csv(csv_path)
    except Exception as e: print(f"CSV読込失敗: {e}"); return
    
    baseline_df = df[df['timestamp'] < 3.0]
    if not baseline_df.empty and baseline_df['pupil_radius'].std() > 0:
        baseline_mean = baseline_df['pupil_radius'].mean(); baseline_std = baseline_df['pupil_radius'].std()
        df['pzs'] = (df['pupil_radius'] - baseline_mean) / (baseline_std if baseline_std > 0 else 1.0)
    else: df['pzs'] = 0

    unique_slides = sorted(df['slide_number'].unique())
    image_files = sorted(glob.glob(os.path.join(stimuli_folder, '*.jpg')) + glob.glob(os.path.join(stimuli_folder, '*.png')))
    stimuli_folder_name = os.path.basename(stimuli_folder)
    all_aoi_results = []; matrix_points = []
    question_number = 1

    for i in range(0, len(unique_slides), 2):
        question_slide_num = unique_slides[i]
        if (i + 1) >= len(unique_slides): continue
        answer_slide_num = unique_slides[i+1]
        print(f"--- 質問 {question_number} (Slide {question_slide_num} & {answer_slide_num}) の分析開始 ---")
        try:
            q_image_path = image_files[question_slide_num - 1]; a_image_path = image_files[answer_slide_num - 1]
            q_slide_image = cv2.imdecode(np.fromfile(q_image_path, np.uint8), cv2.IMREAD_COLOR)
            answer_slide_image = cv2.imdecode(np.fromfile(a_image_path, np.uint8), cv2.IMREAD_COLOR)
            if q_slide_image is None or answer_slide_image is None: raise IOError("画像読込失敗")
        except Exception as e: print(f"画像読込失敗: {e}"); continue
        
        q_slide_df = df[(df['slide_number'] == question_slide_num) & (df['gaze_x'] > 0)]
        a_slide_df = df[(df['slide_number'] == answer_slide_num) & (df['gaze_x'] > 0)]

        for num, s_df, s_img in [(question_slide_num, q_slide_df, q_slide_image), (answer_slide_num, a_slide_df, answer_slide_image)]:
            if s_df.empty: continue
            
            h_s, w_s = s_img.shape[:2]
            aoi_rects_s = {"Question": (0, 0, w_s, h_s // 2), "Choice_A": (0, h_s // 2, w_s // 2, h_s // 2), "Choice_B": (w_s // 2, h_s // 2, w_s // 2, h_s // 2)}
            aoi_metrics_s = analyze_aoi_metrics(s_df, aoi_rects_s)
            
            if num == answer_slide_num: aoi_metrics_s['slide_number'] = num; all_aoi_results.append(aoi_metrics_s)
            
            gaze_points = list(zip(s_df['gaze_x'].astype(int), s_df['gaze_y'].astype(int)))
            cv2.imwrite(os.path.join(output_folder, f"{stimuli_folder_name}_heatmap_slide_{num}.jpg"), generate_heatmap(s_img.copy(), gaze_points))
            cv2.imwrite(os.path.join(output_folder, f"{stimuli_folder_name}_gazeplot_slide_{num}.jpg"), generate_gaze_plot(s_img.copy(), s_df, aoi_rects_s, aoi_metrics_s))
            
            load_fig = generate_cognitive_load_graph(s_df, num)
            load_fig.savefig(os.path.join(output_folder, f"{stimuli_folder_name}_cognitiveload_slide_{num}.png"), dpi=150); plt.close(load_fig)
        
        # ▼▼▼ 変更点：絵文字を削除 ▼▼▼
        print(f"スライド {question_slide_num} & {answer_slide_num} の分析・可視化が完了しました。")

        positive_pzs_q_df = q_slide_df[q_slide_df['pzs'] > 0]; positive_pzs_a_df = a_slide_df[a_slide_df['pzs'] > 0]
        h, w = answer_slide_image.shape[:2]
        x_engagement = positive_pzs_q_df[positive_pzs_q_df['gaze_y'] < h]['pzs'].sum()
        pzs_A = positive_pzs_a_df[(positive_pzs_a_df['gaze_y'] >= h // 2) & (positive_pzs_a_df['gaze_x'] < w // 2)]['pzs'].sum()
        pzs_B = positive_pzs_a_df[(positive_pzs_a_df['gaze_y'] >= h // 2) & (positive_pzs_a_df['gaze_x'] >= w // 2)]['pzs'].sum()
        y_bias = pzs_A - pzs_B
        matrix_points.append({'question_number': question_number, 'slide_number': question_slide_num, 'x_engagement': x_engagement, 'y_bias': y_bias})
        question_number += 1

    if matrix_points:
        matrix_df = pd.DataFrame(matrix_points)
        matrix_df.to_csv(os.path.join(output_folder, f"matrix_data_{timestamp}.csv"), index=False, encoding='utf-8-sig')
        fig = generate_decision_matrix_plot(matrix_df)
        fig.savefig(os.path.join(output_folder, f"decision_matrix_{timestamp}.png"), dpi=300); plt.close(fig)
        print(f"\nディシジョン・マトリクス関連ファイルを保存しました。")
        
    if all_aoi_results:
        summary_df = pd.DataFrame(all_aoi_results)
        summary_df = summary_df[['slide_number'] + [c for c in summary_df.columns if c != 'slide_number']]
        summary_df.to_csv(os.path.join(output_folder, f"aoi_summary_{timestamp}.csv"), index=False, encoding='utf-8-sig')
        print(f"AOI分析サマリーを保存しました。")

    print("\nすべての処理が完了しました。")

if __name__ == '__main__':
    main()