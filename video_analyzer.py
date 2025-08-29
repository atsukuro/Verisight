# video_analyzer.py (v2.2.1 - 動画用・最終完成版)
# 修正点：
# 1. 出力先を永続ディスクではない通常の 'analysis_outputs' フォルダに戻す。
# 2. 分析完了時に、ログにダウンロード用のファイル名を明確に出力するように変更。

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
FRAME_CACHE = {}
LAST_FRAME_IDX = -1
LAST_FRAME_IMG = None

def get_frame_from_video(video_cap, frame_index):
    global LAST_FRAME_IDX, LAST_FRAME_IMG
    if frame_index in FRAME_CACHE: return FRAME_CACHE[frame_index]
    if frame_index != LAST_FRAME_IDX + 1: video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = video_cap.read(); LAST_FRAME_IDX = frame_index
    if ret:
        LAST_FRAME_IMG = frame
        if len(FRAME_CACHE) < 100: FRAME_CACHE[frame_index] = frame
        return frame
    else: return LAST_FRAME_IMG

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

def analyze_advanced_z_scores(df, video_path):
    print("--- Starting Advanced Z-Score Analysis ---")
    video_cap = cv2.VideoCapture(video_path)
    if not video_cap.isOpened():
        print("Error: Could not open video file."); df['pzs'] = 0; return df
    print("Step 1: Calculating foveal luminance..."); tqdm.pandas(desc="Luminance Calc")
    df['luminance'] = df.progress_apply(lambda row: get_foveal_luminance(get_frame_from_video(video_cap, int(row.name)), row['gaze_x'], row['gaze_y']), axis=1)
    FRAME_CACHE.clear()
    print("Step 2: Determining adaptation state...")
    df['adaptation'] = determine_adaptation_state(df['luminance'])
    print("Step 3: Calculating PZS within groups...")
    df['pzs'] = np.nan
    df['lum_bin'] = pd.cut(df['luminance'], bins=np.arange(0, 257, 25.6), labels=False, right=False)
    valid_pupil_df = df[df['pupil_radius'] > 0].copy()
    for (lum_bin, adaptation), group in tqdm(valid_pupil_df.groupby(['lum_bin', 'adaptation']), desc="PZS Group Calc"):
        if len(group) < 10: continue
        mean_pupil = group['pupil_radius'].mean(); std_pupil = group['pupil_radius'].std()
        if std_pupil > 0:
            df.loc[group.index, 'pzs'] = (group['pupil_radius'] - mean_pupil) / std_pupil
    video_cap.release()
    df['pzs'].fillna(0, inplace=True)
    print("Advanced Z-Score calculation complete.")
    return df

def create_analyzed_video(video_path, df, output_path):
    print("--- Starting Analyzed Video Generation ---")
    try:
        video_clip = VideoFileClip(video