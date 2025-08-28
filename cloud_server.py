# cloud_server.py (v2.0.0 - 動画分析対応版)
from flask import Flask, request, jsonify
import pandas as pd
import os
import time
import subprocess
import sys

app = Flask(__name__)
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# 各種パスを定義
STIMULI_PATH = os.path.join(BASE_PATH, "stimuli")
VIDEOS_PATH = os.path.join(BASE_PATH, "videos")
ANALYZER_STATIC_PATH = os.path.join(BASE_PATH, "analyzer.py")
ANALYZER_VIDEO_PATH = os.path.join(BASE_PATH, "video_analyzer.py")

@app.route('/upload', methods=['POST'])
def upload_data():
    if not request.is_json: return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    gaze_data = data.get('gaze_data', [])
    item_name = data.get('stimuli_folder_name') # キー名はクライアントと合わせる
    is_video = data.get('is_video', False)

    if not gaze_data or not item_name: return jsonify({"error": "Missing data"}), 400

    print(f"✅ Received {len(gaze_data)} data points for '{item_name}'. Video: {is_video}")
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    
    # 一時的にCSVファイルとして保存
    temp_csv_path = os.path.join(BASE_PATH, f"temp_data_{timestamp}.csv")
    try:
        pd.DataFrame(gaze_data, columns=["timestamp", "frame_index", "pupil_radius", "gaze_x", "gaze_y"]).to_csv(temp_csv_path, index=False, encoding='utf-8-sig')
    except Exception as e:
        print(f"   - ERROR saving CSV: {e}"); return jsonify({"error": "Failed to save data on server"}), 500
        
    # is_videoフラグに応じて、呼び出すアナライザーと刺激物のパスを切り替える
    if is_video:
        analyzer_script = ANALYZER_VIDEO_PATH
        item_path = os.path.join(VIDEOS_PATH, item_name)
        if not os.path.exists(item_path):
             print(f"   - ERROR: Video file not found: {item_path}")
             return jsonify({"error": f"Video file '{item_name}' not found on server"}), 404
        command = [sys.executable, analyzer_script, "--csv", temp_csv_path, "--video", item_path]
    else:
        analyzer_script = ANALYZER_STATIC_PATH
        item_path = os.path.join(STIMULI_PATH, item_name)
        if not os.path.isdir(item_path):
            print(f"   - ERROR: Stimuli folder not found: {item_path}")
            return jsonify({"error": f"Stimuli folder '{item_name}' not found on server"}), 404
        command = [sys.executable, analyzer_script, "--csv", temp_csv_path, "--stimuli", item_path]

    print(f"   - Triggering analysis with '{os.path.basename(analyzer_script)}'...")
    try:
        result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', check=False)
        
        print("\n--- analyzer output ---")
        print(result.stdout)
        print("--- end of output ---\n")

        if result.returncode != 0:
            print(f"   - ERROR: Analysis script failed with exit code {result.returncode}")
            print("\n--- analyzer error ---"); print(result.stderr); print("--- end of error ---\n")
            if os.path.exists(temp_csv_path): os.remove(temp_csv_path)
            return jsonify({"error": "Analysis script failed", "details": result.stderr}), 500
        
        print("   - Analysis process completed successfully.")
        if os.path.exists(temp_csv_path): os.remove(temp_csv_path)
        return jsonify({"status": "success", "message": "Analysis completed."}), 200
    except Exception as e:
        print(f"   - ERROR: Failed to execute analysis script: {e}")
        if os.path.exists(temp_csv_path): os.remove(temp_csv_path)
        return jsonify({"error": "Failed to start analysis process"}), 500

if __name__ == '__main__':
    # 起動時に必要なフォルダが存在するか確認
    for path in [STIMULI_PATH, VIDEOS_PATH]:
        if not os.path.isdir(path): os.makedirs(path)
    # Gunicornがサーバーを起動するため、この部分はローカルテスト用
    app.run(host='0.0.0.0', port=10000, debug=False)