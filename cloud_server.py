# cloud_server.py (v2.2.0 - 最終完成版：究極の非同期処理)
# 修正点：
# 1. CSVへの書き出し処理もバックグラウンドで行うように変更。
# 2. メインスレッドはデータを受け取ったら即座に応答を返すだけに特化させ、タイムアウトを完全に撲滅。

from flask import Flask, request, jsonify
import pandas as pd
import os
import time
import subprocess
import sys
import threading

app = Flask(__name__)
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# 各種パス定義
STIMULI_PATH = os.path.join(BASE_PATH, "stimuli")
VIDEOS_PATH = os.path.join(BASE_PATH, "videos")
ANALYZER_STATIC_PATH = os.path.join(BASE_PATH, "analyzer.py")
ANALYZER_VIDEO_PATH = os.path.join(BASE_PATH, "video_analyzer.py")

def run_analysis_in_background(command, gaze_data, temp_csv_path):
    """【新】バックグラウンドでCSV保存から分析まで全て行う関数"""
    print(f"--- Starting background task for {temp_csv_path} ---")
    try:
        # ステップ1：バックグラウンドでCSVファイルに書き出す
        print(f"   - Saving {len(gaze_data)} points to {os.path.basename(temp_csv_path)}...")
        pd.DataFrame(gaze_data, columns=["timestamp", "frame_index", "pupil_radius", "gaze_x", "gaze_y"]).to_csv(temp_csv_path, index=False, encoding='utf-8-sig')
        print("   - CSV save complete.")

        # ステップ2：分析スクリプトを実行する
        print(f"   - Executing analysis: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', check=False)
        
        print("\n--- Background Analyzer STDOUT ---")
        print(result.stdout)
        print("--- END STDOUT ---\n")
        
        if result.returncode != 0:
            print(f"--- ERROR: Background analysis failed with exit code {result.returncode} ---")
            print("\n--- Background Analyzer STDERR ---")
            print(result.stderr)
            print("--- END STDERR ---\n")
        else:
            print("--- Background analysis completed successfully. ---")
            
    except Exception as e:
        print(f"--- CRITICAL: An error occurred in the background task: {e} ---")
        traceback.print_exc()
    finally:
        # 処理が終わったら一時ファイルを削除
        if os.path.exists(temp_csv_path):
            os.remove(temp_csv_path)
            print(f"Temporary file {os.path.basename(temp_csv_path)} deleted.")

@app.route('/upload', methods=['POST'])
def upload_data():
    if not request.is_json: return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    gaze_data = data.get('gaze_data', [])
    item_name = data.get('stimuli_folder_name')
    is_video = data.get('is_video', False)

    if not gaze_data or not item_name: return jsonify({"error": "Missing data"}), 400

    print(f"✅ Received request for '{item_name}'. Dispatching to background.")
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    temp_csv_path = os.path.join(BASE_PATH, f"temp_data_{timestamp}.csv")
        
    if is_video:
        analyzer_script = ANALYZER_VIDEO_PATH
        item_path = os.path.join(VIDEOS_PATH, item_name)
        if not os.path.exists(item_path):
             return jsonify({"error": f"Video file '{item_name}' not found on server"}), 404
        command = [sys.executable, analyzer_script, "--csv", temp_csv_path, "--video", item_path]
    else:
        analyzer_script = ANALYZER_STATIC_PATH
        item_path = os.path.join(STIMULI_PATH, item_name)
        if not os.path.isdir(item_path):
            return jsonify({"error": f"Stimuli folder '{item_name}' not found on server"}), 404
        command = [sys.executable, analyzer_script, "--csv", temp_csv_path, "--stimuli", item_path]

    # ★★★ CSV書き出しも含め、すべての重い処理をバックグラウンドに渡す ★★★
    analysis_thread = threading.Thread(target=run_analysis_in_background, args=(command, gaze_data, temp_csv_path))
    analysis_thread.start()
    
    # 即座に応答を返す
    return jsonify({"status": "success", "message": "Data received. All processing will run in the background."}), 202

if __name__ == '__main__':
    for path in [STIMULI_PATH, VIDEOS_PATH]:
        if not os.path.isdir(path): os.makedirs(path)
    app.run(host='0.0.0.0', port=10000, debug=False)