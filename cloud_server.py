# cloud_server.py (v2.1.0 - 最終完成版：非同期処理対応)
# 修正点：
# 1. 時間のかかる分析処理をバックグラウンドのスレッドで実行するように変更。
# 2. クライアントにはデータ受信後すぐに「受付完了」の応答を返し、タイムアウトを防ぐ。

from flask import Flask, request, jsonify
import pandas as pd
import os
import time
import subprocess
import sys
import threading

app = Flask(__name__)
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# 各種パスを定義
STIMULI_PATH = os.path.join(BASE_PATH, "stimuli")
VIDEOS_PATH = os.path.join(BASE_PATH, "videos")
ANALYZER_STATIC_PATH = os.path.join(BASE_PATH, "analyzer.py")
ANALYZER_VIDEO_PATH = os.path.join(BASE_PATH, "video_analyzer.py")

def run_analysis_in_background(command, temp_csv_path):
    """バックグラウンドで分析スクリプトを実行する関数"""
    print(f"--- Starting background analysis: {' '.join(command)} ---")
    try:
        # text=Trueとencoding='utf-8'はPython 3.7+で推奨
        result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', check=False)
        
        # 標準出力と標準エラーをログに記録
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
        print(f"--- CRITICAL: Failed to execute background analysis: {e} ---")
    finally:
        # 処理が終わったら一時ファイルを削除
        if os.path.exists(temp_csv_path):
            os.remove(temp_csv_path)
            print(f"Temporary file {temp_csv_path} deleted.")


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
        return jsonify({"error": f"Failed to save data on server: {e}"}), 500
        
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

    # ★★★ 修正点：分析をバックグラウンドのスレッドで実行 ★★★
    analysis_thread = threading.Thread(target=run_analysis_in_background, args=(command, temp_csv_path))
    analysis_thread.start()
    
    # すぐにクライアントに応答を返す
    print("   - Analysis task has been dispatched to the background.")
    return jsonify({"status": "success", "message": "Data received. Analysis is running in the background."}), 202


if __name__ == '__main__':
    # 起動時に必要なフォルダが存在するか確認
    for path in [STIMULI_PATH, VIDEOS_PATH]:
        if not os.path.isdir(path): os.makedirs(path)
    # Gunicornがサーバーを起動するため、この部分はローカルテスト用
    app.run(host='0.0.0.0', port=10000, debug=False)