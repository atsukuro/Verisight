# app.py (v5.2.0 - 統合キャリブレーション対応・最終完成版)
# 新機能：
# 1. クライアントから送信される 'calibration_data' を受信する機能を追加。
# 2. 受信したキャリブレーションデータを一時的なCSVファイルとして保存。
# 3. video_analyzer.py を呼び出す際に、--calib_csv 引数を使ってそのCSVファイルのパスを渡す。

from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import os
import time
import subprocess
import sys
import threading
import traceback

app = Flask(__name__)
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
STIMULI_PATH = os.path.join(BASE_PATH, "stimuli")
VIDEOS_PATH = os.path.join(BASE_PATH, "videos")
ANALYZER_STATIC_PATH = os.path.join(BASE_PATH, "analyzer_static.py")
ANALYZER_VIDEO_PATH = os.path.join(BASE_PATH, "analyzer_video.py")
OUTPUTS_PATH = os.path.join(BASE_PATH, "analysis_outputs")

try:
    os.makedirs(OUTPUTS_PATH, exist_ok=True)
    print(f"✅ Output directory '{OUTPUTS_PATH}' is ready.")
except Exception as e:
    print(f"❌ CRITICAL ERROR: Could not create output directory '{OUTPUTS_PATH}': {e}")


def run_analysis_in_background(command, gaze_data, calibration_data, temp_gaze_csv_path, temp_calib_csv_path):
    """【新】バックグラウンドでCSV保存から分析まで全て行う関数"""
    print(f"--- Starting background task for {os.path.basename(temp_gaze_csv_path)} ---")
    try:
        # ステップ1：バックグラウンドで両方のCSVファイルを書き出す
        print(f"   - Saving {len(gaze_data)} gaze points...")
        pd.DataFrame(gaze_data, columns=["timestamp", "frame_index", "pupil_radius", "gaze_x", "gaze_y"]).to_csv(temp_gaze_csv_path, index=False, encoding='utf-8-sig')
        
        # calibration_dataが存在する場合のみ保存
        if calibration_data:
            print(f"   - Saving {len(calibration_data)} calibration points...")
            pd.DataFrame(calibration_data, columns=["luminance", "pupil_radius"]).to_csv(temp_calib_csv_path, index=False, encoding='utf-8-sig')
        
        print("   - CSV save complete.")

        # ステップ2：分析スクリプトを実行する
        print(f"   - Executing analysis: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', check=False)
        
        print("\n--- Background Analyzer STDOUT ---"); print(result.stdout); print("--- END STDOUT ---\n")
        if result.returncode != 0:
            print(f"--- ERROR: Background analysis failed with exit code {result.returncode} ---")
            print("\n--- Background Analyzer STDERR ---"); print(result.stderr); print("--- END STDERR ---\n")
        else:
            print("--- Background analysis completed successfully. ---")
            
    except Exception as e:
        print(f"--- CRITICAL: An error occurred in the background task: {e}"); traceback.print_exc()
    finally:
        # 処理が終わったら両方の一時ファイルを削除
        for f_path in [temp_gaze_csv_path, temp_calib_csv_path]:
            if f_path and os.path.exists(f_path):
                os.remove(f_path)
                print(f"Temporary file {os.path.basename(f_path)} deleted.")

@app.route('/upload', methods=['POST'])
def upload_data():
    if not request.is_json: return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    gaze_data = data.get('gaze_data', [])
    # ★★★ 新しいデータを受信 ★★★
    calibration_data = data.get('calibration_data', [])
    item_name = data.get('item_name')
    is_video = data.get('is_video', False)

    if not gaze_data or not item_name: return jsonify({"error": "Missing data"}), 400

    print(f"✅ Received request for '{item_name}'. Dispatching to background.")
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    
    # ★★★ 一時ファイル名を2種類用意 ★★★
    temp_gaze_csv_path = os.path.join(OUTPUTS_PATH, f"temp_gaze_{timestamp}.csv")
    temp_calib_csv_path = os.path.join(OUTPUTS_PATH, f"temp_calib_{timestamp}.csv")
        
    if is_video:
        analyzer_script = ANALYZER_VIDEO_PATH; item_path = os.path.join(VIDEOS_PATH, item_name)
        if not os.path.exists(item_path): return jsonify({"error": f"Video file '{item_name}' not found on server"}), 404
        # ★★★ 新しいコマンドを組み立て ★★★
        command = ["python", analyzer_script, "--csv", temp_gaze_csv_path, "--video", item_path, "--calib_csv", temp_calib_csv_path]
    else: # 静止画の場合は、今のところキャリブレーションデータは使わない
        analyzer_script = ANALYZER_STATIC_PATH; item_path = os.path.join(STIMULI_PATH, item_name)
        if not os.path.isdir(item_path): return jsonify({"error": f"Stimuli folder '{item_name}' not found on server"}), 404
        command = ["python", analyzer_script, "--csv", temp_gaze_csv_path, "--stimuli", item_path]
        temp_calib_csv_path = None # 静止画では使わないのでNoneに

    # ★★★ 新しいデータをバックグラウンド関数に渡す ★★★
    analysis_thread = threading.Thread(target=run_analysis_in_background, args=(command, gaze_data, calibration_data, temp_gaze_csv_path, temp_calib_csv_path))
    analysis_thread.start()
    
    return jsonify({"status": "success", "message": "Data received. All processing will run in the background."}), 202

@app.route('/download/<path:filename>')
def download_file(filename):
    # (変更なし)
    print(f"⬇️ Download request received for: {filename}")
    try:
        return send_from_directory(OUTPUTS_PATH, filename, as_attachment=True)
    except FileNotFoundError:
        return jsonify({"error": "File not found"}), 404

if __name__ == '__main__':
    # (変更なし)
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port, debug=True)