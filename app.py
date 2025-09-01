# app.py (v5.3.0 - 最終完成版：出撃前自己修復機能)
# 新機能：
# 1. バックグラウンド処理を開始する直前に、司令塔自身が必要なライブラリの存在を確認し、
#    なければ強制的にインストールする「出撃前点検・自己修復機能」を実装。
# 2. これにより、Cloud Runの不可解な環境不整合問題を完全に解決する。

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
    print(f"--- Starting background task for {os.path.basename(temp_gaze_csv_path)} ---")
    try:
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★★★ ここが最後の、そして最も重要な自己修復機能 ★★★
        print("   - [Pre-flight Check] Verifying critical libraries before analysis...")
        try:
            # 最も問題が発生しやすいmoviepyを直接インポートしてみる
            from moviepy.editor import VideoFileClip
            print("   - [Pre-flight Check] All critical libraries seem to be available.")
        except ImportError:
            print("   - ⚠️ [Pre-flight Check] A required library is missing. Attempting to self-heal...")
            required_libs = ["moviepy", "imageio-ffmpeg", "tqdm"]
            pip_command = [sys.executable, "-m", "pip", "install"] + required_libs
            result = subprocess.run(pip_command, capture_output=True, text=True, encoding='utf-8')
            if result.returncode != 0:
                print("   - ❌ [Self-healing] ERROR: Failed to install/verify libraries.")
                print(result.stderr)
                raise RuntimeError("Failed to prepare analysis environment.")
            else:
                print("   - ✅ [Self-healing] All critical libraries are now ready.")
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

        print(f"   - Saving {len(gaze_data)} gaze points...")
        pd.DataFrame(gaze_data, columns=["timestamp", "frame_index", "pupil_radius", "gaze_x", "gaze_y"]).to_csv(temp_gaze_csv_path, index=False, encoding='utf-8-sig')
        
        if calibration_data:
            print(f"   - Saving {len(calibration_data)} calibration points...")
            pd.DataFrame(calibration_data, columns=["luminance", "pupil_radius"]).to_csv(temp_calib_csv_path, index=False, encoding='utf-8-sig')
        
        print("   - CSV save complete.")

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
        for f_path in [temp_gaze_csv_path, temp_calib_csv_path]:
            if f_path and os.path.exists(f_path):
                os.remove(f_path)
                print(f"Temporary file {os.path.basename(f_path)} deleted.")

@app.route('/upload', methods=['POST'])
def upload_data():
    if not request.is_json: return jsonify({"error": "Request must be JSON"}), 400
    data = request.get_json(); gaze_data = data.get('gaze_data', []); calibration_data = data.get('calibration_data', []); item_name = data.get('item_name'); is_video = data.get('is_video', False)
    if not gaze_data or not item_name: return jsonify({"error": "Missing data"}), 400

    print(f"✅ Received request for '{item_name}'. Dispatching to background.")
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    
    temp_gaze_csv_path = os.path.join(OUTPUTS_PATH, f"temp_gaze_{timestamp}.csv")
    temp_calib_csv_path = os.path.join(OUTPUTS_PATH, f"temp_calib_{timestamp}.csv")
        
    if is_video:
        analyzer_script = ANALYZER_VIDEO_PATH; item_path = os.path.join(VIDEOS_PATH, item_name)
        if not os.path.exists(item_path): return jsonify({"error": f"Video file '{item_name}' not found on server"}), 404
        command = ["python", analyzer_script, "--csv", temp_gaze_csv_path, "--video", item_path, "--calib_csv", temp_calib_csv_path]
    else:
        analyzer_script = ANALYZER_STATIC_PATH; item_path = os.path.join(STIMULI_PATH, item_name)
        if not os.path.isdir(item_path): return jsonify({"error": f"Stimuli folder '{item_name}' not found on server"}), 404
        command = ["python", analyzer_script, "--csv", temp_gaze_csv_path, "--stimuli", item_path]
        temp_calib_csv_path = None

    analysis_thread = threading.Thread(target=run_analysis_in_background, args=(command, gaze_data, calibration_data, temp_gaze_csv_path, temp_calib_csv_path))
    analysis_thread.start()
    
    return jsonify({"status": "success", "message": "Data received. All processing will run in the background."}), 202

@app.route('/download/<path:filename>')
def download_file(filename):
    print(f"⬇️ Download request received for: {filename}")
    try:
        return send_from_directory(OUTPUTS_PATH, filename, as_attachment=True)
    except FileNotFoundError:
        return jsonify({"error": "File not found"}), 404

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port, debug=True)```