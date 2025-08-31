# app.py (v5.1.0 - 最終完成版：自己フォルダ作成機能)
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

# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# ★★★ ここが最後の、そして最も重要な修正点 ★★★
# サーバー起動時に、出力先フォルダが存在するか確認し、なければ作成する
try:
    os.makedirs(OUTPUTS_PATH, exist_ok=True)
    print(f"✅ Output directory '{OUTPUTS_PATH}' is ready.")
except Exception as e:
    print(f"❌ CRITICAL ERROR: Could not create output directory '{OUTPUTS_PATH}': {e}")
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★


def run_analysis_in_background(command, gaze_data, temp_csv_path):
    print(f"--- Starting background task for {temp_csv_path} ---")
    try:
        print(f"   - Saving {len(gaze_data)} points to {os.path.basename(temp_csv_path)}...")
        pd.DataFrame(gaze_data, columns=["timestamp", "frame_index", "pupil_radius", "gaze_x", "gaze_y"]).to_csv(temp_csv_path, index=False, encoding='utf-8-sig')
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
        if os.path.exists(temp_csv_path):
            os.remove(temp_csv_path)
            print(f"Temporary file {os.path.basename(temp_csv_path)} deleted.")

@app.route('/upload', methods=['POST'])
def upload_data():
    if not request.is_json: return jsonify({"error": "Request must be JSON"}), 400
    data = request.get_json(); gaze_data = data.get('gaze_data', []); item_name = data.get('stimuli_folder_name'); is_video = data.get('is_video', False); analysis_level = data.get('analysis_level', 'standard')
    if not gaze_data or not item_name: return jsonify({"error": "Missing data"}), 400

    print(f"✅ Received request for '{item_name}'. Dispatching to background.")
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    # 一時ファイルも出力フォルダに作成
    temp_csv_path = os.path.join(OUTPUTS_PATH, f"temp_data_{timestamp}.csv") 
        
    if is_video:
        analyzer_script = ANALYZER_VIDEO_PATH; item_path = os.path.join(VIDEOS_PATH, item_name)
        if not os.path.exists(item_path): return jsonify({"error": f"Video file '{item_name}' not found on server"}), 404
        command = ["python", analyzer_script, "--csv", temp_csv_path, "--video", item_path, "--level", analysis_level]
    else:
        analyzer_script = ANALYZER_STATIC_PATH; item_path = os.path.join(STIMULI_PATH, item_name)
        if not os.path.isdir(item_path): return jsonify({"error": f"Stimuli folder '{item_name}' not found on server"}), 404
        command = ["python", analyzer_script, "--csv", temp_csv_path, "--stimuli", item_path]

    analysis_thread = threading.Thread(target=run_analysis_in_background, args=(command, gaze_data, temp_csv_path))
    analysis_thread.start()
    
    return jsonify({"status": "success", "message": "Data received. All processing will run in the background."}), 202

@app.route('/download/<path:filename>')
def download_file(filename):
    print(f"⬇️ Download request received for: {filename}")
    return send_from_directory(OUTPUTS_PATH, filename, as_attachment=True)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port, debug=True)