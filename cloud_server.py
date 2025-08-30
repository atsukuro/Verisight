# cloud_server.py (v3.1.1 - 最終完成版：ヘルスチェック対応)
# 修正点：
# 1. Gunicornから起動されることを前提とし、`if __name__ == '__main__':` ブロックを修正。
# 2. これにより、起動スクリプト(start.sh)と連携し、自己診断機能が完了してから
#    Webサーバーが正式に起動するため、Renderのヘルスチェックによるタイムアウトを防ぐ。

import os
import sys
import time
import subprocess
import threading
import traceback

# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# --- サーバー起動前 自己診断 ---
# サーバーが起動する前に、バックグラウンド処理に必要なライブラリが
# 全て利用可能かを確認します。ここで失敗すれば、デプロイログに直接原因が表示されます。
try:
    print("✅ [Self-check] Starting library import check...")
    import pandas
    import cv2
    import numpy
    from moviepy.editor import VideoFileClip
    import matplotlib
    print("✅ [Self-check] All required libraries are successfully imported.")
except ImportError as e:
    print(f"❌ [Self-check] CRITICAL ERROR: A required library is missing.")
    print(f"❌ [Self-check] Missing module: {e.name}")
    print(f"❌ [Self-check] Please ensure the library is correctly listed in requirements.txt.")
    # エラーを発生させてデプロイを意図的に失敗させる
    sys.exit(1)
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★


from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__)
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

STIMULI_PATH = os.path.join(BASE_PATH, "stimuli")
VIDEOS_PATH = os.path.join(BASE_PATH, "videos")
ANALYZER_STATIC_PATH = os.path.join(BASE_PATH, "analyzer.py")
ANALYZER_VIDEO_PATH = os.path.join(BASE_PATH, "video_analyzer.py")
OUTPUTS_PATH = os.path.join(BASE_PATH, "analysis_outputs")

def run_analysis_in_background(command, gaze_data, temp_csv_path):
    # (変更なし)
    print(f"--- Starting background task for {temp_csv_path} ---")
    try:
        print("   - [Self-healing] Verifying critical libraries before analysis...")
        required_libs = ["moviepy", "imageio", "imageio-ffmpeg"]
        pip_command = [sys.executable, "-m", "pip", "install"] + required_libs
        result = subprocess.run(pip_command, capture_output=True, text=True, encoding='utf-8')
        if result.returncode != 0:
            print("   - [Self-healing] ERROR: Failed to install/verify libraries.")
            print(result.stderr)
            raise RuntimeError("Failed to prepare analysis environment.")
        else:
            print("   - [Self-healing] All critical libraries are ready.")

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
    # (変更なし)
    if not request.is_json: return jsonify({"error": "Request must be JSON"}), 400
    data = request.get_json(); gaze_data = data.get('gaze_data', []); item_name = data.get('stimuli_folder_name'); is_video = data.get('is_video', False); analysis_level = data.get('analysis_level', 'standard')
    if not gaze_data or not item_name: return jsonify({"error": "Missing data"}), 400

    print(f"✅ Received request for '{item_name}'. Dispatching to background.")
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    temp_csv_path = os.path.join(BASE_PATH, f"temp_data_{timestamp}.csv")
        
    if is_video:
        analyzer_script = ANALYZER_VIDEO_PATH; item_path = os.path.join(VIDEOS_PATH, item_name)
        if not os.path.exists(item_path): return jsonify({"error": f"Video file '{item_name}' not found on server"}), 404
        command = [sys.executable, analyzer_script, "--csv", temp_csv_path, "--video", item_path, "--level", analysis_level]
    else:
        analyzer_script = ANALYZER_STATIC_PATH; item_path = os.path.join(STIMULI_PATH, item_name)
        if not os.path.isdir(item_path): return jsonify({"error": f"Stimuli folder '{item_name}' not found on server"}), 404
        command = [sys.executable, analyzer_script, "--csv", temp_csv_path, "--stimuli", item_path]

    analysis_thread = threading.Thread(target=run_analysis_in_background, args=(command, gaze_data, temp_csv_path))
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

# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# ★★★ ここが最後の修正点 ★★★
if __name__ == '__main__':
    # このスクリプトが直接実行された場合（ローカルでのテストなど）のフォールバック。
    # Render (Gunicorn) から起動される場合は、このブロックは通常は実行されない。
    # フォルダ作成は、ファイルが存在しない場合のみ試行する安全な方法に。
    print("cloud_server.py is run directly. This is intended for local testing.")
    print("On Render, this should be started via Gunicorn in start.sh.")
    
    for path in [STIMULI_PATH, VIDEOS_PATH, OUTPUTS_PATH]:
        if not os.path.isdir(path):
            try:
                os.makedirs(path)
                print(f"Directory {path} created on-demand.")
            except Exception as e:
                print(f"Warning: Could not create directory {path}: {e}")
    
    # ローカルテスト用にapp.runを呼び出す
    app.run(host='0.0.0.0', port=10000, debug=True)
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★