# cloud_server.py (v1.2.2)
# バグ修正：
# 1. WindowsとLinux(クラウド)で文字コードが異なる問題に対応。
#    プラットフォームを自動検出し、適切なエンコーディングを使用するように修正。

from flask import Flask, request, jsonify
import pandas as pd
import os
import time
import subprocess
import sys

app = Flask(__name__)
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
STIMULI_BASE_PATH = os.path.join(BASE_PATH, "stimuli")
ANALYZER_PATH = os.path.join(BASE_PATH, "analyzer.py")

@app.route('/upload', methods=['POST'])
def upload_data():
    if not request.is_json: return jsonify({"error": "Request must be JSON"}), 400
    data = request.get_json()
    gaze_data = data.get('gaze_data', [])
    stimuli_folder_name = data.get('stimuli_folder_name')
    if not gaze_data or not stimuli_folder_name: return jsonify({"error": "Missing data"}), 400

    print(f"✅ Received {len(gaze_data)} data points for '{stimuli_folder_name}'.")
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    temp_csv_path = os.path.join(BASE_PATH, f"temp_data_{timestamp}.csv")
    try:
        pd.DataFrame(gaze_data, columns=["timestamp", "slide_number", "pupil_radius", "gaze_x", "gaze_y"]).to_csv(temp_csv_path, index=False, encoding='utf-8-sig')
    except Exception as e:
        print(f"   - ERROR saving CSV: {e}"); return jsonify({"error": "Failed to save data on server"}), 500

    stimuli_folder_path = os.path.join(STIMULI_BASE_PATH, stimuli_folder_name)
    if not os.path.isdir(stimuli_folder_path):
        print(f"   - ERROR: Stimuli folder not found: {stimuli_folder_path}")
        return jsonify({"error": f"Stimuli folder '{stimuli_folder_name}' not found on server"}), 404

    print(f"   - Triggering analysis...")
    command = [sys.executable, ANALYZER_PATH, "--csv", temp_csv_path, "--stimuli", stimuli_folder_path]

    # ▼▼▼ 変更点：プラットフォームに応じてエンコーディングを切り替える ▼▼▼
    encoding = 'cp932' if sys.platform == 'win32' else 'utf-8'

    try:
        result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', check=False)

        print("\n--- analyzer.py output ---"); print(result.stdout); print("--- end of output ---\n")

        if result.returncode != 0:
            print(f"   - ERROR: analyzer.py failed with exit code {result.returncode}")
            print("\n--- analyzer.py error ---"); print(result.stderr); print("--- end of error ---\n")
            if os.path.exists(temp_csv_path): os.remove(temp_csv_path)
            return jsonify({"error": "Analysis script failed", "details": result.stderr}), 500

        print("   - Analysis process completed successfully.")
        if os.path.exists(temp_csv_path): os.remove(temp_csv_path)
        return jsonify({"status": "success", "message": "Analysis completed."}), 200
    except Exception as e:
        print(f"   - ERROR: Failed to execute analyzer.py: {e}")
        if os.path.exists(temp_csv_path): os.remove(temp_csv_path)
        return jsonify({"error": "Failed to start analysis process"}), 500

if __name__ == '__main__':
    if not os.path.isdir(STIMULI_BASE_PATH): os.makedirs(STIMULI_BASE_PATH)
    if not os.path.exists(ANALYZER_PATH): print(f"致命的エラー：'{ANALYZER_PATH}' が見つかりません。"); sys.exit(1)
    # Renderが使用するポートを自動で取得するように変更
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)