import os
import uuid
import cv2
from ultralytics import YOLO
from flask import Flask, send_file, render_template, Response, request, jsonify
from detect_video import (
    generate_live_feed,
    start_live,
    stop_live,
    DETECTION_LOG
)
from waitress import serve  # ✅ Import waitress for production server

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/start', methods=['POST'])
def start_detection():
    data = request.get_json(silent=True)
    ip_url = data.get('ip') if data else None

    if ip_url:
        print(f"🔗 Starting with IP stream: {ip_url}")
        start_live(ip_url)
    else:
        print("🎥 Starting with default webcam")
        start_live()  # No argument = default webcam

    return jsonify({'status': 'started'})



@app.route("/stop", methods=["POST"])
def stop_detection():
    stop_live()
    return jsonify({"status": "stopped"})

@app.route("/video_feed")
def video_feed():
    return Response(generate_live_feed(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/detection_log")
def detection_log():
    from detect_video import DETECTION_LOG
    # Return the logs in reverse order (newest first)
    return jsonify({"logs": list(reversed(DETECTION_LOG))})

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/upload", methods=["POST"])
def upload_file():
    from PIL import Image

    file = request.files["file"]
    filename = f"{uuid.uuid4().hex}_{file.filename}"
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    model = YOLO("models/best.pt")

    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        # Handle image
        img = cv2.imread(path)
        results = model(img)[0]
        result_img = results.plot()

        output_path = os.path.join(UPLOAD_FOLDER, f"result_{filename}")
        cv2.imwrite(output_path, result_img)

        return send_file(output_path, mimetype="image/jpeg")

    elif filename.lower().endswith((".mp4", ".avi", ".mov")):
        # Handle video
        cap = cv2.VideoCapture(path)
        output_path = os.path.join(UPLOAD_FOLDER, f"result_{filename}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)[0]
            annotated = results.plot()
            out.write(annotated)

        cap.release()
        out.release()

        return send_file(output_path, mimetype="video/mp4")

    else:
        return "Unsupported file type", 400

@app.route("/export_logs")
def export_logs():
    from detect_video import DETECTION_LOG
    import csv
    from io import StringIO
    import html
    
    # Create CSV data in memory with UTF-8 encoding
    csv_data = StringIO()
    writer = csv.writer(csv_data)
    
    # Write UTF-8 BOM for Excel compatibility
    csv_data.write('\ufeff')
    
    # Write header
    writer.writerow(["Timestamp", "Category", "Status", "Details"])
    
    # Write log data
    for log in reversed(DETECTION_LOG):
        # Clean and format the message
        clean_message = log['message'].replace('👷', 'Workers:')
        clean_message = log['message'].replace('👤', 'Person')
        clean_message = log['message'].replace('✅', 'Yes')
        clean_message = log['message'].replace('❌', 'No')
        
        # Split into status and details
        if "|" in clean_message:
            status, details = clean_message.split("|", 1)
        else:
            status = clean_message
            details = ""
            
        writer.writerow([
            log['timestamp'],
            log['category'].capitalize(),
            status.strip(),
            details.strip()
        ])
    
    # Create response with CSV data
    response = Response(
        csv_data.getvalue(),
        mimetype="text/csv; charset=utf-8-sig",
        headers={
            "Content-disposition": "attachment; filename=ppe_detection_logs.csv"
        }
    )
    
    return response

if __name__ == "__main__":
    print("🚀 Starting app with Waitress on http://localhost:8000")
    serve(app, host='0.0.0.0', port=8000)
