import gradio as gr
import cv2
from ultralytics import YOLO
import os
import numpy as np
import tempfile

# Load your YOLOv8 model
model = YOLO("models/best.pt")

# Image processing function
def process_image(img):
    results = model(img)[0]
    annotated = results.plot()
    return annotated

# Video processing function
def process_video(video_file):
    temp_dir = tempfile.mkdtemp()
    cap = cv2.VideoCapture(video_file)
    
    output_path = os.path.join(temp_dir, "output.mp4")
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

    return output_path

# Gradio UI
def detect_file(file):
    if file.endswith((".jpg", ".jpeg", ".png")):
        img = cv2.imread(file)
        return process_image(img)
    elif file.endswith((".mp4", ".avi", ".mov")):
        return process_video(file)
    else:
        return "Unsupported file format"

demo = gr.Interface(
    fn=detect_file,
    inputs=gr.File(label="Upload Image or Video", file_types=["image", "video"]),
    outputs=gr.Image(type="numpy") | gr.Video(),
    title="🛡️ PPE Detection using YOLOv8",
    description="Upload an image or video to detect safety helmets or PPE equipment."
)

if __name__ == "__main__":
    demo.launch()
