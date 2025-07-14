# 👷 AI CCTV PPE Detection System  
**Real-time Personal Protective Equipment (PPE) detection** using YOLO and Flask. Detects helmets, masks, and workers in live streams or uploaded images/videos.  

![Demo](https://via.placeholder.com/800x400.png?text=Demo+Preview) *(Replace with actual screenshot)*  

---

## 🚀 Features  
- **Live Detection**: Monitor RTSP/IP camera feeds for PPE compliance.  
- **Upload Analysis**: Process images/videos for PPE violations.  
- **Real-time Logs**: Track detections with timestamps and violation alerts.  
- **Export Data**: Download logs as CSV for reporting.  
- **Mobile-Friendly**: Works with IP camera apps (e.g., IP Webcam).  

---

## 🛠️ Technologies  
- **Backend**: Python (Flask, OpenCV, Ultralytics YOLO)  
- **Frontend**: HTML/CSS/JS  
- **Deployment**: Waitress (production-ready WSGI server)  

---

## 📦 Installation  

### 1. Clone the Repository  
```bash
git clone https://github.com/yourusername/ai-ppe-detection.git
cd ai-ppe-detection
```

### 2. Install Dependencies  
```bash
pip install -r requirements.txt
```

### 3. Download YOLO Model  
Place your `best.pt` model in the `models/` folder.  

---

## ⚙️ Configuration  
1. **Camera Setup**:  
   - Update `IP_CAMERA_URL` in `.env` or `detect_video.py`:  
     ```python
     IP_CAMERA_URL = "rtsp://username:password@ip:port/stream"  # For IP cameras
     # OR for mobile (IP Webcam):
     IP_CAMERA_URL = "http://your_phone_ip:8080/video"
     ```

2. **Environment Variables** (optional):  
   Create a `.env` file:  
   ```plaintext
   IP_CAMERA_URL=your_camera_url
   ```

---

## 🖥️ Usage  

### Run the Application  
```bash
python app.py
```
Access the web interface at: `http://localhost:8000`  

### Key Functionalities  
- **Live Monitoring**: Start/stop camera feed detection.  
- **Upload Files**: Process images/videos via the upload section.  
- **View Logs**: Real-time detection history with violation highlights.  

---

## 📸 Screenshots  
*(Add actual screenshots here)*  
| Live Detection | Upload Analysis | Log Export |  
|---------------|----------------|------------|  
| ![Live](https://via.placeholder.com/300x200.png?text=Live+Feed) | ![Upload](https://via.placeholder.com/300x200.png?text=Upload+Demo) | ![Logs](https://via.placeholder.com/300x200.png?text=Logs+Export) |  

---

## 📜 Log Format  
```json
{
  "timestamp": "14:30:45",
  "message": "👷 2 Person(s) detected | 👤P1: ✅ Helmet, ❌ No Mask",
  "category": "violation"
}
```

---

## 🤝 Contributing  
1. Fork the repository.  
2. Create a branch (`git checkout -b feature/your-feature`).  
3. Submit a PR.  

---

## 📄 License  
MIT © [Your Name]  

---

## 🔗 Links  
- **Model Training**: [YOLO Documentation](https://docs.ultralytics.com/)  
- **Mobile Camera Apps**: [IP Webcam](https://play.google.com/store/apps/details?id=com.pas.webcam)  

---
