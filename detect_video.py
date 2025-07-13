from ultralytics import YOLO
import cv2
from datetime import datetime
import threading
import time

model = YOLO("models/best.pt")

# Global state
LIVE_FEED_ACTIVE = False
camera_released = True
DETECTION_LOG = []  # Stores latest 20 logs
detection_thread = None
cap = None

def iou_overlap(boxA, boxB, threshold: float = 0.2) -> bool:
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter_area = max(0, xB - xA) * max(0, yB - yA)
    if inter_area == 0:
        return False
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    iou = inter_area / float(boxA_area)
    return iou > threshold
def detection_worker():
    """Separate thread for detection processing and logging"""
    global LIVE_FEED_ACTIVE, DETECTION_LOG, cap
    
    while LIVE_FEED_ACTIVE:
        if cap is not None and cap.isOpened():
            success, frame = cap.read()
            if success:
                try:
                    # Suppress verbose output from YOLO model
                    CONF_THRESH = 0.75
                    results = model(frame, conf=CONF_THRESH, verbose=False)[0]
                 
                    # Extract detection info
                    names = model.names
                    boxes = results.boxes.xyxy.cpu().numpy()
                    classes = results.boxes.cls.cpu().numpy().astype(int)

                    person_boxes = [boxes[i] for i, c in enumerate(classes) if names[c] == "Person"]
                    helmet_boxes = [boxes[i] for i, c in enumerate(classes) if names[c] == "Hardhat"]
                    mask_boxes = [boxes[i] for i, c in enumerate(classes) if names[c] == "Mask"]

                    # Build the message string
                    message = f"{datetime.now().strftime('%H:%M:%S')} - "
                    
                    if person_boxes:
                        message += f"👷 {len(person_boxes)} Person(s) detected | "
                        for idx, pbox in enumerate(person_boxes):
                            has_helmet = any(iou_overlap(pbox, hbox) for hbox in helmet_boxes)
                            has_mask = any(iou_overlap(pbox, mbox) for mbox in mask_boxes)

                            message += f"👤P{idx+1}: "
                            message += "✅ Helmet, " if has_helmet else "❌ No Helmet, "
                            message += "✅ Mask" if has_mask else "❌ No Mask"
                            if idx < len(person_boxes) - 1:
                                message += " | "
                    else:
                        message += "❌ No person detected"

                    # Create log entry
                    log_entry = {
                        "id": str(datetime.now().timestamp()),
                        "timestamp": datetime.now().strftime('%H:%M:%S'),
                        "message": message,
                        "category": "violation" if "❌" in message else "normal"
                    }
                    
                    DETECTION_LOG.append(log_entry)
                    if len(DETECTION_LOG) > 20:
                        DETECTION_LOG.pop(0)
                    
                    # Print to terminal (optional)
                    print(message) 
                    
                except Exception as e:
                    print(f"Detection error: {e}")
        
        time.sleep(1)

def generate_live_feed():
    """Generates video frames for streaming"""
    global LIVE_FEED_ACTIVE, camera_released, cap
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return

    camera_released = False
    
    while LIVE_FEED_ACTIVE:
        success, frame = cap.read()
        if not success:
            break

        try:
            # Suppress verbose output from YOLO model for the live feed as well
            results = model(frame, verbose=False)[0] 
            annotated = results.plot()

            # Stream frame
            ret, buffer = cv2.imencode('.jpg', annotated)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            print(f"Frame processing error: {e}")
            break

    cap.release()
    cv2.destroyAllWindows()
    camera_released = True
    print("🛑 Camera feed closed.")

def start_live():
    global LIVE_FEED_ACTIVE, detection_thread, DETECTION_LOG
    if not LIVE_FEED_ACTIVE:
        LIVE_FEED_ACTIVE = True
        DETECTION_LOG = []  # Reset log when starting
        
        # Start detection thread
        detection_thread = threading.Thread(target=detection_worker, daemon=True)
        detection_thread.start()
        
        print("🚀 Live detection started")

def stop_live():
    global LIVE_FEED_ACTIVE, detection_thread
    LIVE_FEED_ACTIVE = False
    
    if detection_thread is not None:
        detection_thread.join(timeout=2)
    
    print("🛑 Live detection stopped")