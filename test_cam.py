from ultralytics import YOLO
import cv2

def run_live_detection():
    model = YOLO("models/best.pt")
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)[0]
        annotated = results.plot()

        cv2.imshow("Live PPE Detection", annotated)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
