import cv2
from ultralytics import YOLO

model = YOLO("best.pt")

video_path = "video.mp4"

cap = cv2.VideoCapture(video_path)

detection_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf=0.5) 
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            detection_count += 1

            x1, y1, x2, y2 = box.xyxy[0]
            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0), 2
            )
            label = f"cortante {conf:.2f}"
            cv2.putText(
                frame,
                label,
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    cv2.imshow("Deteccoes", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"Total de detecções de objetos cortantes: {detection_count}")