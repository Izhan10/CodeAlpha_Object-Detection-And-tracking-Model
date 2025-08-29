from ultralytics import YOLO
import cv2

model = YOLO("custom_yolo11.pt")

cap = cv2.VideoCapture("vid5.mp4")
if not cap.isOpened():
    raise ValueError("Error: Could not open video file.")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter("output_video_2.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model.predict(source=frame, show=True, conf=0.7, line_width=2, save_crop=False, 
                           save_txt=False, show_labels=True, show_conf=True, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8])
    annotated_frame = results[0].plot() 
    out.write(annotated_frame)
    cv2.imshow("Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()