
import cv2
from ultralytics import YOLO

# -----------------------------
# SETTINGS
# -----------------------------
VIDEO_PATH = "input.mp4"
MODEL_PATH = "yolov8n.pt"

# ROI coordinates
ROI_X1 = 70
ROI_Y1 = 70
ROI_X2 = 1045
ROI_Y2 = 550

# -----------------------------
# LOAD MODEL
# -----------------------------
model = YOLO(MODEL_PATH)

# Video
cap = cv2.VideoCapture(VIDEO_PATH)

# Tracking data
previous_positions = {}
entered_ids = set()
exited_ids = set()

entry_count = 0
exit_count = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # --------------------------------
    # YOLO TRACKING
    # --------------------------------
    results = model.track(
        frame,
        persist=True,
        classes=[0],       # class 0 = person
        conf=0.30,
        verbose=False
    )

    active_in_roi = 0

    # --------------------------------
    # PROCESS DETECTIONS
    # --------------------------------
    if results[0].boxes.id is not None:

        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        confidences = results[0].boxes.conf.cpu().numpy()

        for box, track_id, confidence in zip(
            boxes, ids, confidences
        ):

            x1, y1, x2, y2 = map(int, box)

            # Center of person
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # --------------------------------
            # CHECK ROI
            # --------------------------------
            inside_roi = (
                ROI_X1 <= cx <= ROI_X2
                and
                ROI_Y1 <= cy <= ROI_Y2
            )

            # --------------------------------
            # COUNT ACTIVE PEOPLE
            # --------------------------------
            if inside_roi:
                active_in_roi += 1

            # --------------------------------
            # ENTRY / EXIT DETECTION
            # --------------------------------

            if track_id in previous_positions:

                old_x, old_y = previous_positions[track_id]

                # Person crosses ROI top boundary
                if old_y < ROI_Y1 and cy >= ROI_Y1:

                    if track_id not in entered_ids:
                        entry_count += 1
                        entered_ids.add(track_id)

                # Person crosses ROI bottom boundary
                if old_y > ROI_Y2 and cy <= ROI_Y2:

                    if track_id not in exited_ids:
                        exit_count += 1
                        exited_ids.add(track_id)

            previous_positions[track_id] = (cx, cy)

            # --------------------------------
            # DRAW PERSON BOX
            # --------------------------------
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Label
            label = f"Person #{track_id} {confidence:.2f}"

            cv2.putText(
                frame,
                label,
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1
            )

            # Center point
            cv2.circle(
                frame,
                (cx, cy),
                5,
                (0, 0, 255),
                -1
            )

    # --------------------------------
    # DRAW ROI
    # --------------------------------
    cv2.rectangle(
        frame,
        (ROI_X1, ROI_Y1),
        (ROI_X2, ROI_Y2),
        (0, 255, 255),
        3
    )

    # --------------------------------
    # INFORMATION PANEL
    # --------------------------------
    cv2.rectangle(
        frame,
        (0, 0),
        (365, 105),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        "INTELLIGENT SECURITY MONITOR",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Active in ROI: {active_in_roi}",
        (15, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Entries: {entry_count} | Exits: {exit_count}",
        (15, 86),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    # --------------------------------
    # DISPLAY
    # --------------------------------
    cv2.imshow(
        "Intelligent Security Monitor",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# --------------------------------
# RELEASE
# --------------------------------
cap.release()
cv2.destroyAllWindows()
