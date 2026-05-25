import cv2
import pandas as pd
import re

# load results
results = pd.read_csv("test.csv")

# open video
cap = cv2.VideoCapture("sample1.mp4")

# get FPS for proper playback speed
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps)


def parse_bbox(bbox_string):
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", str(bbox_string))
    numbers = [float(n) for n in numbers]
    return numbers[-4:]


while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame_nmr = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    frame_data = results[results["frame_nmr"] == frame_nmr]

    for _, row in frame_data.iterrows():

        car_bbox = parse_bbox(row["car_bbox"])
        plate_bbox = parse_bbox(row["license_plate_bbox"])

        x1, y1, x2, y2 = map(int, car_bbox)
        px1, py1, px2, py2 = map(int, plate_bbox)

        plate_text = str(row["license_plate_text"])

        # draw car box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # draw plate box
        cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 255), 3)

        # ---------- BLACK LABEL ABOVE PLATE ----------
        label_height = 35

        cv2.rectangle(
            frame,
            (px1, py1 - label_height),
            (px2, py1),
            (0, 0, 0),
            -1
        )

        # plate text in white
        cv2.putText(
            frame,
            plate_text,
            (px1 + 5, py1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

    cv2.imshow("ALPR Detection", frame)

    if cv2.waitKey(delay) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()