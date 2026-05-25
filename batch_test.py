import os
import csv
import cv2
from ultralytics import YOLO
from util import read_license_plate

# load model
model = YOLO("best.pt")

input_folder = "test_images"
output_folder = "output_images"
debug_folder = "debug_plates"

os.makedirs(output_folder, exist_ok=True)
os.makedirs(debug_folder, exist_ok=True)

# clear old files
for file in os.listdir(output_folder):
    os.remove(os.path.join(output_folder, file))

for file in os.listdir(debug_folder):
    os.remove(os.path.join(debug_folder, file))

# CSV
csv_file = open(
    "batch_results.csv",
    "w",
    newline=""
)

writer = csv.writer(csv_file)

writer.writerow([
    "image_name",
    "license_plate_text",
    "confidence"
])

# process images
for image_name in os.listdir(input_folder):

    image_path = os.path.join(
        input_folder,
        image_name
    )

    frame = cv2.imread(image_path)

    if frame is None:
        continue

    results = model(frame)[0]

    found = False

    for idx, result in enumerate(
        results.boxes.data.tolist()
    ):

        x1, y1, x2, y2, score, class_id = result

        # larger padding
        padding = 20

        x1 = max(0, int(x1 - padding))
        y1 = max(0, int(y1 - padding))
        x2 = min(frame.shape[1], int(x2 + padding))
        y2 = min(frame.shape[0], int(y2 + padding))

        # crop
        plate_crop = frame[y1:y2, x1:x2]

        # save debug crop
        debug_path = os.path.join(
            debug_folder,
            f"{image_name}_{idx}.jpg"
        )

        cv2.imwrite(debug_path, plate_crop)

        # OCR
        plate_text, text_score = read_license_plate(
            plate_crop
        )

        if plate_text is None:
            continue

        found = True

        # save csv
        writer.writerow([
            image_name,
            plate_text,
            text_score
        ])

        # yellow box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 255),
            2
        )

        # black label
        cv2.rectangle(
            frame,
            (x1, y1 - 40),
            (x2, y1),
            (0, 0, 0),
            -1
        )

        # white text
        cv2.putText(
            frame,
            plate_text,
            (x1 + 5, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

    if not found:

        writer.writerow([
            image_name,
            "NO PLATE DETECTED",
            0
        ])

    # save output
    output_path = os.path.join(
        output_folder,
        image_name
    )

    cv2.imwrite(output_path, frame)

    print(f"Processed: {image_name}")

csv_file.close()

print("\nBatch testing completed.")