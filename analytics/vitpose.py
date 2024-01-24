"""
Pipeline for applying object detection on images and after that pose estimation.

TODO: Pose Estimation only works on one person per image
    -> Apply object detection first and then pose estimation on all persons.
"""

import cv2
from ultralytics import YOLO
import onepose

from pathlib import Path
import pickle

detection_model = YOLO("yolov8m.pt")

pose_estimation_model = onepose.create_model().to("cuda")

for year in range(2022, 1980, -1):

    print()
    print(year)
    print()
    Path(f"output/{year}").mkdir(parents=True, exist_ok=True)

    sub_paths = [sp for sp in Path(f"./imdb_image_data_sampled/{year}").iterdir() if sp.suffix == ".jpg"]
    for i, img_path in enumerate(sub_paths):

        if i % 25 == 0:
            print(f"{year}: {i} - {i / len(list(sub_paths)) * 100:.2f}%")

        img = cv2.imread(str(img_path))
        draw_img = img.copy()

        results = detection_model(img, verbose=False)[0]
        boxes = results.boxes.xyxy
        clses = results.boxes.cls
        probs = results.boxes.conf

        for i, cls, box, prob in zip(range(len(clses)), clses, boxes, probs):
            if cls != 0:
                continue

            x1, y1, x2, y2 = box
            # crop image
            person_img = img[int(y1):int(y2), int(x1):int(x2)]
            cv2.rectangle(draw_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)

            keypoints = pose_estimation_model(person_img)
            num_keypoints = len(keypoints['points'])

            for i in range(num_keypoints):
                keypoints['points'][i][0] += x1
                keypoints['points'][i][1] += y1

            onepose.visualize_keypoints(draw_img, keypoints, pose_estimation_model.keypoint_info,
                                        pose_estimation_model.skeleton_info)

            # save keypoints
            with open(f"output/{year}/{img_path.stem}_{i}.pkl", "wb") as f:
                pickle.dump(keypoints, f)

        cv2.imwrite(f"output/{year}/{img_path.name}", draw_img)
        cv2.waitKey(0)


