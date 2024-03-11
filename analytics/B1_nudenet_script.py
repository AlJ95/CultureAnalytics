"""
This script is for the prediction of the sex of the actors in the images.
"""
import pandas as pd
from nudenet import NudeDetector
import cv2
from ultralytics import YOLO

from pathlib import Path
import pickle

detection_model = YOLO("yolov8m.pt").to("cuda")
nude_detector = NudeDetector()

data_path = Path("./data/output/nudenet/raw_images")

def guess_sex(sub_paths):

    sex_data = pd.DataFrame([], columns=["tid", "sex"])

    for i, img_path in enumerate(sub_paths):

        if i % 25 == 0:
            print(f"{sub_paths[0].stem}: {i} - {i / len(list(sub_paths)) * 100:.2f}%")

        sex = nude_detector.detect(img_path)

        sex_data = pd.concat([
            sex_data,
            pd.DataFrame([{"tid": img_path.stem, "sex": sex}])
        ])

    print(f"----------------------{len(sex_data)}")
    sex_data.loc[:, "sex"] = sex_data.sex.apply(lambda x: x[0]["class"] == "FACE_FEMALE" if x else pd.NA)
    sex_data.loc[:, "sex"] = sex_data.sex.apply(
        lambda x: "F" if not pd.isna(x) and x else "M" if not pd.isna(x) and not x else pd.NA)

    sex_data.to_pickle(f"../data/output/sex_data_{sub_paths[0].stem}.pickle")

if __name__ == '__main__':
    path = Path("./data/output/nudenet/pickles/")

    files = list(path.iterdir())

    data = pd.concat([pd.read_pickle(p) for p in files])
    data.to_pickle("data/metadata_sex.pickle")