from pathlib import Path

import pandas as pd
import re
from bs4 import BeautifulSoup

from utils.helper import get_all_sampled_html_file_paths

GENRE_CLASS_NAME = "ipc-chip ipc-chip--on-baseAlt"
RATING_CLASS_NAME = "sc-bde20123-1 cMEQkK"

data_path = Path("./data/")

if not data_path.exists():
    data_path = Path("../data/")


def extract_html_information():
    metadata = pd.DataFrame([], columns=["tid", "Genres", "Rating"])

    paths = get_all_sampled_html_file_paths()

    for i, path in enumerate(paths):
        if i < 35001:
            continue

        if i % 100 == 0:
            print(i, "/", len(paths))

        content = BeautifulSoup(open(path, "r", encoding="utf-8"))

        # genres
        try:
            items = content.findAll("a", {"class": GENRE_CLASS_NAME})
            genres = ";".join([next(item.children).text for item in items])
        except Exception as e:
            print(path.stem, e)
            genres = ""

        # rating
        try:
            items = content.findAll("span", {"class": RATING_CLASS_NAME})
            rating = [next(item.children).text for item in items][0]
        except Exception as e:
            print(path.stem, e)
            rating = None

        metadata = pd.concat([
            metadata,
            pd.DataFrame([[path.stem, genres, rating]], columns=["tid", "Genres", "Rating"])
        ], ignore_index=True)

        if i % 1000 == 0:
            metadata.to_csv(f"./data/metadata{i}.csv")
            metadata = pd.DataFrame([], columns=["tid", "Genres", "Rating"])

    metadata.to_csv(f"./data/metadata{i}.csv")


def read_metadata(type="genres"):
    return pd.read_pickle(data_path / f"metadata_{type}.pickle")


def get_all_genres():
    return read_metadata(type="genres").columns.tolist()


if __name__ == '__main__':
    metadata = [pd.read_csv(fp) for fp in data_path.iterdir() if fp.stem.startswith("metadata") and fp.suffix == ".csv"]
    metadata = pd.concat(metadata)
    metadata = metadata.loc[~pd.isna(metadata.Genres)]

    all_genres = metadata.Genres.str.split(";").explode().unique().tolist()

    metadata.loc[:, "Genres Exploded"] = metadata.Genres.str.split(";")
    metadata = metadata.explode("Genres Exploded")
    metadata = metadata.loc[:, ["tid", "Genres Exploded"]]
    metadata.loc[:, "value"] = 1

    metadata = metadata.pivot(index="tid", columns="Genres Exploded", values="value").fillna(0)

    metadata.to_pickle(data_path / "metadata_genres.pickle")
