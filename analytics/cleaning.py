import pandas as pd
import re
from bs4 import BeautifulSoup

from utils.helper import get_all_sampled_html_file_paths


GENRE_CLASS_NAME = "ipc-chip ipc-chip--on-baseAlt"
RATING_CLASS_NAME = "sc-bde20123-1 cMEQkK"

if __name__ == '__main__':

    metadata = pd.DataFrame([], columns=["tid", "Genres", "Rating"])

    paths = get_all_sampled_html_file_paths()

    for i, path in enumerate(paths):
        if i < 17375:
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

    metadata.to_csv("./data/metadata2.csv")
