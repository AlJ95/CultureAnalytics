from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

data_path = Path("./data/imdb_html_data_sampled/")

if not data_path.exists():
    data_path = Path("../data/imdb_html_data_sampled/")

html_files = list(data_path.iterdir())
data = pd.DataFrame([], columns=["tid", "genres"])
for i, html_file_path in enumerate(html_files):
    if (i % 100) == 0:
        print(i, "/" , len(html_files))
    with open(html_file_path, "r", encoding="utf-8") as html_file:
        html = html_file.read()

    html = BeautifulSoup(html, features="html.parser")

    genres = html.findAll("a", {"class": "ipc-chip ipc-chip--on-baseAlt"})

    data = pd.concat([
        data,
        pd.DataFrame([{"tid": html_file_path.stem, "genres": ";".join([g.text for g in genres])}]),
        ], ignore_index=True)

data.to_pickle(data_path.parent / "html_genre_data.pickle")
