"""
This script analyzes the data demographics of the project. It is used to get an overview of the data
"""

import re

import pandas as pd
from pathlib import Path

from analytics.C_full_pipeline import filter_pose_data

if __name__ == '__main__':
    """Data Demographics"""
    data_path = Path("./data")

    # Image Counts (1 per movie)
    imdb_nc_data_count = len(pd.read_csv(data_path / "imdb_data" / "title.basics.tsv" / "data.tsv", sep="\t"))
    imdb_image_data_count_by_year = {i: len(list((data_path / "imdb_image_data_sampled" / str(i)).glob(f"*")))
                                     for i in range(1981, 2022)}
    imdb_image_data_count = sum(imdb_image_data_count_by_year.values())

    genre_data = pd.read_pickle(data_path / "metadata_genres.pickle")
    genre_data_count = len(genre_data)

    print(f"IMDB NC Data Count: {imdb_nc_data_count}"
          f"\nIMDB Image Data Count: {imdb_image_data_count}"
          f"\nGenre Data Count: {genre_data_count}")

    # Person Image Counts (several per movie)
    yolov8_person_data_count_per_year = {
        i: len(list((data_path / "output" / "VitPose" / "box_clips" / str(i)).glob(f"*.jpg")))
        for i in range(1981, 2022)}
    yolov8_person_data_count = sum(yolov8_person_data_count_per_year.values())

    sex_data = pd.read_pickle(data_path / "metadata_sex.pickle")
    sex_data_count = len(sex_data)

    pose_data = pd.read_pickle("./data/output/VitPose/box_clips/all_data.pkl")
    pose_data_count = sum([len(x) for x in pose_data.values()])

    joined_data_count = sex_data.tid.isin(pose_data.keys()).sum()

    print(f"YOLOv8 Person Data Count: {yolov8_person_data_count}"
          f"\nSex Data Count: {sex_data_count}"
          f"\nPose Data Count: {pose_data_count}",
          f"\nJoined Data Count: {joined_data_count}")

    # Confidence Filter Counts
    pose_data_filtered = filter_pose_data(pose_data, 0.5, 0.2)

    print(f"Pose Data Filtered Count: {sum([len(x) for x in pose_data_filtered.values()])}")

    # Full Information Counts
    tids = set(re.sub(r"_\d*", "", i) for i in pose_data_filtered.keys())
    temp = genre_data.loc[genre_data.index.isin(tids)]
    pose_sex_genre_movies = sex_data.loc[
        sex_data.tid.str.replace(r"_\d*", "", regex=True).isin(temp.index)
    ]
    pose_sex_genre_movies_count = len(pose_sex_genre_movies)

    print(f"Person Images with Genre and Sex Information and necessary Pose Confidence: "
          f"{pose_sex_genre_movies_count}")

    print("Among these:"
          f"\nMen: {len(pose_sex_genre_movies.loc[pose_sex_genre_movies.sex == 'M'])}"
          f"\nWomen: {len(pose_sex_genre_movies.loc[pose_sex_genre_movies.sex == 'F'])}"
          f"\nUnknown: {len(pose_sex_genre_movies.loc[pose_sex_genre_movies.sex == 'NA'])}")

    movies_with_full_information = pose_sex_genre_movies.assign(
        movie_tid=lambda x: x.tid.str.replace(r'_\d*', '', regex=True)
    ).query("sex!='NA'")
    print("Movies with Genre and Sex Information and necessary Pose Confidence:"
          f"{movies_with_full_information.movie_tid.nunique()}")

    # Genre Counts
    genre_counts = genre_data.loc[genre_data.index.isin(movies_with_full_information.movie_tid)].sum()

    print(f"In {movies_with_full_information.movie_tid.nunique()} movies with full information we have "
          f"{genre_counts.sum()} genres. So movies have more than one genre."
          f"\n\nGenre Counts: {genre_counts}")

    tids_by_year = {i: list((data_path / "imdb_image_data_sampled" / str(i)).glob(f"*"))
                    for i in range(1981, 2022)}
    tids_by_year = {v.stem: i for i, l in tids_by_year.items() for v in l}
    tids_by_year = pd.DataFrame([tids_by_year]).T
    tids_by_year.columns = ["year"]

    sampled_movies_with_full_information = (
        movies_with_full_information
        .set_index("movie_tid")
        .join(tids_by_year)
        .query("year>1982")
        .assign(year10=lambda x: (x.year - 1982) // 10 * 10 + 1982)
        .assign(year10=lambda x: x.year10.astype(str).str.cat((x.year10 + 9).astype(str), sep="-"))
    )

    # Join Image Cut Data and Save it for Analysis
    sampled_data = pd.DataFrame([{k: re.sub(r"_\d*", "", k) for k, v in pose_data_filtered.items()}]).T.reset_index()
    sampled_data.columns = ["join_tid", "tid"]
    sampled_data = sampled_data.astype({"join_tid": str})

    tmp = sampled_movies_with_full_information.reset_index()
    tmp.columns = ["tid", "join_tid", "Geschlecht", "Jahr", "Jahrzehnt"]
    tmp = tmp.astype({"join_tid": str})
    sampled_data = sampled_data.merge(tmp, on="join_tid", how="left")
    sampled_data = sampled_data.loc[~sampled_data.tid_y.isna()]

    sd = pd.DataFrame([], columns=sampled_data.columns)
    for r in ["1982-1991", "1992-2001", "2002-2011", "2012-2021"]:
        for g in ["F", "M"]:
            sd = pd.concat([sd, sampled_data.query(f"Jahrzehnt=='{r}' and Geschlecht=='{g}'").sample(186)])
    sd.to_pickle("./data/sample_data.pickle")

    images_by_year10_and_sex_count = (
        sampled_movies_with_full_information
        .groupby(["year10", "sex"])
        .count()
        .reset_index()
        # .pivot(index="year10", columns="sex", values="v")
    )
    images_by_year10_and_sex_count.columns = ["Jahrzehnt", "Geschlecht", "tid", "Anzahl"]

    # Visualize
    import plotly.express as px
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=1, subplot_titles=("Anzahl Personen nach Genre", "Genre Counts"))
    # remove legend
    fig.update_layout(showlegend=False)
    # y axis label
    fig.update_yaxes(title_text="Anzahl Personen", row=1, col=1)
    fig.add_trace(px.bar(genre_counts).data[0], row=1, col=1)
    fig.show()

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Anzahl Bilder nach Geschlecht",
                                                        "Anzahl der Bilder nach Jahrzehnt "
                                                        "und Geschlecht"))
    # remove legend
    # y axis label
    fig.update_yaxes(title_text="Anzahl Personen", row=1, col=1)
    sex_result_count = pose_sex_genre_movies.groupby(["sex"]).count()
    sex_result_count.index = ["Frauen", "Männer", "Unbekannt"]
    fig.add_trace(px.bar(sex_result_count).data[0], row=1, col=1)
    # fig.add_trace(px.bar(images_by_year10_and_sex_count).data[0], row=1, col=2)
    # stacked bar chart
    fig.add_trace(px.bar(images_by_year10_and_sex_count, x="Jahrzehnt", y="Anzahl", color="Geschlecht").data[0], row=1,
                  col=2)
    fig.add_trace(px.bar(images_by_year10_and_sex_count, x="Jahrzehnt", y="Anzahl", color="Geschlecht").data[1], row=1,
                  col=2)
    fig.show()
