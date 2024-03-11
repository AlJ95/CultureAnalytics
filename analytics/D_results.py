"""
This script is used to visualize the results of the clustering. It contains the code for the final visualizations in the report.
"""
import math
import time
from random import random
from PIL import Image
from plotly.subplots import make_subplots
import pandas as pd
import plotly.graph_objects as go
from utils.helper import RESULTS_PATH, get_all_output_full_image_paths
from PIL import ImageOps

"""Load Results"""
data = pd.read_pickle("./results/data.pickle").drop_duplicates(["Identifier", "cluster"], keep="first")
data.loc[:, "Women_Percentage"] = data.Men_Percentage  # Fix wrong column name
data = data.drop(columns=["Men_Percentage"])
genre_results_pivoted = data.loc[data.Genre & ~data.Years] \
    .pivot(index="cluster", columns="Identifier", values="Women_Percentage")
genre_results_pivoted.to_excel("./results/Genre Results Pivoted.xlsx")

"""General Results"""
results = pd.read_pickle(RESULTS_PATH / "data.pickle")
homogenetic_poses = (
    results
    .assign(Jahrzehnt=lambda x: x.Identifier.str[9:13])
    .loc[(results.n == 372) & (
            results.loc[:, "Cluster N"] == 14)]  # Only results from 14 clusters and equally sampled data
    .groupby("Jahrzehnt")
    .mean()
    .loc[:, ("Silhouette Score")]
    .sort_values()
)
homogenetic_poses.index = homogenetic_poses.index.str.cat((homogenetic_poses.index.astype(int) + 10).astype(str),
                                                          sep="-")

# visualisation
import plotly.express as px

fig = px.bar(homogenetic_poses, x=homogenetic_poses.index, y="Silhouette Score",
             title="Silhouette Score über die Jahrzehnte", )
fig.show()

# Homogenity seems to increase over time

"""Genre Results"""

genres = ["Action", "Dokumentarfilm", "Drama", "Komödie", "Krimi", "Liebesfilm"]

data_genres_POSE = {genre: pd.read_pickle(RESULTS_PATH / f"K14P25020allY{genre}noARnoKR" / genre / "data.pickle") for genre in genres}
title_suffix_POSE = "Allgemeine Pose"
data_genres_AR = {genre: pd.read_pickle(RESULTS_PATH / "arm_ranges" / f"K14P25020allY{genre}ARnoKR" / genre / "data.pickle") for genre in genres}
title_suffix_AR = "Distanz der Arme/Hände"
data_genres_ARKR = {genre: pd.read_pickle(RESULTS_PATH / "arm_ranges" / f"K14P25020allY{genre}ARKR" / genre / "data.pickle")
               for genre in genres}
title_suffix_ARKR = "Distanz der Arme/Hände und Knie"

for data_genres, title_suffix in zip([data_genres_POSE, data_genres_AR, data_genres_ARKR],
                                     [title_suffix_POSE, title_suffix_AR, title_suffix_ARKR]):

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=("Frauenanteil in Stichprobe", "Std-Abw Frauenanteile",
                                        "Frauenanteil im Cluster vs. Clustergröße", "Std-Abw Frauenanteile vs. Clustergröße"),
                        # specs=[[{"type": "box"}, {"type": "box"}],
                        #        [{"type": "scatter", "colspan": 1}, None]]
                        )
    fig.update_layout(title=f"Geschlechter Balance ({title_suffix})", yaxis_title="Frauenanteil in Cluster")

    mean_data, std_data = [], []
    potential_normal_data = []
    for genre, data in data_genres.items():
        mean_data.append(data.drop_duplicates(["Identifier", "cluster"]).Women_Percentage.mean())
        std_data.append(data.drop_duplicates(["Identifier", "cluster"]).Women_Percentage.std())
        observ_data = data.drop_duplicates(["Identifier", "cluster"]).loc[:, ["cluster_size", "Women_Percentage"]]
        potential_normal_data += observ_data.Women_Percentage.tolist()

        fig.add_trace(go.Scatter(x=observ_data.Women_Percentage,
                                 y=observ_data.cluster_size,
                                 mode="markers", name=genre),
                      row=2, col=1)

    genre_data = pd.DataFrame({"Genre": genres, "Mean": mean_data, "Std": std_data}).sort_values("Std")

    fig.add_trace(go.Bar(x=genre_data.Genre, y=genre_data.Mean, name="Mittelwert"), row=1, col=1)
    fig.add_trace(go.Bar(x=genre_data.Genre, y=genre_data.Std, name="Standardabweichung"), row=1, col=2)
    fig.add_trace(go.Scatter(x=std_data, y=[len(x) for x in data_genres.values()], mode="markers", name="Clustergröße"),
                  row=2, col=2)
    # edit axis labels
    fig['layout']['yaxis']['title'] = "Mittelwert"
    fig['layout']['yaxis2']['title'] = "Standardabweichung"
    fig['layout']['yaxis3']['title'] = "Clustergröße"
    fig['layout']['yaxis4']['title'] = "Clustergröße"
    fig['layout']['xaxis3']['title'] = "Frauenanteil in Cluster"
    fig['layout']['xaxis4']['title'] = "Standardabweichung"
    fig.show()

"""Jahrzehnte Results"""

years = [1982, 1992, 2002, 2012]
data_years_POSE = {year: pd.read_pickle(RESULTS_PATH / f"K14P25020{year}allGnoARnoKR" / "data.pickle") for year in years}
title_suffix_POSE = "Allgemeine Pose"
data_years_AR = {year: pd.read_pickle(RESULTS_PATH / "arm_ranges" / f"K14P25020{year}allGARnoKR" /"data.pickle") for year in years}
title_suffix_AR = "Distanz der Arme/Hände"
data_years_ARKR = {year: pd.read_pickle(RESULTS_PATH / "arm_ranges" / f"K14P25020{year}allGARKR" /"data.pickle")
               for year in years}
title_suffix_ARKR = "Distanz der Arme/Hände und Knie"

for data_years, title_suffix in zip([data_years_POSE, data_years_AR, data_years_ARKR],
                                     [title_suffix_POSE, title_suffix_AR, title_suffix_ARKR]):

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=("Frauenanteil in Stichprobe", "Std-Abw Frauenanteile",
                                        "Frauenanteil im Cluster vs. Clustergröße", "Std-Abw Frauenanteile vs. Clustergröße"),
                        specs=[[{"type": "box"}, {"type": "box"}],
                               [{"type": "scatter", "colspan": 2}, None]]
                        )
    fig.update_layout(title=f"Geschlechter Balance ({title_suffix})", yaxis_title="Frauenanteil in Cluster")

    mean_data, std_data = [], []
    potential_normal_data = []
    for year, data in data_years.items():
        data.loc[:, "Women_Percentage"] = data.Men_Percentage  # Fix wrong column name
        data = data.drop(columns=["Men_Percentage"])

        mean_data.append(data.drop_duplicates(["Identifier", "cluster"]).Women_Percentage.mean())
        std_data.append(data.drop_duplicates(["Identifier", "cluster"]).Women_Percentage.std())
        observ_data = data.drop_duplicates(["Identifier", "cluster"]).loc[:, ["cluster_size", "Women_Percentage"]]
        potential_normal_data += observ_data.Women_Percentage.tolist()

        fig.add_trace(go.Scatter(x=observ_data.Women_Percentage,
                                 y=observ_data.cluster_size,
                                 mode="markers", name=f"{year}-{year + 9}"),
                      row=2, col=1)

    year_data = pd.DataFrame({"year": [f"{year}-{year + 9}" for year in years], "Mean": mean_data, "Std": std_data})

    fig.add_trace(go.Bar(x=year_data.year, y=year_data.Mean, name="Mittelwert"), row=1, col=1)
    fig.add_trace(go.Bar(x=year_data.year, y=year_data.Std, name="Standardabweichung"), row=1, col=2)
    # edit axis labels
    fig['layout']['yaxis']['title'] = "Mittelwert"
    fig['layout']['yaxis2']['title'] = "Standardabweichung"
    fig['layout']['yaxis3']['title'] = "Clustergröße"
    fig['layout']['xaxis3']['title'] = "Frauenanteil in Cluster"
    fig.show()

data_POSE = data_years_POSE
data_POSE.update(data_genres_POSE)
data_AR = data_years_AR
data_AR.update(data_genres_AR)

for k, data_ in {"POSE": data_POSE, "AR": data_AR}.items():

    for key in [1982, 1992, 2002, 2012]:
        if key in data_:
            data_[f"{str(key)}-{str(key + 9)}"] = data_.pop(key)

    image_paths = {x.stem:x for x in get_all_output_full_image_paths()}
    for key, data in data_.items():

        t0 = time.time()
        df = data

        # normalize data to fit images
        # 25 and 50 fit good for the genre Action that has 200 samples
        # so we need to adjust the factor to fit the images for other genres and years
        x_factor = 25 * (math.sqrt(len(df)) / math.sqrt(200))
        y_factor = 50 * (math.sqrt(len(df)) / math.sqrt(200))
        df["x"] *= x_factor / (df["x"].max() - df["x"].min())
        df["y"] *= y_factor / (df["y"].max() - df["y"].min())

        # create figure
        fig = px.scatter(df,
                         x="x",
                         y="y",
                         color="sex",
                         color_discrete_sequence=["violet", "orange"],
                         symbol="sex",
                         symbol_sequence=['circle', 'circle-open'],
                         hover_data=["label", "cluster_size", "distance_to_center"]
                         )

        for i, row in df.iterrows():

            if row.sex == "M":
                # make the image half the size
                img = Image.open(image_paths[row.label])
                img = ImageOps.expand(img, border=20, fill="orange")
            else:
                img = Image.open(image_paths[row.label])
                img = ImageOps.expand(img, border=20, fill="violet")

            w, h = img.size
            img = img.resize((w // 2, h // 2))

            fig.add_layout_image(
                x=row.x,
                y=row.y,
                source=img,
                xref="x",
                yref="y",
                sizex=2,
                sizey=2,
                xanchor="center",
                yanchor="middle",
            )

        # update title
        fig.update_layout(
            title=f"{k} Clustering {key}",
            xaxis_title="",
            yaxis_title="",
            # showlegend=False,
        )

        # save figure
        fig.write_html(f"./results/{key}_GESCHLECHT_{k}.html")

        t1 = time.time()

        print(f"Time taken: {t1-t0}")
        print(f"Full Time for all: {2 * 10 *(t1-t0)}")

