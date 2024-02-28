from shutil import copy

import pandas as pd
import plotly.graph_objects as go
from utils.helper import RESULTS_PATH, get_all_output_full_image_paths

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

fig = px.bar(homogenetic_poses, x=homogenetic_poses.index, y="Silhouette Score", title="Silhouette Score über die Jahrzehnte",)
fig.show()

# Homogenity seems to increase over time

"""Genre Results"""
data_paths_noAR_noKR = [sp / "data.pickle" for sp in RESULTS_PATH.iterdir()
                        if sp.name.startswith("K14P") and sp.name.endswith("noKR") and sp.is_dir() and "allG" in sp.name]

data_paths_AR_noKR = [sp / "data.pickle" for sp in (RESULTS_PATH / "arm_ranges").iterdir()
                      if sp.name.startswith("K14P") and sp.name.endswith("ARnoKR") and sp.is_dir() and "allG" in sp.name]

data_paths_AR_KR = [sp / "data.pickle" for sp in (RESULTS_PATH / "arm_ranges").iterdir()
                    if sp.name.startswith("K14P") and sp.name.endswith("ARKR") and sp.is_dir() and "allG" in sp.name]


def plot_results(data_paths, title_suffix):
    full_data = pd.concat([pd.read_pickle(sp) for sp in data_paths], ignore_index=True)
    full_data.loc[:, "Women_Percentage"] = full_data.Men_Percentage  # Fix wrong column name
    full_data = full_data.drop(columns=["Men_Percentage"])
    genres = full_data.iloc[:, 9:-20].sum().loc[full_data.iloc[:, 9:-20].sum() > 50].index

    genre_demographics = pd.Series(index=genres, dtype=float)
    genre_results = pd.Series(index=genres, dtype=float)
    for genre in genres:
        genre_demographics[genre] = full_data.loc[full_data.loc[:, genre].astype(bool), "Women_Percentage"].mean()
        genre_results[genre] = full_data.loc[full_data.loc[:, genre].astype(bool), "Women_Percentage"].unique().std()

    genre_results = genre_results.sort_values()
    genre_demographics = genre_demographics.loc[genre_results.index]

    # visualisation 2 subplots
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Frauenanteil in Stichprobe", "Std-Abw Frauenanteile in Cluster"))
    fig.update_layout(title=f"Box-Plot: Geschlechter Balance ({title_suffix})", yaxis_title="Frauenanteil in Cluster")
    fig.add_trace(go.Bar(x=genre_demographics.index, y=genre_demographics, name="Demographics"), row=1, col=1)
    fig.add_trace(go.Bar(x=genre_results.index, y=genre_results, name="Results"), row=1, col=2)
    fig.show()

    # plot Boxplot
    # family = full_data.loc[full_data.Familienfilm.astype(bool), "Women_Percentage"].sort_values().reset_index(drop=True)

    plot_data = {}
    for genre in genre_results.index:
        plot_data[genre] = full_data.loc[
            full_data.loc[:, genre].astype(bool), "Women_Percentage"].sort_values().reset_index(drop=True)

    genres = sorted(plot_data, key=lambda x: plot_data[x].median())
    fig = go.Figure()
    # title
    fig.update_layout(title=f"Box-Plot: Geschlechter Balance ({title_suffix})", xaxis_title="Genre",
                      yaxis_title="Frauenanteil in Cluster")

    for genre in genres:
        fig.add_trace(go.Box(y=plot_data[genre], name=genre))
    fig.show()

plot_results(data_paths_noAR_noKR, "Allgemeine Pose")
plot_results(data_paths_AR_noKR, "Distanz der Arme/Hände")
plot_results(data_paths_AR_KR, "Distanz der Arme/Hände und Knie")


"""Get Images representing center of clusters"""
data_paths = data_paths_noAR_noKR + data_paths_AR_noKR + data_paths_AR_KR
full_data = pd.concat([pd.read_pickle(sp) for sp in data_paths], ignore_index=True)
full_data.loc[:, "Women_Percentage"] = full_data.Men_Percentage  # Fix wrong column name
full_data = full_data.drop(columns=["Men_Percentage"])

full_data = full_data.join(full_data.groupby(["Identifier", "cluster"]).distance_to_center.rank(method="first"), rsuffix="_rank")
data_top4 = full_data.loc[full_data.distance_to_center_rank < 5]
data_top4 = data_top4.loc[:, ["Identifier", "cluster", "label"]].sort_values(["Identifier", "cluster"])
data_top4 = {f"{row.Identifier}_{row.cluster}": [
        row_IC.label for row_IC in data_top4.loc[(data_top4.Identifier == row.Identifier) & (data_top4.cluster == row.cluster)].itertuples()
    ] for row in data_top4.itertuples()}

full_image_paths = get_all_output_full_image_paths()
for key, values in data_top4.items():
    identifier, cluster = key.split("_")
    result_path = RESULTS_PATH / f"images" / identifier
    result_path.mkdir(exist_ok=True, parents=True)

    # Copy image to new location
    for i, value in enumerate(values):
        copy(full_image_paths[[x.stem for x in full_image_paths].index(value)], result_path / f"{cluster}_{i}.jpg")


"""Cluster Analysis"""
cluster_analysis = full_data.loc[full_data.cluster_size > 10]
cluster_mean = cluster_analysis.groupby(["Identifier", "cluster"]).distance_to_center.mean().dropna()
cluster_std = cluster_analysis.groupby(["Identifier", "cluster"]).distance_to_center.std().dropna()

# visualisation a boxplot for each Identifier

identifiers = cluster_mean.index.get_level_values(0).unique()
Identifier_AR = [identifier for identifier in identifiers if "noAR" not in identifier and "noKR" in identifier and "allG" in identifier]
Identifier_KR = [identifier for identifier in identifiers if "noKR" not in identifier and "allG" in identifier]
Identifier_Pose = [identifier for identifier in identifiers if "noARnoKR" in identifier and "allG" in identifier]

# Make 3 subplots each for one identifier_list
for identifiers, title in zip([Identifier_AR, Identifier_KR, Identifier_Pose], ["Distanz der Arme/Hände", "Distanz der Arme/Hände und Knie", "Allgemeine Pose"]):
    fig = go.Figure()
    # title
    fig.update_layout(title=f"Box-Plot: Distanz zum Zentrum der Cluster ({title})", xaxis_title="Cluster",
                      yaxis_title="Distanz zum Zentrum")

    for identifier in identifiers:
        data = cluster_mean.loc[identifier].sort_values().reset_index(drop=True)
        fig.add_trace(go.Box(y=data, name=identifier[9:13]))
    fig.show()


for identifiers, title in zip([Identifier_AR, Identifier_KR, Identifier_Pose], ["Distanz der Arme/Hände", "Distanz der Arme/Hände und Knie", "Allgemeine Pose"]):
    fig = go.Figure()
    # title
    fig.update_layout(title=f"Box-Plot: Distanz zum Zentrum der Cluster ({title})", xaxis_title="Cluster",
                      yaxis_title="Distanz zum Zentrum")

    for identifier in identifiers:
        data = cluster_std.loc[identifier].sort_values()# .reset_index(drop=True)
        fig.add_trace(go.Box(y=data, name=identifier[9:13]))
    fig.show()
