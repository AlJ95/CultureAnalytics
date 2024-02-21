"""
Clustering Poses

Hyperparameter:
    1) Body Connections - Which body parts will be addressed in clustering?
    2)
"""
import plotly.express as px
from scipy.spatial import distance

from analytics.cleaning import read_metadata, get_all_genres
from analytics.plot_image_collage import plot_image_collage
from utils.helper import get_image_path_by_imdb_id
from pathlib import WindowsPath
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
import pandas as pd
import numpy as np
from typing import List, Tuple
from utils.helper import get_all_vitpose_pickle_paths, RESULTS_PATH

try:
    from angle import calculate_unit_circle_pos
    from const import *
except ModuleNotFoundError:
    from analytics.const import *
    from analytics.angle import calculate_unit_circle_pos


def filter_pose_data(pose_data_raw, median_confidence=0.75, min_confidence=0.25):
    """
    Filter pose data by median confidence and minimum confidence
    :param pose_data_raw: dict of pose data
    :param median_confidence: median confidence of all keypoints
    :param min_confidence: minimum confidence of all keypoints
    :return: filtered pose data as dict
    """
    return {key: pose["points"] for key, pose in pose_data_raw.items()
            if np.median(pose["confidence"]) > median_confidence and pose["confidence"].min() > min_confidence
            and (~np.isnan(pose["points"])).all()}


def calculate_angles(pose_data: dict, body_connections: List[Tuple[int]],
                     backbone_angle=True) -> List[List[Tuple[int]]]:
    # plot lines between keypoints on image
    angles = []
    for pose in pose_data.values():

        angle_set = []
        for p1, p2 in body_connections:
            x, y = 0, 1
            vec_x = [pose[p1][x], pose[p2][x]]
            vec_y = [pose[p1][y], pose[p2][y]]

            # calculate angle between two vectors
            angle = calculate_unit_circle_pos(np.array(vec_x), np.array(vec_y))

            angle_set.append(angle)

        vec_x = [0, 1]
        vec_y = [pose[p1][y], pose[p2][y]]
        angles.append(angle_set)

    return angles


def join_with_metadata(pose_dict: dict, pickle_paths: List[WindowsPath]):
    pose_df = pd.DataFrame(pose_dict.items())
    pose_df = pose_df.set_index(0, drop=True)
    pose_df.index.name = "ImgId"
    pose_df.columns = ["Angles"]

    year_df = pd.DataFrame(pickle_paths)
    year_df.loc[:, "Year"] = year_df.iloc[:, 0].apply(lambda path: path.parent.name)
    year_df.iloc[:, 0] = year_df.iloc[:, 0].apply(lambda path: path.stem)
    year_df = year_df.set_index(0, drop=True)
    year_df.index.name = "ImgId"

    pose_df = pose_df.join(year_df)

    pose_df.loc[:, "join_index"] = pose_df.index.str.replace(r"_\d*$", "", regex=True)
    metadata = read_metadata()
    metadata.index.name = "join_index"
    pose_df = pose_df.join(metadata, on="join_index")

    pose_df.loc[:, metadata.columns] = pose_df.loc[:, metadata.columns].fillna(0)

    pose_df.loc[:, "join_index"] = pose_df.index
    metadata = read_metadata("sex")
    metadata.columns = ["join_index", "sex"]
    metadata = metadata.set_index("join_index")
    pose_df = pose_df.join(metadata, on="join_index")

    return pose_df.drop(columns=["join_index"])


if __name__ == '__main__':

    pickle_paths = get_all_vitpose_pickle_paths()
    # pose_data_raw = {pickle_path.stem: pd.read_pickle(pickle_path) for pickle_path in pickle_paths}

    # Hyperparameter Tuning
    median_confidence = 0.75
    min_confidence = 0.25
    n_clusters = [16]  # [14, 15, 17, 18]
    body_connection_names = [
        "body_connections_2"]  # [f"body_connections_{i}" for i in range(1, 4)] + ["body_connections_full"]
    cluster_names = ["KMeans"]  # "SpectralClustering", "AgglomerativeClustering"
    dim_red_algorithms = ["tsne"]  # ["tsne", "PCA"]
    years = []
    genres = [None]  # get_all_genres()
    sex = None

    combinations = [(n_cluster, body_connection_name, cluster_name, dim_red_algorithm, genre)
                    for n_cluster in n_clusters
                    for body_connection_name in body_connection_names
                    for cluster_name in cluster_names
                    for dim_red_algorithm in dim_red_algorithms
                    for genre in genres
                    ]

    for i, combination in enumerate(combinations):
        print(i + 1, "/", len(combinations))

        n_cluster, body_connection_name, cluster_name, dim_red_algorithm, genre = combination

        pose_data_raw = pd.read_pickle("./data/output/VitPose/box_clips/all_data.pkl")
        pose_data = filter_pose_data(pose_data_raw, median_confidence=0.75, min_confidence=0.25)

        # filter for Hyperparameter
        pose_data = join_with_metadata(pose_dict=pose_data, pickle_paths=pickle_paths)

        if genre:
            pose_data = pose_data.loc[pose_data.loc[:, genre] == 1]

        if years:
            pose_data = pose_data.loc[pose_data.Year.isin(years), :]

        # preprocess
        body_connections = eval(body_connection_name)
        angles = calculate_angles(pose_data.Angles.to_dict(), body_connections)

        """
        START K-MEANS
        """

        # convert angles to numpy array
        if isinstance(angles[0], np.float32):
            angles = np.array(angles)
        else:
            angles = np.array([[x for y in angle_set for x in y] for angle_set in angles])

        cluster_fct = {"KMeans": KMeans,
                       "AgglomerativeClustering": AgglomerativeClustering,
                       "SpectralClustering": SpectralClustering
                       }[cluster_name]

        # create k-means model
        kwargs = {}

        if cluster_fct in [KMeans, SpectralClustering]:
            kwargs.update({"random_state": 0})

        cluster_result = cluster_fct(n_clusters=n_cluster).fit(angles)

        # get cluster labels
        labels = cluster_result.labels_

        # get cluster centers
        if cluster_fct == KMeans:
            centers = cluster_result.cluster_centers_

        else:
            center_res = [angles[labels == c].mean(0) for c in range(n_cluster)]
            centers = [center_res[c] for c in labels]

        # distance to cluster center
        distance_to_center = [round(distance.euclidean(a, centers[cluster]), 3) for a, cluster in zip(angles, labels)]

        # get cluster sizes
        cluster_sizes = np.unique(labels, return_counts=True)[1]

        """
        START DIMENSIONALITY REDUCTION & PLOTTING
        """

        dim_rec_fct = PCA if dim_red_algorithm == "PCA" else TSNE

        # reduce dimensionality
        angles_2d = dim_rec_fct(n_components=2, random_state=0).fit_transform(angles)

        """
        Use Plotly for annotated scatter plot
        """

        # create dataframe
        df = pd.DataFrame(angles_2d, columns=["x", "y"])
        df["label"] = pose_data.index.values
        df["cluster"] = labels
        df["cluster"] = df.cluster.astype("category")
        df["distance_to_center"] = distance_to_center
        df["cluster_size"] = cluster_sizes[df["cluster"].cat.codes]
        df = df.set_index("label").join(pose_data).reset_index()
        df = df.loc[df.sex != "NA"]

        # create figure
        fig = px.scatter(df,
                         x="x",
                         y="y",
                         color="cluster",
                         color_discrete_sequence=px.colors.qualitative.Plotly,
                         symbol="sex",
                         symbol_sequence= ['circle', 'circle-open'],
                         hover_data=["label", "cluster_size", "distance_to_center"]
                         )

        path = (
                RESULTS_PATH /
                f"{cluster_name} - "
                f"{dim_red_algorithm} - "
                f"{body_connection_name} - "
                f"{'all years' if not years else ''.join([str(y) for y in years])} - "
                f"n={n_cluster} - "
                f"med_conf_limit={int(median_confidence * 100)}% - "
                f"min_conf_limit={int(min_confidence * 100)}%"
        )
        path.mkdir(exist_ok=True)
        path = path / genre
        path.mkdir(exist_ok=True)

        fig.write_html(path / "Cluster Plot.html")

        collage_samples = [df.loc[df["cluster"] == cluster] for cluster in range(n_cluster)]
        # collage_samples = [data.sample(min(len(data), 60)).loc[:, , "label"] for data in collage_samples]
        collage_samples = [data.sort_values("distance_to_center").loc[:, "label"].iloc[:20] for data in collage_samples]
        collage_samples = [list(data.apply(lambda x: get_image_path_by_imdb_id(x, use_vitpose_image=True))) for data in
                           collage_samples]

        for i, cluster in enumerate(collage_samples):
            print(f"Cluster {i} has {len(cluster)} samples")
            plot_image_collage(cluster, f"cluster_{i}", path)

        EVAL_PATH = "./results/eval.csv"

        result = pd.DataFrame({
            "Clustering": cluster_name,
            "Cluster N": n_cluster,
            "Dimnesion Reduction Algorithm": dim_red_algorithm,
            "Body Connections": body_connection_name,
            "Years": 'all years' if not years else ''.join([str(y) for y in years]),
            "Median Confidence Lower Bound": median_confidence,
            "Minimum Confidence Lower Bound": min_confidence,
            "Silhouette Score": silhouette_score(angles, labels, metric='euclidean'),
            "Calinski Harabasz Score": calinski_harabasz_score(angles, labels),
            "Davis Bouldin Score": davies_bouldin_score(angles, labels),
            "Variance of Higher Dimensional Data": angles.var(),
            "Variance of 2D Data": angles_2d.var(),
            "% Lost Variance": 1 - angles_2d.var() / angles.var(),
            "Genre": genre
        }, index=[1])

        pd.concat([
            pd.read_csv(EVAL_PATH, sep=";", decimal=","),
            result
        ], ignore_index=True).to_csv(EVAL_PATH, sep=";", decimal=",", index=False)

        sex_cluster_representation = df.groupby(["cluster", "sex"]).count().reset_index().pivot(index="cluster",
                                                                                                columns="sex",
                                                                                                values="label")
        sex_cluster_representation = sex_cluster_representation.apply(lambda x: x.F / (x.M + x.F), axis=1).sort_values()
        print(sex_cluster_representation)