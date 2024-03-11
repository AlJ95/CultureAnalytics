"""
Clustering Poses

Hyperparameter:
    1) Body Connections - Which body parts will be addressed in clustering?
    2)
"""
import itertools

import plotly.express as px
from scipy.spatial import distance

from loader.D_metadata_cleaning import read_metadata, get_all_genres
from utils.helper import get_image_path_by_imdb_id
from pathlib import WindowsPath
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
import pandas as pd
import numpy as np
from typing import List, Tuple
from utils.helper import get_all_vitpose_pickle_paths, RESULTS_PATH

# necessary, because I switch between python live interpreter and running the script
try:
    from utils import calculate_unit_circle_pos, plot_image_collage
    from const import *
except ModuleNotFoundError:
    from analytics.const import *
    from analytics.utils import calculate_unit_circle_pos, plot_image_collage


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
        for (p1, p2), (p3, p4) in itertools.combinations(body_connections, 2):
            x, y = 0, 1
            vec_1 = np.array(pose[p1])
            vec_2 = np.array(pose[p2])
            vec_3 = np.array(pose[p3])
            vec_4 = np.array(pose[p4])

            vec_12 = vec_1 - vec_2
            vec_34 = vec_3 - vec_4

            # calculate angle between two vectors
            angle = calculate_unit_circle_pos(vec_12, vec_34)

            angle_set.append(angle)

        if backbone_angle:
            vec_12 = np.array([0, 1])

            vec_1 = np.array(pose[9])  # left_wrist
            vec_2 = np.array(pose[10])  # right_wrist
            vec_3 = np.array(pose[11])  # left_hip
            vec_4 = np.array(pose[12])  # right_hip

            vec_34 = np.mean(np.vstack([vec_1, vec_2]), axis=1) - np.mean(np.vstack([vec_3, vec_4]), axis=1)
            angle = calculate_unit_circle_pos(vec_12, vec_34)
            angle_set.append(angle)

        angles.append(angle_set)

    return angles


def calculate_arm_ranges(pose_data_raw: dict) -> List[List[Tuple[int]]]:
    """
    Calculate the euclidean distance between of left elbow and right elbow and left wrist and right wrist
    and min of both

    # left elbow: 7
    # right elbow: 8
    # left wrist: 9
    # right wrist: 10
    """
    arm_ranges = []
    for pose in pose_data_raw.values():
        arm_range = []
        for (p1, p2) in [(i, j) for i in range(7, 11) for j in range(7, 11) if i != j]:
            vec_1 = np.array(pose[p1])
            vec_2 = np.array(pose[p2])
            arm_range.append(distance.euclidean(vec_1, vec_2))
        arm_ranges.append(np.array([arm_range]))  # need to be same dimension as calculate_angles

    return arm_ranges


def calculate_knee_ranges(pose_data_raw: dict) -> List[List[Tuple[int]]]:
    """
    Calculate the euclidean distance between of left knee and right knee and left ankle and right ankle
    and min of both

    # left knee: 13
    # right knee: 14
    """
    knee_ranges = []
    for pose in pose_data_raw.values():
        p1, p2 = 13, 14
        vec_1 = np.array(pose[p1])
        vec_2 = np.array(pose[p2])

        # need to be same dimension as calculate_angles
        knee_ranges.append(np.array([[distance.euclidean(vec_1, vec_2)]]))

    return knee_ranges


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
    median_confidence = 0.5
    min_confidence = 0.2
    n_clusters = [14]   # [12, 13, 15, 16]
    body_connection_names = [
        "body_connections_2"]  # [f"body_connections_{i}" for i in range(1, 4)] + ["body_connections_full"]
    cluster_names = ["KMeans"]  # ["SpectralClustering", "AgglomerativeClustering"]
    dim_red_algorithms = ["PCA"]  # ["PCA", "TSNE"]
    years = [None]  # [[y + i for i in range(10)] for y in range(1982, 2022, 10)]
    genres = get_all_genres()   # [None]
    arm_ranges = False
    knee_ranges = True

    combinations = [(n_cluster, body_connection_name, cluster_name, dim_red_algorithm, genre, year_batch)
                    for n_cluster in n_clusters
                    for body_connection_name in body_connection_names
                    for cluster_name in cluster_names
                    for dim_red_algorithm in dim_red_algorithms
                    for genre in genres
                    for year_batch in years
                    ]

    for i, combination in enumerate(combinations):

        n_cluster, body_connection_name, cluster_name, dim_red_algorithm, genre, year_batch = combination

        pose_data_raw = pd.read_pickle("./data/output/VitPose/box_clips/all_data.pkl")
        pose_data = filter_pose_data(pose_data_raw, median_confidence=0.5, min_confidence=0.2)

        # filter for Hyperparameter
        pose_data = join_with_metadata(pose_dict=pose_data, pickle_paths=pickle_paths)

        # sampled data regarding sex and years
        pose_data = pose_data.loc[pose_data.index.isin(pd.read_pickle("./data/sample_data.pickle").join_tid.values)]

        if genre:
            pose_data = pose_data.loc[pose_data.loc[:, genre] == 1]

        if year_batch:
            pose_data = pose_data.loc[pose_data.Year.astype(int).isin(year_batch), :]

        if len(pose_data) < n_cluster * 2 or len(pose_data) < 100:
            print(f"Skipping {combination} due to insufficient data: {len(pose_data)} samples.")
            continue

        # preprocess
        body_connections = eval(body_connection_name)

        if arm_ranges:
            feature_set = calculate_arm_ranges(pose_data_raw=pose_data.Angles.to_dict())
        elif knee_ranges:
            feature_set = calculate_knee_ranges(pose_data_raw=pose_data.Angles.to_dict())
        elif arm_ranges and knee_ranges:
            feature_set = [np.concatenate([f1, f2], axis=1)
                           for f1, f2 in zip(
                    calculate_arm_ranges(pose_data_raw=pose_data.Angles.to_dict()),
                    calculate_knee_ranges(pose_data_raw=pose_data.Angles.to_dict())
                )]
        else:
            feature_set = calculate_angles(pose_data.Angles.to_dict(), body_connections)

        """
        START K-MEANS
        """

        # convert angles to numpy array
        if isinstance(feature_set[0], np.float32):
            feature_set = np.array(feature_set)
        else:
            feature_set = np.array([[x for y in angle_set for x in y] for angle_set in feature_set])

        cluster_fct = {"KMeans": KMeans,
                       "AgglomerativeClustering": AgglomerativeClustering,
                       "SpectralClustering": SpectralClustering
                       }[cluster_name]

        # create k-means model
        kwargs = {}

        if cluster_fct in [KMeans, SpectralClustering]:
            kwargs.update({"random_state": 0})

        cluster_result = cluster_fct(n_clusters=n_cluster).fit(feature_set)

        # get cluster labels
        labels = cluster_result.labels_

        # get cluster centers
        if cluster_fct == KMeans:
            centers = cluster_result.cluster_centers_

        else:
            center_res = [feature_set[labels == c].mean(0) for c in range(n_cluster)]
            centers = [center_res[c] for c in labels]

        # distance to cluster center
        distance_to_center = [round(distance.euclidean(a, centers[cluster]), 3) for a, cluster in
                              zip(feature_set, labels)]

        # get cluster sizes
        cluster_sizes = np.unique(labels, return_counts=True)[1]

        """
        START DIMENSIONALITY REDUCTION & PLOTTING
        """

        dim_rec_fct = PCA if dim_red_algorithm == "PCA" else TSNE

        # reduce dimensionality
        angles_2d = dim_rec_fct(n_components=2, random_state=0).fit_transform(feature_set)

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
                         symbol_sequence=['circle', 'circle-open'],
                         hover_data=["label", "cluster_size", "distance_to_center"]
                         )

        identifier = (cluster_name[0] + str(n_cluster) + dim_red_algorithm[0] +
                      body_connection_name.split("_")[-1] +
                      str(int(median_confidence * 100)) + str(int(min_confidence * 100)) +
                      ("allY" if not year_batch else str(year_batch[0])) +
                      (genre if genre else "allG") +
                      ("AR" if arm_ranges else "noAR") +
                      ("KR" if knee_ranges else "noKR"))

        path = RESULTS_PATH / identifier

        if arm_ranges:
            path = path.parent / "arm_ranges" / path.name

        path.mkdir(exist_ok=True, parents=True)

        if genre:
            path = path / genre
            path.mkdir(exist_ok=True)

        fig.write_html(path / "Cluster Plot.html")

        collage_samples = [df.loc[df["cluster"] == cluster] for cluster in range(n_cluster)]
        # collage_samples = [data.sample(min(len(data), 60)).loc[:, , "label"] for data in collage_samples]
        collage_samples = [data.sort_values("distance_to_center").loc[:, "label"].iloc[:60] for data in
                           collage_samples]
        collage_samples = [list(data.apply(lambda x: get_image_path_by_imdb_id(x, use_vitpose_image=True))) for
                           data
                           in
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
            "Years": 'all years' if not year_batch else ''.join([str(y) for y in year_batch]),
            "Median Confidence Lower Bound": median_confidence,
            "Minimum Confidence Lower Bound": min_confidence,
            "Silhouette Score": silhouette_score(feature_set, labels, metric='euclidean'),
            "Calinski Harabasz Score": calinski_harabasz_score(feature_set, labels),
            "Davis Bouldin Score": davies_bouldin_score(feature_set, labels),
            "Variance of Higher Dimensional Data": feature_set.var(),
            "Variance of 2D Data": angles_2d.var(),
            "% Lost Variance": 1 - angles_2d.var() / feature_set.var(),
            "Genre": genre,
            "Arm Ranges": arm_ranges,
            "Knee Ranges": knee_ranges
        }, index=[1])

        pd.concat([
            pd.read_csv(EVAL_PATH, sep=";", decimal=","),
            result
        ], ignore_index=True).to_csv(EVAL_PATH, sep=";", decimal=",", index=False)

        sex_cluster_representation = df.groupby(["cluster", "sex"]).count().reset_index().pivot(index="cluster",
                                                                                                columns="sex",
                                                                                                values="label")
        sex_cluster_representation = sex_cluster_representation.apply(lambda x: x.F / (x.M + x.F),
                                                                      axis=1).sort_values()

        df.loc[:, "Identifier"] = identifier
        df.loc[:, "Arm Ranges"] = arm_ranges
        df.loc[:, "Knee Ranges"] = knee_ranges
        df.loc[:, "Genre"] = genre
        df.loc[:, "Genre"] = bool(genres)
        df.loc[:, "Years"] = bool(year_batch)
        df.loc[:, "Median Confidence Lower Bound"] = median_confidence
        df.loc[:, "Minimum Confidence Lower Bound"] = min_confidence
        df.loc[:, "Cluster N"] = n_cluster
        df.loc[:, "Clustering"] = cluster_name
        df.loc[:, "Body Connections"] = body_connection_name
        df.loc[:, "Dimnesion Reduction Algorithm"] = dim_red_algorithm
        df.loc[:, "Silhouette Score"] = silhouette_score(feature_set, labels, metric='euclidean')
        df.loc[:, "Calinski Harabasz Score"] = calinski_harabasz_score(feature_set, labels)
        df.loc[:, "Davis Bouldin Score"] = davies_bouldin_score(feature_set, labels)
        df.loc[:, "Variance of Higher Dimensional Data"] = feature_set.var()
        df.loc[:, "Variance of 2D Data"] = angles_2d.var()
        df.loc[:, "% Lost Variance"] = 1 - angles_2d.var() / feature_set.var()
        df.loc[:, "n"] = len(df)

        df = df.join(pd.DataFrame(sex_cluster_representation, columns=["Women_Percentage"]), on="cluster")
        df.to_pickle(path / "data.pickle")
        pd.concat([
            pd.read_pickle(RESULTS_PATH / "data.pickle"),
            df.drop_duplicates(subset=["Identifier", "cluster"], keep="first")
        ], ignore_index=True).to_pickle(RESULTS_PATH / "data.pickle")

        print(i + 1, "/", len(combinations), " - ", identifier, ":", len(pose_data), "samples")
        # print(sex_cluster_representation)
