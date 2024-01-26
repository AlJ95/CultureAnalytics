import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from utils.helper import get_all_vitpose_pickle_paths, RESULTS_PATH
from angle import calculate_unit_circle_pos
from const import *


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


if __name__ == '__main__':

    pickle_paths = get_all_vitpose_pickle_paths()
    pose_data_raw = {pickle_path.stem: pd.read_pickle(pickle_path) for pickle_path in pickle_paths}
    pose_data = filter_pose_data(pose_data_raw)

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

        angles.append(angle_set)

    """
    START K-MEANS
    """

    # convert angles to numpy array
    if isinstance(angle, np.float32):
        angles = np.array(angles)
    else:
        angles = np.array([[x for y in angle_set for x in y] for angle_set in angles])

    # create k-means model
    kmeans = KMeans(n_clusters=16, random_state=0).fit(angles)

    # get cluster labels
    labels = kmeans.labels_

    # get cluster centers
    centers = kmeans.cluster_centers_

    # get cluster sizes
    cluster_sizes = np.unique(labels, return_counts=True)[1]

    """
    START DIMENSIONALITY REDUCTION & PLOTTING
    """

    from sklearn.decomposition import PCA

    # reduce dimensionality
    pca = PCA(n_components=2)
    angles_PCA = pca.fit_transform(angles)

    # plot angles
    plt.scatter(angles_PCA[:, 0], angles_PCA[:, 1], s=20, c=labels)
    plt.show()

    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=2, random_state=0)
    angles_TSNE = tsne.fit_transform(angles)

    # plot angles
    plt.scatter(angles_TSNE[:, 0], angles_TSNE[:, 1], s=20, c=labels)
    plt.show()

    """
    Use Plotly for annotated scatter plot
    """
    import plotly.express as px
    import plotly.graph_objects as go

    # create dataframe
    df = pd.DataFrame(angles_TSNE, columns=["x", "y"])
    df["label"] = pose_data.keys()
    df["cluster"] = labels
    df["cluster"] = df.cluster.astype("category")
    df["cluster_size"] = cluster_sizes[df["cluster"].cat.codes]

    # create figure
    fig = px.scatter(df,
                     x="x",
                     y="y",
                     color="cluster",
                     color_discrete_sequence=px.colors.qualitative.Plotly,
                     symbol="cluster",
                     hover_data=["label", "cluster_size"]
                     )

    fig.show()

    from analytics.plot_image_collage import plot_image_collage
    from utils.helper import get_image_path_by_imdb_id
    from pathlib import Path

    collage_samples = [df.loc[df["cluster"] == cluster, "label"] for cluster in range(16)]
    collage_samples = [data.sample(min(len(data), 60)) for data in collage_samples]
    collage_samples = [list(data.apply(lambda x: get_image_path_by_imdb_id(x, use_vitpose_image=True))) for data in
                       collage_samples]

    input("Du wirst gleich alte Speicherstände überschreiben. Drücke Enter zum Fortfahren.")
    for i, cluster in enumerate(collage_samples):
        print(f"Cluster {i} has {len(cluster)} samples")
        plot_image_collage(cluster, f"cluster_{i}",
                           RESULTS_PATH / "tsne cluster all years n=16 unit circle pos rel body parts min_conf 0 25")
