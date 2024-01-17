import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from utils.helper import get_all_vitpose_pickle_paths, RESULTS_PATH

labels = {0: 'nose', 1: 'left_eye', 2: 'right_eye', 3: 'left_ear', 4: 'right_ear', 5: 'left_shoulder',
          6: 'right_shoulder', 7: 'left_elbow', 8: 'right_elbow', 9: 'left_wrist',
          10: 'right_wrist', 11: 'left_hip',
          12: 'right_hip', 13: 'left_knee',
          14: 'right_knee', 15: 'left_ankle',
          16: 'right_ankle'}


def calculate_angle(vec1, vec2):
    """
    Calculate the angle between two vectors.
    :param vec1: vector 1
    :param vec2: vector 2
    :return: angle in degrees
    """
    angle = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
    return np.degrees(angle)


if __name__ == '__main__':

    # body_connections = [[0, 1], [0, 2], [1, 3], [2, 4], [5, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12],
    #                     [11, 12], [11, 13], [13, 15], [12, 14], [14, 16]]
    body_connections = [[i, j] for i in range(17) for j in range(17) if i < j]

    pickle_paths = get_all_vitpose_pickle_paths()
    pose_data_raw = {pickle_path.stem: pd.read_pickle(pickle_path) for pickle_path in pickle_paths}
    pose_data = {key: pose["points"] for key, pose in pose_data_raw.items()
                 if np.median(pose["confidence"]) > 0.75 and pose["confidence"].min() > 0.1
                 and (~np.isnan(pose["points"])).all()}

    # plot lines between keypoints on image
    angles = []
    for pose in pose_data.values():

        angle_set = []
        for p1, p2 in body_connections:
            x, y = 0, 1
            vec_x = [pose[p1][x], pose[p2][x]]
            vec_y = [pose[p1][y], pose[p2][y]]

            # calculate angle between two vectors
            angle = calculate_angle(np.array(vec_x), np.array(vec_y))

            if np.isnan(angle):
                angle = 0

            angle_set.append(angle)

            # plt.plot(vec_x, vec_y, linewidth=2, marker='o',
            #          markersize=0.5, markerfacecolor='red', markeredgecolor='red')

        # plot keypoints on image
        # plt.scatter(pose_example[:, 0], pose_example[:, 1], s=20, c='r')

        # plt.imshow(image)
        # plt.show()

        angles.append(angle_set)

    """
    START K-MEANS
    """

    # convert angles to numpy array
    angles = np.array(angles)

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
    collage_samples = [list(data.apply(lambda x: get_image_path_by_imdb_id(x, use_vitpose_image=True))) for data in collage_samples]

    input("Du wirst gleich alte Speicherstände überschreiben. Drücke Enter zum Fortfahren.")
    for i, cluster in enumerate(collage_samples):
        print(f"Cluster {i} has {len(cluster)} samples")
        plot_image_collage(cluster, f"cluster_{i}", RESULTS_PATH / "tsne cluster all years n=16 angle cross product all body parts")
