import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

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

    pose_data = pd.read_pickle("data/output/VitPose_confidences.pkl")

    pose_data_2022 = [el["points"] for el in pose_data[2022] if np.median(el["confidence"]) > 0.5 and el["confidence"].min() > 0.15]

    # pose_example = pose_data_2022[0]
    # pose_example = pd.read_pickle("data/output/VitPose/2022/tt2634732.pkl")["points"]

    body_connections = [[0, 1], [0, 2], [1, 3], [2, 4], [5, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12],
                        [11, 12], [11, 13], [13, 15], [12, 14], [14, 16]]

    # image = plt.imread("data/imdb_image_data_sampled/2022/tt2634732.jpg")

    # plot lines between keypoints on image
    angles = []
    for pose in pose_data_2022:

        angle_set = []
        for p1, p2 in body_connections:
            x, y = 0, 1
            vec_x = [pose[p1][x], pose[p2][x]]
            vec_y = [pose[p1][y], pose[p2][y]]

            # calculate angle between two vectors
            angle = calculate_angle(np.array(vec_x), np.array(vec_y))
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
    kmeans = KMeans(n_clusters=10, random_state=0).fit(angles)

    # get cluster labels
    labels = kmeans.labels_

    # get cluster centers
    centers = kmeans.cluster_centers_

    # get cluster sizes
    cluster_sizes = np.unique(labels, return_counts=True)[1]

    """
    START DIMENSIONALITY REDUCTION
    """

    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA

    # reduce dimensionality
    # tsne = TSNE(n_components=2, random_state=0)
    # angles = tsne.fit_transform(angles)

    pca = PCA(n_components=2)
    angles = pca.fit_transform(angles)

    """
    START PLOTTING
    """

    # plot angles
    plt.scatter(angles[:, 0], angles[:, 1], s=20, c=labels)
    plt.show()

