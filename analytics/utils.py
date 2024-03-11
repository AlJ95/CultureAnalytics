import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List


def calculate_angle(vec1, vec2):
    """
    Calculate the angle between two vectors.
    :param vec1: vector 1
    :param vec2: vector 2
    :return: angle in degrees
    """
    angle = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
    return 0 if np.isnan(np.degrees(angle)) else np.degrees(angle)


def calculate_unit_circle_pos(vec1, vec2):
    angle = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

    angle = np.array([np.cos(angle), np.sin(angle)])

    if any(np.isnan(angle)):
        angle = np.zeros((2,))

    return angle


def plot_image_collage(image_paths: List[Path], name: str, path: Path):
    """
    Plot image collage in 2 rows and 5 columns.
    :param path: Path to save image collage
    :param image_paths: List of image paths
    :param name: Name of the image collage to be saved
    """

    if len(image_paths) > 60:
        raise ValueError("Only 60 images can be plotted in one row")

    fig, axs = plt.subplots(6, 10, figsize=(40, 30))
    for i, ax in enumerate(axs.flat):
        if i < len(image_paths):
            ax.imshow(plt.imread(image_paths[i]))
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(path / f"{name}.png")
