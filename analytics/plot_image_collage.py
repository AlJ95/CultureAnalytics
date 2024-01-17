"""
This module contains functions for plotting image collages.
"""
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List


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
