import numpy as np
import matplotlib.pyplot as plt
from utils.helper import OUTPUT_PATH
import pandas as pd

lowest_numbers_of_images_per_year = [100, 200, 250, 300]

conf_df = pd.read_csv(OUTPUT_PATH / "VitPose_nr_of_images_by_confidence.csv", index_col=0)
conf_df.columns = [f"{c}%" for c in conf_df.columns]
conf_df.index = [f"{year - year % 5}-{year - (year % 5) + 4}" for year in conf_df.index]
conf_df = conf_df.groupby(conf_df.index).sum()
conf_df = conf_df.iloc[:, ::2]
conf_df = conf_df.T

# create 4 subplots
fig, ax = plt.subplots(2, 2, figsize=(10, 7))
ax = ax.flatten()

for i, upper in enumerate(lowest_numbers_of_images_per_year):

    im = ax[i].imshow(conf_df.clip(upper=upper).values, cmap="YlGn")

    # We want to show all ticks...
    ax[i].set_xticks(np.arange(len(conf_df.columns)))
    ax[i].set_yticks(np.arange(len(conf_df.index)))
    # ... and label them with the respective list entries
    ax[i].set_xticklabels(conf_df.columns)
    ax[i].set_yticklabels(conf_df.index)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax[i].get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    ax[i].set_title(f"Green = Min {upper} images")

# legend
# cbar = ax.figure.colorbar(im, ax=ax)


title = "Confidence Distribution of Pose Estimation 1980 - 2022"
fig.suptitle(title, fontsize=20)
plt.tight_layout()
plt.savefig(OUTPUT_PATH / f"{title}.png")
plt.show()