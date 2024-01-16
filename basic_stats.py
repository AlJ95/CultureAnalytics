import os
import PIL
import matplotlib.pyplot as plt
import pandas as pd
from loader.omdb_api import make_request
from utils.helper import IMDB_IMAGE_DATA_SAMPLED_PATH, IMDB_DATA_POST_PROCESSED_PATH


def save_downloaded_image(start_year: int):
    sampled_data = pd.read_csv(IMDB_IMAGE_DATA_SAMPLED_PATH / "sampled_1800.csv", encoding="utf-8", sep="\t",
                               low_memory=False)
    sampled_data = sampled_data.loc[sampled_data.startYear.astype(int) == start_year].titleId.values

    if not (IMDB_IMAGE_DATA_SAMPLED_PATH / str(start_year)).exists():
        os.mkdir(IMDB_IMAGE_DATA_SAMPLED_PATH / str(start_year))

    count = 0
    for tid in sampled_data:
        try:
            image = make_request(tid)
            image.save(str(IMDB_IMAGE_DATA_SAMPLED_PATH / str(start_year) / f"{tid}.jpg"))
            count += 1
        except PIL.UnidentifiedImageError:
            print(f"Could not download image for {tid}")

        if count % 100 == 0:
            print(f"Downloaded {count} images for year {start_year}")

        if count == 1000:
            break


def sample_data():
    data = pd.read_csv(IMDB_DATA_POST_PROCESSED_PATH / "us_title.akas.csv", encoding="utf-8")

    data = data.loc[(data.startYear != "\\N") & (~pd.isna(data.startYear))]
    data = data.loc[(data.titleType == "movie")]
    data = data.loc[data.startYear.astype(int) >= 1980]
    data = data.loc[data.startYear.astype(int) <= 2023]
    data = data.loc[data.isAdult == 0]

    plt.hist(data.startYear.astype(int), bins=100)
    plt.show()

    # sample 1000 movies from each year from 1980 to 2023
    sampled_data = pd.DataFrame()
    for year in range(1980, 2023):
        sampled_data = pd.concat([sampled_data, data.loc[data.startYear.astype(int) == year].sample(1300)])

    sampled_data.to_csv(IMDB_IMAGE_DATA_SAMPLED_PATH / "sampled_1800.csv", encoding="utf-8", sep="\t")

    for year in range(1980, 2023):
        # count images in folder
        print(f"{year}: {len(os.listdir(IMDB_IMAGE_DATA_SAMPLED_PATH / str(year)))}")


if __name__ == '__main__':
    pass
    # sample_data()
