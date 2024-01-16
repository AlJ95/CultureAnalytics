import multiprocessing as mp
import os

import PIL
import matplotlib.pyplot as plt
import pandas as pd
from load import make_request

def save_downloaded_image(year):
    data = pd.read_csv("imdb_image_data_sampled/sampled_1800.csv", encoding="utf-8", sep="\t",
                       low_memory=False)
    data = data.loc[data.startYear.astype(int) == year].titleId.values

    if not os.path.exists(f"imdb_image_data_sampled/{year}"):
        os.mkdir(f"imdb_image_data_sampled/{year}")

    count = 0
    for tid in data:
        try:
            image = make_request(tid)
            image.save(f"./imdb_image_data_sampled/{year}/{tid}.jpg")
            count += 1
        except PIL.UnidentifiedImageError:
            print(f"Could not download image for {tid}")

        if count % 100 == 0:
            print(f"Downloaded {count} images for year {year}")

        if count == 1000:
            break


if __name__ == '__main__':
    data = pd.read_csv("./imdb_data_post_processed/us_title.akas.csv", encoding="utf-8")

    data = data.loc[(data.startYear != "\\N") & (~pd.isna(data.startYear))]
    data = data.loc[(data.titleType == "movie")]
    data = data.loc[data.startYear.astype(int) >= 1980]
    data = data.loc[data.startYear.astype(int) <= 2023]
    data = data.loc[data.isAdult == 0]


    plt.hist(data.startYear.astype(int), bins=100)
    plt.show()

    # sample 1000 movies from each year from 1980 to 2023
    sample_data = pd.DataFrame()
    for year in range(1980, 2023):
        sample_data = pd.concat([sample_data, data.loc[data.startYear.astype(int) == year].sample(1300)])

    sample_data.to_csv("./imdb_image_data_sampled/sampled_1800.csv", encoding="utf-8", sep="\t")


    # pool = mp.Pool(mp.cpu_count())
    # pool.map(save_downloaded_image, range(1980, 2023))

    for year in range(1980, 2023):
        # count images in folder
        print(f"{year}: {len(os.listdir(f'imdb_image_data_sampled/{year}'))}")
