import io
import os
import typing
import pandas as pd
from PIL import Image
import PIL
import requests
from dotenv import load_dotenv
from utils.helper import *

OMDB_URL = "https://img.omdbapi.com/?apikey={key}&i={imdb_id}&plot=full"


def make_request(imdb_id) -> Image:
    """
    get request for an image from omdb
    :param imdb_id: imdb id of the movie
    :return: image
    :rtype:
    """
    load_dotenv()

    key = os.getenv("OMDB_KEY")
    url = OMDB_URL.format(key=key, imdb_id=imdb_id)

    # response is an image
    response = requests.get(url)

    image_stream = io.BytesIO(response.content)
    # convert to PIL image
    image = Image.open(image_stream)
    return image


def download_images_over_all_years(years: typing.Iterable[int]):
    """
    Download all images from imdb without multiprocessing (see. loader.omdb_api_mp.py)
    :param years: years to download
    """
    for year in years:
        make_amazon_requests(year, False)
        print(f"Downloaded all images for year {year}")


def make_amazon_requests(year: int, verbose=True):
    """
    There are a lot of urls gathered from IMDB that have to be downloaded from amazon, because
    IMDB has all the images on amazon. This function downloads all the images from amazon and keeps track of
    which images have been downloaded so far.
    :param year: data
    :param verbose: print progress
    """
    if not os.path.exists(IMDB_IMAGE_URLS_SAMPLED_PATH / f"{year}.csv"):
        return

    data = pd.read_csv(IMDB_IMAGE_URLS_SAMPLED_PATH / f"{year}.csv", encoding="utf-8", sep="\t").reset_index(drop=True)
    data.loc[:, "image_downloaded"] = False

    for index, row in data.iterrows():

        if row["image_downloaded"]:
            continue

        try:
            response = requests.get(row["image_url"])

            image_stream = io.BytesIO(response.content)
            # convert to PIL image
            image = Image.open(image_stream)
            image.save(IMDB_IMAGE_DATA_SAMPLED_PATH / str(year) / f"{row['imdb_id']}.jpg")
            data.loc[index, "image_downloaded"] = True
        except (PIL.UnidentifiedImageError, requests.exceptions.MissingSchema):
            print(f"Could not download image for {row['imdb_id']}")

        if (index % 100 == 0) and verbose:
            print(f"Downloaded {index} images for year {year}")

    data.to_csv(IMDB_IMAGE_URLS_SAMPLED_PATH / f"{year}.csv", encoding="utf-8", sep="\t")


if __name__ == '__main__':

    downloaded_images = get_all_sampled_image_paths()
    image_ids = [x.stem for x in downloaded_images]

    sample_data = pd.read_csv(IMDB_IMAGE_DATA_SAMPLED_PATH/ "sampled_1800.csv", encoding="utf-8", sep="\t")

    sample_data_missing = sample_data.loc[~sample_data.titleId.isin(image_ids)]

    print(sample_data_missing)
