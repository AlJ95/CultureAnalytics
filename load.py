import io
import os
from pathlib import Path
import pandas as pd
from PIL import Image
import PIL
import requests
from dotenv import load_dotenv

omdb_url = "https://img.omdbapi.com/?apikey={key}&i={imdb_id}&plot=full"


def make_request(imdb_id) -> Image:
    """
    get request for an image from omdb
    :param imdb_id: imdb id of the movie
    :return: image
    :rtype:
    """
    load_dotenv()

    key = os.getenv("OMDB_KEY")
    url = omdb_url.format(key=key, imdb_id=imdb_id)

    # response is an image
    response = requests.get(url)

    image_stream = io.BytesIO(response.content)
    # convert to PIL image
    image = Image.open(image_stream)
    return image

def make_amazon_request(year: int):
    """
    get request for an image from amazon
    :param data: dataframe containing the data
    :return: image
    :rtype:
    """
    if not os.path.exists(f"imdb_image_urls_sampled/{year}.csv"):
        return

    data = pd.read_csv(f"imdb_image_urls_sampled/{year}.csv", encoding="utf-8", sep="\t").reset_index(drop=True)
    data.loc[:, "image_downloaded"] = False

    for index, row in data.iterrows():

        if row["image_downloaded"]:
            continue

        try:
            response = requests.get(row["image_url"])

            image_stream = io.BytesIO(response.content)
            # convert to PIL image
            image = Image.open(image_stream)
            image.save(f"./imdb_image_data_sampled/{year}/{row['imdb_id']}.jpg")
            data.loc[index, "image_downloaded"] = True
        except (PIL.UnidentifiedImageError, requests.exceptions.MissingSchema):
            print(f"Could not download image for {row['imdb_id']}")

        if index % 100 == 0:
            print(f"Downloaded {index} images for year {year}")

    data.to_csv(f"imdb_image_urls_sampled/{year}.csv", encoding="utf-8", sep="\t")


if __name__ == '__main__':
    # data = pd.read_csv("./imdb_data_post_processed/us_title.akas.csv", encoding="utf-8", nrows=1000)
    #
    # for index, row in data.iterrows():
    #     try:
    #         image = make_request(row["titleId"])
    #         image.save(f"./test_images/{row['titleId']}.jpg")
    #     except PIL.UnidentifiedImageError:
    #         print(f"Could not download image for {row['titleId']}")

    sample_path = Path("imdb_image_data_sampled")
    downloaded_images = [x for path in sample_path.iterdir() if path.is_dir()
                         for x in path.iterdir() if path.is_dir()
                         if x.is_file()]

    image_ids = [x.stem for x in downloaded_images]

    sample_data = pd.read_csv("imdb_image_data_sampled/sampled_1800.csv", encoding="utf-8", sep="\t")

    sample_data_missing = sample_data.loc[~sample_data.titleId.isin(image_ids)]

    for index, row in sample_data_missing.iterrows():
        try:
            image = requests.get(f"https://www.imdb.com/title/{row['titleId']}/")
        except PIL.UnidentifiedImageError:
            print(f"Could not download image for {row['titleId']}")