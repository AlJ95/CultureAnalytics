"""
This script is used to start the scraper for all years in parallel.
It looks for title ids in the sampled data and checks if the data is already downloaded.
If not, it starts the scraper for the year and only downloads the missing data.
"""
import multiprocessing as mp
from pathlib import Path

import pandas as pd

from scraper import IMDBScraper
from utils.helper import IMDB_IMAGE_DATA_SAMPLED_PATH, get_all_sampled_image_paths


def start_scraper(scraper: IMDBScraper):
    scraper.scrape_all_information()


if __name__ == '__main__':
    cpu_count = mp.cpu_count()
    pool = mp.Pool(3)

    years = list(range(1980, 2023))

    sampled_data = pd.read_csv(IMDB_IMAGE_DATA_SAMPLED_PATH / "sampled_1800.csv", encoding="utf-8", sep="\t")

    downloaded_images = [path.stem for path in get_all_sampled_image_paths()]

    imdb_ids = {
        year: sampled_data.loc[
            # year condition and not downloaded yet
            (sampled_data.startYear == year) & (~sampled_data.titleId.isin(downloaded_images))
            ].titleId.values for year in years}

    # initialize scraper for each year
    scrapers = [IMDBScraper(imdb_ids=pd.Series(imdb_ids[year]), year=year) for year in years]

    # start scraper for each year in parallel
    pool.map(start_scraper, scrapers)
