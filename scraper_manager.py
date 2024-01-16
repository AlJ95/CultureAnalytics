import multiprocessing as mp
from pathlib import Path

import pandas as pd

from scraper import VintedScraper


def start_scraper(scraper: VintedScraper):
    scraper.scrape_all_information()


if __name__ == '__main__':

    cpu_count = mp.cpu_count()
    pool = mp.Pool(3)

    years = list(range(1980, 2023))

    sampled_data = pd.read_csv("./imdb_image_data_sampled/sampled_1800.csv", encoding="utf-8", sep="\t")

    downloaded_images = [x.stem for path in Path("imdb_image_data_sampled").iterdir() if path.is_dir()
                         for x in path.iterdir() if path.is_dir()
                         if x.is_file()]

    imdb_ids = {year:
        sampled_data.loc[(sampled_data.startYear == year) & (~sampled_data.titleId.isin(downloaded_images))]
        .titleId.values for year in years}

    scrapers = [VintedScraper(imdb_ids=pd.Series(imdb_ids[year]), year=year) for year in years]

    pool.map(start_scraper, scrapers)
