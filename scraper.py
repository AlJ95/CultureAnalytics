"""SCRAPING THE WEBSITE VINTED.DE FOR CLOTHES AND ACCESSORIES"""
import pandas as pd
import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By

IMG_CLASS = "ipc-image"
TITLE_CLASS = "hero__primary-text"
INFORMATION_CLASS = ".ipc-metadata-list.ipc-metadata-list--dividers-all.title-pc-list.ipc-metadata-list--baseAlt"


class VintedScraper:
    """Class for scraping the website vinted.de"""
    scraper_instances = 0

    BASE_URL = "https://www.imdb.com/title/{imdb_id}/"

    def __init__(self, imdb_ids: pd.Series, year: int, **kwargs):
        self.imdb_ids = imdb_ids
        self.year = year
        self.kwargs = kwargs
        self.count = 0
        self.data = []

    def scrape_all_information(self, debug=False):
        """Performs a full market research"""
        driver = self.init_driver()
        self.iter_pages(driver)
        self.save_data()

        if not debug:
            driver.quit()

    def iter_pages(self, driver):
        """Iterates over the pages"""
        for tid in self.imdb_ids.values:
            try:
                self.get_page(driver, tid)
                self.data.append(self.get_information(driver, tid))
                self.count += 1
            except Exception as e:
                self.save_data()
                driver.quit()
                print(f"Error for {tid}: {e}")
                raise e

    def init_driver(self):
        """Initializes the driver"""
        driver = webdriver.Firefox()

        # smallest size and position the window
        driver.set_window_size(480, 539)

        position = [(x, y) for _ in range(20) for x in range(0, 1920, 480) for y in range(0, 1080, 539)]

        driver.set_window_position(*position[VintedScraper.scraper_instances])

        VintedScraper.scraper_instances += 1

        return driver

    def get_page(self, driver, tid):
        """Gets the page"""
        if self.count % 50 == 0:
            print(f"{self.year}: {int(round(self.count / len(self.imdb_ids) * 100))}% done")
        driver.get(VintedScraper.BASE_URL.format(imdb_id=tid))
        driver.implicitly_wait(3)

    def get_information(self, driver, tid):
        """Gets the items"""
        # get full page
        html = driver.page_source

        with open(f"imdb_html_data_sampled/{tid}.html", "w", encoding="utf-8") as f:
            f.write(html)

        try:
            title = driver.find_element(By.CLASS_NAME, TITLE_CLASS).get_attribute("innerHTML")
        except selenium.common.exceptions.NoSuchElementException:
            title = None

        # get movie cover url
        try:
            image = driver.find_element(By.CLASS_NAME, IMG_CLASS)
            image_url = image.get_attribute("src")
            image_alt = image.get_attribute("alt")
            image_srcset = image.get_attribute("srcset")
            success_image_information = bool(image_url) and bool(image_alt) and bool(image_srcset)
        except:
            image_url = None
            image_alt = None
            image_srcset = None
            success_image_information = False

        try:
            information = driver.find_element(By.CSS_SELECTOR, INFORMATION_CLASS).get_attribute("innerHTML")

            with open(f"imdb_html_data_sampled/{tid}_information.html", "w", encoding="utf-8") as f:
                f.write(information)
        except selenium.common.exceptions.NoSuchElementException:
            information = None

        return pd.Series({
            "imdb_id": tid,
            "title": title,
            "image_url": image_url,
            "image_alt": image_alt,
            "image_srcset": image_srcset,
            "success_image_information": success_image_information,
            "information": information
        })

    def get_data(self):
        """Gets the data"""
        return pd.DataFrame(self.data)

    def save_data(self):
        """Saves the data"""
        self.get_data().to_csv(f"./imdb_image_urls_sampled/{self.year}.csv", encoding="utf-8", sep="\t")


if __name__ == '__main__':
    data = pd.read_csv("./imdb_data_post_processed/us_title.akas.csv", encoding="utf-8")

    irobot = data.loc[data.titleId == "tt0343818"]

    imdb_ids = irobot.titleId
    year = irobot.startYear.values[0]

    scraper = VintedScraper(imdb_ids, year)
    scraper.scrape_all_information()
