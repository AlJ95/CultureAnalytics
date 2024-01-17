from pathlib import Path
from typing import List
from functools import lru_cache

import PIL.Image

DATA_PATH = Path("data")
IMDB_DATA_PATH = Path("data", "imdb_data")
IMDB_IMAGE_DATA_SAMPLED_PATH = Path("data", "imdb_image_data_sampled")
IMDB_HTML_DATA_SAMPLED_PATH = Path("data", "imdb_html_data_sampled")
IMDB_IMAGE_URLS_SAMPLED_PATH = Path("data", "imdb_image_urls_sampled")
IMDB_DATA_POST_PROCESSED_PATH = Path("data", "imdb_data_post_processed")
OUTPUT_PATH = DATA_PATH / "output"
VITPOSE_PATH = OUTPUT_PATH / "VitPose"
RESULTS_PATH = Path("results")


@lru_cache(maxsize=1)
def get_all_file_paths(path: Path, suffix: str | None, year: int | None) -> List[Path]:
    """
    Get all file paths from a directory

    File structure is as follows:
    data
    ├── imdb_data
    │   ├── 1980
    │   │   ├── 0000001.tsv
    │   │   ├── 0000002.tsv
    │   │   ├── ...
    │   ├── 1981
    │   │   ├── 0000001.tsv
    │   │   ├── 0000002.tsv
    │   │   ├── ...
    │   ├── ...
    │   ├── 2022
    │   │   ├── 0000001.tsv
    │   │   ├── 0000002.tsv
    │   │   ├── ...
    ├── imdb_image_data_sampled
    ...
    :param path: Path to directory
    :param suffix: Suffix of file
    :param year: Year of movie release
    :return: List of file paths
    """
    file_paths = [x for path in Path(path).iterdir() if path.is_dir()
                  for x in path.iterdir() if path.is_dir()
                  if x.is_file()]

    if suffix:
        file_paths = [x for x in file_paths if x.suffix == suffix]

    if year:
        file_paths = [x for x in file_paths if x.parent.name == str(year)]

    return file_paths


@lru_cache(maxsize=35)
def get_all_sampled_image_paths(year=None) -> List[Path]:
    return get_all_file_paths(IMDB_IMAGE_DATA_SAMPLED_PATH, ".jpg", year)


@lru_cache(maxsize=35)
def get_all_sampled_html_file_paths(year=None) -> List[Path]:
    return get_all_file_paths(IMDB_HTML_DATA_SAMPLED_PATH, ".html", year)


@lru_cache(maxsize=35)
def get_all_sampled_image_urls_file_paths() -> List[Path]:
    return get_all_file_paths(IMDB_IMAGE_URLS_SAMPLED_PATH, ".csv", None)


@lru_cache(maxsize=35)
def get_all_vitpose_pickle_paths() -> List[Path]:
    return get_all_file_paths(VITPOSE_PATH, ".pkl", None)


@lru_cache(maxsize=35)
def get_all_vitpose_image_paths() -> List[Path]:
    return get_all_file_paths(VITPOSE_PATH, ".jpg", None)


def show_image_by_imdb_id(imdb_id: str) -> Path:
    """
    Show image by imdb id
    :param imdb_id: IMDB id
    :return: Path to image
    """
    img_paths = get_all_sampled_image_paths()

    image_path = [img_path for img_path in img_paths if imdb_id in img_path.name][0]

    return PIL.Image.open(image_path).show()


def get_image_path_by_imdb_id(imdb_id: str, use_vitpose_image=False) -> Path:
    """
    Show image by imdb id
    :param imdb_id: IMDB id
    :return: Path to image
    """
    if use_vitpose_image:
        img_paths = get_all_vitpose_image_paths()
    else:
        img_paths = get_all_sampled_image_paths()

    image_path = [img_path for img_path in img_paths if imdb_id in img_path.name][0]

    return image_path


if __name__ == '__main__':
    print(get_all_sampled_image_paths())
