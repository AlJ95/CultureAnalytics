"""
This script is used to make requests to the Amazon API in parallel.
"""
import multiprocessing as mp
from loader.B1_omdb_api import make_amazon_requests


if __name__ == '__main__':
    pool = mp.Pool(mp.cpu_count())
    pool.map(make_amazon_requests, range(1980, 2023))