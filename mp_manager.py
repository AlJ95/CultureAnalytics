import multiprocessing as mp
from load import make_amazon_request


if __name__ == '__main__':
    pool = mp.Pool(9)
    pool.map(make_amazon_request, range(1980, 2023))