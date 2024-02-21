import multiprocessing as mp
from pathlib import Path
from nudenet_script import guess_sex

if __name__ == '__main__':
    print(mp.cpu_count())
    pool = mp.Pool(mp.cpu_count())

    sub_paths = [sp for sp in Path(f"../data/output/nudenet/raw_images").iterdir() if sp.suffix == ".jpg"]

    chunk_size = 1000
    s = len(sub_paths) // chunk_size

    sb_batches = [[p for p in sub_paths[c::s]] for c in range(s)]

    pool.map(guess_sex, sb_batches)
