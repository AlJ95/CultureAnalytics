import pickle

import pandas as pd
from pathlib import Path

data_path_vit = Path('data', 'output', "VitPose")


if __name__ == '__main__':
    data = {}
    for year in range(1980, 2023):
        print('-------------------')
        print(year)
        year_path = data_path_vit / str(year)

        data[year] = []
        for i, file in enumerate(year_path.glob('*.pkl')):
            data[year].append(pd.read_pickle(file))

            if i % 1000 == 0:
                print(i)

    print('-------------------')
    print('-------------------')

    for conf in range(0, 100, 5):
        nr_of_images = len([f"{k}-{i}"
                            for k, v in data.items()
                            for i, d in enumerate(v)
                            if d["confidence"].min() > conf/100])

        print(f"Confidence: {conf}% - {nr_of_images} images")


    pickle.dump(data, open("output/VitPose_confidences.pkl", "wb"))