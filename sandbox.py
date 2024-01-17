import pickle

import pandas as pd
from pathlib import Path

from matplotlib import pyplot as plt

from utils.helper import OUTPUT_PATH

data_path_vit = Path('data', 'output', "VitPose")


def gather_and_save_confidence_data():
    """
    Confidence data is saved per image in pickle files.
    This function gathers all the data and saves it in one pickle
    """
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

    pickle.dump(data, open("output/VitPose_confidences.pkl", "wb"))


if __name__ == '__main__':

    data = pickle.load(open(OUTPUT_PATH / "VitPose_confidences.pkl", "rb"))

    print('-------------------')
    print('-------------------')

    conf_df = pd.DataFrame()
    for conf in range(0, 100, 5):
        conf_df = pd.concat([conf_df,
                             pd.DataFrame({year: len([d for i, d in enumerate(v)
                                                      if d["confidence"].min() > conf / 100])
                                           for year, v in data.items()}, index=[conf])])

    print(conf_df.T)

    conf_df.T.to_csv(OUTPUT_PATH / "VitPose_nr_of_images_by_confidence.csv")

    # plot the 2D histogram

    plt.hist2d(conf_df, bins=100)
    plt.show()