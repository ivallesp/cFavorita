import os
import pandas as pd
import sys
import inspect
import gc
import random
import h5py

from tqdm import tqdm
import numpy as np
from src.common_paths import get_data_path
from src.general_utilities import batching
from src.constants import (
    numeric_feats,
    categorical_feats,
    target,
    batch_time_normalizable_feats,
    embedding_sizes,
)
from src.data_tools import (
    FactoryLoader,
    get_batcher_generator,
    get_categorical_cardinalities,
    get_data_cube_from_df,
    shuffle_multiple,
)


def main(sample=False):
    filename="data/cache/master_data.hdf"
    assert os.path.exists(os.path.split(filename)[0])

    df_master = FactoryLoader().load("master", sample=sample)
    df_master.to_hdf(filename, "table")

    df = get_data_cube_from_df(df=df_master)
    with h5py.File(filename, "a") as h:
        h.create_dataset("data_cube", data=df)


if __name__ == "__main__":
    main(False)
