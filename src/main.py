from src.data_tools import (
    FactoryLoader,
    get_batches_generator,
    get_categorical_cardinalities,
    get_records_cube_from_df,
    shuffle_multiple,
    recarray_to_array
)

from src.constants import (
    numeric_feats,
    categorical_feats,
    embedding_sizes,
    batch_time_normalizable_feats,
)

from src.tensorflow_tools import start_tensorflow_session, get_summary_writer
from src.common_paths import get_tensorboard_path
import os
import numpy as np

import tensorflow as tf
from src.architecture import Seq2Seq


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
SAMPLE = False

if __name__ == "__main__":
    # Load data
    df_master = FactoryLoader().load("master", sample=SAMPLE)
    df_master = get_records_cube_from_df(df=df_master)
    cat_cardinalities_time = {
        col: len(np.unique(df_master[col]))
        for col in df_master.dtype.names
        if col in categorical_feats
    }

    df_master_static = FactoryLoader().load("master_timeless", sample=SAMPLE)
    df_master_static = df_master_static.to_records()
    cat_cardinalities_timeless = {
        col: len(np.unique(df_master_static[col]))
        for col in df_master_static.dtype.names
        if col in categorical_feats
    }

    cat_cardinalities = dict(
        list(cat_cardinalities_time.items()) + list(cat_cardinalities_timeless.items())
    )

    batcher = get_batches_generator(
        df_time=df_master, df_static=df_master_static, batch_size=128, shuffle=True
    )

    c=0
    for numeric_time_batch, cat_time_batch, cat_static_batch, target in batcher:
        c+=1
