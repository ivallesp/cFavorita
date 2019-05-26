from src.data_tools import FactoryLoader, get_batcher_generator, get_categorical_cardinalities
from src.constants import numeric_feats, categorical_feats, embedding_sizes

import os
import gc

import tensorflow as tf
from src.architecture import Seq2Seq


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


if __name__ == "__main__":

    fl = FactoryLoader()
    df = fl.load("master", sample=False)

    print("Data sorted successfully!")
    colnames = df.columns.values
    shape = df.store_nbr.nunique()*df.item_nbr.nunique(), df.date.nunique(), df.shape[1]
    print("Data transformed to numpy successfully!")
    gc.collect()
    df = df.to_numpy()
    gc.collect()
    df = df.reshape(shape)
    gc.collect()
    print("Data reshaped successfully!")


    categorical_cardinalities = get_categorical_cardinalities(data_cube=df, categorical_feats=categorical_feats,
                                                              colnames=colnames)
    model = Seq2Seq(n_numerical_features=len(numeric_feats),
                    categorical_cardinalities=categorical_cardinalities,
                    embedding_sizes=embedding_sizes, n_output_ts=30)
    print("Model built successfully!")

    batcher = get_batcher_generator(data_cube=df, model=model, batch_size=128, colnames=colnames)

    for batch in batcher:
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        loss, _ = sess.run([model.losses.loss_mse, model.optimizers.op], batch)
        print(loss)
