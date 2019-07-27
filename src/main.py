from src.data_tools import FactoryLoader, get_batcher_generator, get_categorical_cardinalities
from src.constants import numeric_feats, categorical_feats, embedding_sizes
from src.tensorflow_tools import start_tensorflow_session, get_summary_writer
from src.common_paths import get_tensorboard_path
import os
import gc

import tensorflow as tf
from src.architecture import Seq2Seq


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


if __name__ == "__main__":

    df_master_time = FactoryLoader().load("master")
    df_master_timeless = FactoryLoader().load("master_timeless")

    print("Data sorted successfully!")
    colnames_time = df_master_time.columns.values
    shape = df_master_time.store_nbr.nunique()*df_master_time.item_nbr.nunique(), df_master_time.date.nunique(), df_master_time.shape[1]
    print("Data transformed to numpy successfully!")
    gc.collect()
    df = df_master_time.to_numpy()
    gc.collect()
    df = df.reshape(shape)
    gc.collect()
    colnames_timeless = df_master_timeless.columns.values
    df_timeless = df_master_timeless.to_numpy()
    print("Data reshaped successfully!")


    c = 0

    categorical_cardinalities = get_categorical_cardinalities(data_cube=df, categorical_feats=categorical_feats,
                                                              colnames=colnames_time)
    model = Seq2Seq(n_numerical_features=len(numeric_feats),
                    categorical_cardinalities=categorical_cardinalities,
                    embedding_sizes=embedding_sizes, n_output_ts=30)
    print("Model built successfully!")

    sess = start_tensorflow_session()
    sw = get_summary_writer(sess, get_tensorboard_path(), "CF", "V0")
    sess.run(tf.global_variables_initializer())
    train_df = df[:,:1000]
    dev_df = df[:, (1000-380):(1000+60)]


    for epoch in range(100):
        batcher = get_batcher_generator(data_cube_time=train_df, data_cube_timeless=df_timeless,
                                        model=model, batch_size=128, colnames_time=colnames_time,
                                        history_window_size=380, prediction_window_size=30)
        batcher_test = get_batcher_generator(data_cube_time=dev_df, data_cube_timeless=df_timeless,
                                             model=model, batch_size=128, colnames_time=colnames_time,
                                             history_window_size=380, prediction_window_size=30)



        for batch in batcher:
            c += 1
            loss, _, train_summary = sess.run([model.losses.loss_mse,
                                               model.optimizers.op,
                                               model.summ.scalar_train_performance], batch)
            sw.add_summary(train_summary, c)
            print(loss)

    # TODO: Implement TRAIN Mape in tensorboard
    # TODO: Implement TEST Mape in tensorboard
