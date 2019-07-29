from src.data_tools import FactoryLoader, get_batcher_generator, get_categorical_cardinalities, get_data_cube_from_df, shuffle_multiple
from src.constants import numeric_feats, categorical_feats, embedding_sizes
from src.tensorflow_tools import start_tensorflow_session, get_summary_writer
from src.common_paths import get_tensorboard_path
import os
import gc
import numpy as np
import datetime

import tensorflow as tf
from src.architecture import Seq2Seq


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


if __name__ == "__main__":

    df_master_time = FactoryLoader().load("master", sample=False)
    cat_cardinalities_time = {col: df_master_time[col].nunique() for col in df_master_time.columns if
                              col in categorical_feats}
    colnames_time = df_master_time.columns.values
    df = get_data_cube_from_df(df=df_master_time)

    df_master_timeless = FactoryLoader().load("master_timeless", sample=False)
    cat_cardinalities_timeless = {col: df_master_timeless[col].nunique() for col in df_master_timeless.columns if
                              col in categorical_feats}
    colnames_timeless = df_master_timeless.columns.values
    df_timeless = df_master_timeless.to_numpy()

    cat_cardinalities = dict(list(cat_cardinalities_time.items()) + list(cat_cardinalities_timeless.items()))

    print( datetime.datetime.now().isoformat(), "Shuffling...")
    df, df_timeless = shuffle_multiple(df, df_timeless)
    print( datetime.datetime.now().isoformat(), "Shuffle successful!")

    c = 0

    model = Seq2Seq(n_numerical_features=len(numeric_feats),
                    categorical_cardinalities=cat_cardinalities,
                    embedding_sizes=embedding_sizes,
                    n_output_ts=30)

    print("Model built successfully!")

    sess = start_tensorflow_session()
    sw = get_summary_writer(sess, get_tensorboard_path(), "CF", "V0")
    sess.run(tf.global_variables_initializer())
    train_df = df[:,:1000]
    dev_df = df[:, (1000-380):(1000+60)]


    for epoch in range(10000):
        print("Epoch", epoch)
        batcher = get_batcher_generator(data_cube_time=train_df, data_cube_timeless=df_timeless,
                                        model=model, batch_size=128, colnames_time=colnames_time,
                                        colnames_timeless=colnames_timeless,
                                        history_window_size=380, prediction_window_size=30)
        batcher_test = get_batcher_generator(data_cube_time=dev_df, data_cube_timeless=df_timeless,
                                             model=model, batch_size=128, colnames_time=colnames_time,
                                             colnames_timeless=colnames_timeless,
                                             history_window_size=380, prediction_window_size=30)
        losses = []
        target_sum = 0
        pred_sum = 0
        for batch, stats in batcher_test:
            loss, pred = sess.run([model.losses.loss_mse, model.core_model.output], batch)
            losses.append(loss)
            target = batch[model.ph.target]*stats[1] + stats[0]
            pred = pred*stats[1] + stats[0]
            target_sum += target.sum()
            pred_sum += pred.sum()
            #print(loss)
        mape = np.abs(pred_sum-target_sum)/target_sum
        s = sess.run(model.summ.scalar_dev_performance, feed_dict={model.ph.loss_dev: np.mean(losses),
                                                                   model.ph.mape_dev: mape})
        sw.add_summary(s, c)
        target_sum = 0
        pred_sum = 0
        for batch, stats in batcher:
            c += 1
            loss, _, pred, train_summary = sess.run([model.losses.loss_mse,
                                               model.optimizers.op,
                                               model.core_model.output,
                                               model.summ.scalar_train_performance], batch)
            sw.add_summary(train_summary, c)
            target = batch[model.ph.target]*stats[1] + stats[0]
            pred = pred*stats[1] + stats[0]
            target_sum += target.sum()
            pred_sum += pred.sum()
        mape = np.abs(pred_sum-target_sum)/target_sum
        s = sess.run(model.summ.scalar_train_performance_manual, feed_dict={model.ph.mape_train: mape})

    # TODO: Implement TRAIN Mape in tensorboard
    # TODO: Implement TEST Mape in tensorboard
