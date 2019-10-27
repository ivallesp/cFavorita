from src.data_tools import (
    FactoryLoader,
    get_batcher_generator,
    get_categorical_cardinalities,
    get_records_cube_from_df,
    shuffle_multiple,
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
import datetime

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

    print(datetime.datetime.now().isoformat(), "Shuffling...")
    df_master, df_master_static = shuffle_multiple(df_master, df_master_static)
    print(datetime.datetime.now().isoformat(), "Shuffle successful!")

    # Assure perfect alignment
    case_static = df_master_static[["store_nbr", "item_nbr"]]
    case_time = df_master[:, 0][["store_nbr", "item_nbr"]]
    assert (case_static == case_time).all()
    c = 0

    model = Seq2Seq(
        n_numerical_features=len(numeric_feats),
        categorical_cardinalities=cat_cardinalities,
        embedding_sizes=embedding_sizes,
        n_output_ts=30,
    )

    print("Model built successfully!")

    sess = start_tensorflow_session()
    sw = get_summary_writer(sess, get_tensorboard_path(), "CF", "V0")
    sess.run(tf.global_variables_initializer())

    train_end_idx = 1500
    ts_size = 1000
    pred_window_size = 30

    train_df = df[:, :train_end_idx]
    dev_df = df[:, (train_end_idx - ts_size) : (train_end_idx + pred_window_size)]

    non_zero_cases = train_df[:, :, 4].mean(axis=1) != 0
    train_df = train_df[non_zero_cases]
    dev_df = dev_df[non_zero_cases]
    df_static = df_static[non_zero_cases]

    norm_feats_idx = np.where(
        np.array(batch_time_normalizable_feats)[None]
        == np.array(colnames_time)[:, None]
    )[0]
    means = train_df[:, :, norm_feats_idx].mean(axis=1)
    stds = train_df[:, :, norm_feats_idx].std(axis=1)
    stds[stds == 0] = 1

    train_df[:, :, norm_feats_idx] = (
        train_df[:, :, norm_feats_idx] - means[:, None]
    ) / stds[:, None]
    dev_df[:, :, norm_feats_idx] = (
        dev_df[:, :, norm_feats_idx] - means[:, None]
    ) / stds[:, None]

    for epoch in range(10000):
        print("Epoch", epoch)
        batcher = get_batcher_generator(
            data_cube_time=train_df,
            data_cube_timeless=df_static,
            model=model,
            batch_size=128,
            colnames_time=colnames_time,
            colnames_timeless=colnames_timeless,
            history_window_size=ts_size,
            prediction_window_size=30,
            means=means,
            stds=stds,
        )
        batcher_test = get_batcher_generator(
            data_cube_time=dev_df,
            data_cube_timeless=df_static,
            model=model,
            batch_size=128,
            colnames_time=colnames_time,
            colnames_timeless=colnames_timeless,
            history_window_size=ts_size,
            prediction_window_size=30,
            means=means,
            stds=stds,
        )
        losses = []
        target_sum = 0
        pred_sum = 0
        for batch, stats in batcher_test:
            loss, pred = sess.run(
                [model.losses.loss_mse, model.core_model.output], batch
            )
            losses.append(loss)
            target = (
                batch[model.ph.target] * stats[1][:, 0][:, None, None]
                + stats[0][:, 0][:, None, None]
            )
            pred = pred * stats[1][:, 0][:, None, None] + stats[0][:, 0][:, None, None]
            target_sum += target.sum()
            pred_sum += pred.sum()
            # print(loss)
        mape = np.abs(pred_sum - target_sum) / target_sum
        s = sess.run(
            model.summ.scalar_dev_performance,
            feed_dict={model.ph.loss_dev: np.mean(losses), model.ph.mape_dev: mape},
        )
        sw.add_summary(s, c)
        target_sum = 0
        pred_sum = 0
        for batch, stats in batcher:
            c += 1
            loss, _, pred, train_summary = sess.run(
                [
                    model.losses.loss_mse,
                    model.optimizers.op,
                    model.core_model.output,
                    model.summ.scalar_train_performance,
                ],
                batch,
            )
            sw.add_summary(train_summary, c)
            target = (
                batch[model.ph.target] * stats[1][:, 0][:, None, None]
                + stats[0][:, 0][:, None, None]
            )
            pred = pred * stats[1][:, 0][:, None, None] + stats[0][:, 0][:, None, None]
            target_sum += target.sum()
            pred_sum += pred.sum()
        mape = np.abs(pred_sum - target_sum) / target_sum
        s = sess.run(
            model.summ.scalar_train_performance_manual,
            feed_dict={model.ph.mape_train: mape},
        )

    # TODO: Implement TRAIN Mape in tensorboard
    # TODO: Implement TEST Mape in tensorboard
