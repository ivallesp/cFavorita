import os
import numpy as np

from src.data_tools import (
    FactoryLoader,
    get_batches_generator,
    get_records_cube_from_df,
)

from src.constants import numeric_feats, categorical_feats
from src.general_utilities import get_custom_project_config
from src.common_paths import get_tensorboard_path
from tensorboardX import SummaryWriter
from src.architecture import Seq2Seq
import logging.config

logging.config.fileConfig(get_log_config_filepath(), disable_existing_loggers=False)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    config = get_custom_project_config()
    alias = config["alias"]
    random_seed = config["random_seed"]
    sample = config["sample"]
    cuda = config["cuda"]

    # Load data dependent on time
    df_master = FactoryLoader().load("master", sample=sample)
    df_master = get_records_cube_from_df(df=df_master)
    cat_cardinalities_time = {
        col: len(np.unique(df_master[col]))
        for col in df_master.dtype.names
        if col in categorical_feats
    }

    # Load static data
    df_master_static = FactoryLoader().load("master_timeless", sample=sample)
    df_master_static = df_master_static.to_records()
    cat_cardinalities_timeless = {
        col: len(np.unique(df_master_static[col]))
        for col in df_master_static.dtype.names
        if col in categorical_feats
    }

    # Feature groups definitions
    num_time_feats = np.intersect1d(numeric_feats, df_master.dtype.names)
    num_static_feats = np.intersect1d(numeric_feats, df_master_static.dtype.names)
    cat_time_feats = np.intersect1d(categorical_feats, df_master.dtype.names)
    cat_static_feats = np.intersect1d(categorical_feats, df_master_static.dtype.names)

    # Model delfinition
    s2s = Seq2Seq(
        n_num_time_feats=len(num_time_feats),
        cardinalities_time=cat_cardinalities_time,
        cardinalities_static=cat_cardinalities_timeless,
        n_forecast_timesteps=7,
        lr=1e-4,
        cuda=cuda,
    )

    # Define summary writer
    sw = SummaryWriter(
        log_dir=os.path.join(get_tensorboard_path(), alias + "_" + str(random_seed))
    )

    global_step = 0
    for epoch in range(1000):  #  Epochs loop
        # ! Validation phase
        batcher_dev = get_batches_generator(
            df_time=df_master,
            df_static=df_master_static,
            batch_size=128,
            forecast_horizon=7,
            shuffle=True,
            shuffle_present=False,
            cuda=cuda,
        )
        loss_dev = 0
        for (c, (ntb, ctb, csb, target)) in enumerate(batcher_dev):
            loss, forecast = s2s.loss(
                x_num_time=ntb,
                x_cat_time=ctb,
                x_cat_static=csb,
                cat_time_names=cat_time_feats,
                cat_static_names=cat_static_feats,
                target=target,
            )
            loss_dev += loss.data.cpu().numpy()
        sw.add_scalar("validation/epoch/loss", loss_dev / c, epoch)

        # ! Training phase
        batcher_train = get_batches_generator(
            df_time=df_master[:, :-7],
            df_static=df_master_static,
            batch_size=128,
            forecast_horizon=7,
            shuffle=True,
            shuffle_present=True,
            cuda=cuda,
        )
        loss_train = 0
        for (c, (ntb, ctb, csb, target)) in enumerate(batcher_train):
            loss, forecast = s2s.step(
                x_num_time=ntb,
                x_cat_time=ctb,
                x_cat_static=csb,
                cat_time_names=cat_time_feats,
                cat_static_names=cat_static_feats,
                target=target,
            )
            global_step += 1
            sw.add_scalar("train/in-batch/loss", loss, global_step)

            loss_train += loss.data.cpu().numpy()
        sw.add_scalar("train/epoch/loss", loss_train / c, epoch)
