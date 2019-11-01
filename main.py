import os
import numpy as np

from src.data_tools import (
    FactoryLoader,
    get_batches_generator,
    get_records_cube_from_df,
)

from src.constants import numeric_feats, categorical_feats
from src.general_utilities import get_custom_project_config, log_config
from src.common_paths import get_tensorboard_path, get_log_config_filepath
from tensorboardX import SummaryWriter
from src.architecture import Seq2Seq
import logging
import logging.config

logging.config.fileConfig(get_log_config_filepath(), disable_existing_loggers=False)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    config = get_custom_project_config()
    alias = config["alias"]
    random_seed = config["random_seed"]
    sample = config["sample"]
    cuda = config["cuda"]
    log_config(config)

    # Load data dependent on time
    logger.info("Generating time-dependent dataset...")
    df_master = FactoryLoader().load("master", sample=sample)
    logger.info(f"Time dataset generated successfully! Shape: {df_master.shape}")
    logger.info("Converting time-dependent dataset to data cube...")
    df_master = get_records_cube_from_df(df=df_master)
    cat_cardinalities_time = {
        col: len(np.unique(df_master[col]))
        for col in df_master.dtype.names
        if col in categorical_feats
    }
    logger.info(f"Data cube successfully generated! Shape: {df_master.shape}")

    # Load static data
    logger.info("Generating static dataset...")
    df_master_static = FactoryLoader().load("master_timeless", sample=sample)
    df_master_static = df_master_static.to_records()
    cat_cardinalities_timeless = {
        col: len(np.unique(df_master_static[col]))
        for col in df_master_static.dtype.names
        if col in categorical_feats
    }
    logger.info(f"Static data generated successfully! Shape: {df_master_static.shape}")

    # TODO: Check and delete item_nbr and store_number from time dataset

    # Feature groups definitions
    num_time_feats = np.intersect1d(numeric_feats, df_master.dtype.names)
    num_static_feats = np.intersect1d(numeric_feats, df_master_static.dtype.names)
    cat_time_feats = np.intersect1d(categorical_feats, df_master.dtype.names)
    cat_static_feats = np.intersect1d(categorical_feats, df_master_static.dtype.names)
    logging.info(f"Numeric time-dependent feats: {num_time_feats}")
    logging.info(f"Numeric static feats: {num_static_feats}")
    logging.info(f"Categorical time-dependent feats: {cat_time_feats}")
    logging.info(f"Categorical static feats: {cat_static_feats}")

    # Model delfinition
    logging.info("Building the architecture...")
    s2s = Seq2Seq(
        n_num_time_feats=len(num_time_feats),
        cardinalities_time=cat_cardinalities_time,
        cardinalities_static=cat_cardinalities_timeless,
        n_forecast_timesteps=7,
        lr=1e-4,
        cuda=cuda,
        name=alias,
    )
    logging.info("Architecture built successfully!")
    epoch, global_step, best_loss = s2s.load_checkpoint(best=False)

    # Define summary writer
    summaries_path = os.path.join(
        get_tensorboard_path(), alias + "_" + str(random_seed)
    )
    sw = SummaryWriter(log_dir=summaries_path)
    logging.info(f"Summary writer instantiated at {summaries_path}")

    logging.info(f"Starting the training loop!")
    for epoch in range(epoch, 1000):  #  Epochs loop
        logging.info(f"EPOCH: {epoch:06d} | Validation phase started...")
        is_best = False
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
            logger.debug(f"Dev batch {c} generated successfully. Calculating loss...")
            loss, forecast = s2s.loss(
                x_num_time=ntb,
                x_cat_time=ctb,
                x_cat_static=csb,
                cat_time_names=cat_time_feats,
                cat_static_names=cat_static_feats,
                target=target,
            )
            loss = loss.data.cpu().numpy()
            loss_dev += loss
            logger.debug(f"Dev batch loss = {loss}")
        sw.add_scalar("validation/epoch/loss", loss_dev / c, epoch)
        logging.info(f"EPOCH: {epoch:06d} | Validation finished. Loss = {loss_dev}")

        # ! Model serialization
        if loss_dev < best_loss:
            is_best = True
            best_loss = loss_dev
        s2s.save_checkpoint(
            epoch=epoch, best_loss=best_loss, is_best=is_best, global_step=global_step
        )

        # ! Training phase
        logging.info(f"EPOCH: {epoch:06d} | Training phase started...")
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
            logger.debug(f"Train batch {c} generated successfully. Running step...")
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
            loss = loss.data.cpu().numpy()
            loss_train += loss
            logger.debug(f"Train batch pre-step loss = {loss}")
        sw.add_scalar("train/epoch/loss", loss_train / c, epoch)
        logging.info(f"EPOCH: {epoch:06d} | Training finished. Loss = {loss_train}")
        logging.info(f"EPOCH: {epoch:06d} finished. Global steps: {global_step}")
