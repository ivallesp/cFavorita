import logging
import logging.config
import os

import numpy as np
import wandb
from tensorboardX import SummaryWriter

from src.architecture import Seq2Seq
from src.common_paths import (
    get_log_config_filepath,
    get_model_path,
    get_tensorboard_path,
)
from src.constants import categorical_feats, numeric_feats
from src.data_tools import (
    FactoryLoader,
    get_batches_generator,
    get_records_cube_from_df,
)
from src.general_utilities import get_custom_project_config, log_config

logging.config.fileConfig(get_log_config_filepath(), disable_existing_loggers=False)
logger = logging.getLogger(__name__)

wandb.init("cFavorita")

if __name__ == "__main__":
    config = get_custom_project_config()
    alias = config["alias"]
    random_seed = config["random_seed"]
    sample = config["sample"]
    cuda = config["cuda"]
    batch_size = config["batch_size"]
    forecast_horizon = config["forecast_horizon"]
    learning_rate = config["learning_rate"]
    log_config(config)
    wandb.config.update(config)

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
    df_master_static = df_master_static.to_records(index=False)
    cat_cardinalities_timeless = {
        col: len(np.unique(df_master_static[col]))
        for col in df_master_static.dtype.names
        if col in categorical_feats
    }
    logger.info(f"Static data generated successfully! Shape: {df_master_static.shape}")

    # TODO: Check and delete item_nbr and store_number from time dataset
    # keys = ["date", "store_nbr", "item_nbr"]
    # df = df.drop(keys, axis=1)

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
        n_forecast_timesteps=forecast_horizon,
        lr=learning_rate,
        cuda=cuda,
        name=alias,
    )
    logging.info("Architecture built successfully!")
    epoch, global_step, best_loss = s2s.load_checkpoint(best=False)
    wandb.watch(s2s)

    # Define summary writer
    summaries_path = os.path.join(get_tensorboard_path(), alias)
    sw = SummaryWriter(log_dir=summaries_path)
    logging.info(f"Summary writer instantiated at {summaries_path}")

    logging.info(f"Starting the training loop!")
    for epoch in range(epoch, 10000):  # Epochs loop
        logging.info(f"EPOCH: {epoch:06d} | Validation phase started...")
        is_best = False
        # ! Validation phase
        batcher_dev = get_batches_generator(
            df_time=df_master,
            df_static=df_master_static,
            batch_size=batch_size,
            forecast_horizon=forecast_horizon,
            shuffle=True,
            shuffle_present=False,
            cuda=cuda,
        )
        loss_dev = 0
        male_dev = 0
        total_abs_miss_dev = 0
        total_log_abs_miss_dev = 0
        total_target_dev = 0
        total_log_target_dev = 0
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
            forecast = forecast.data.cpu().numpy()
            target = target.data.cpu().numpy()
            # Log metrics
            loss_dev += loss
            male_dev += np.mean(np.abs(forecast - target))
            log_abs_miss = np.abs(forecast - target)
            total_log_abs_miss_dev += log_abs_miss.sum()
            total_log_target_dev += target.sum()
            # Linear metrics
            forecast = np.expm1(forecast)
            target = np.expm1(target)
            abs_miss = np.abs(forecast - target)
            total_abs_miss_dev += abs_miss.sum()
            total_target_dev += target.sum()
        loss_dev /= c + 1
        male_dev /= c + 1
        log_mape_dev = total_log_abs_miss_dev / total_log_target_dev
        mape_dev = total_abs_miss_dev / total_target_dev
        sw.add_scalar("validation/epoch/loss", loss_dev, epoch)
        sw.add_scalar("validation/epoch/male", male_dev, epoch)
        sw.add_scalar("validation/epoch/mape", mape_dev, epoch)
        sw.add_scalar("validation/epoch/log_mape", log_mape_dev, epoch)
        logging.info(
            f"EPOCH: {epoch:06d} | Validation finished. Loss = {loss_dev} – "
            f"MALE = {male_dev} – MAPE = {mape_dev} - MAPLE = {log_mape_dev}"
        )

        # ! Model serialization
        if loss_dev < best_loss:
            is_best = True
        s2s.save_checkpoint(
            epoch=epoch, best_loss=best_loss, is_best=is_best, global_step=global_step
        )

        # ! Training phase
        logging.info(f"EPOCH: {epoch:06d} | Training phase started...")
        batcher_train = get_batches_generator(
            df_time=df_master[:, :-forecast_horizon],
            df_static=df_master_static,
            batch_size=batch_size,
            forecast_horizon=forecast_horizon,
            shuffle=True,
            shuffle_present=True,
            cuda=cuda,
        )
        loss_train = 0
        male_train = 0
        total_abs_miss_train = 0
        total_log_abs_miss_train = 0
        total_target_train = 0
        total_log_target_train = 0
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
            loss = loss.data.cpu().numpy()
            forecast = forecast.data.cpu().numpy()
            target = target.data.cpu().numpy()
            # Log metrics
            sw.add_scalar("train/in-batch/loss", loss, global_step)
            loss_train += loss
            male_train += np.mean(np.abs(forecast - target))
            log_abs_miss = np.abs(forecast - target)
            total_log_abs_miss_train += log_abs_miss.sum()
            total_log_target_train += target.sum()
            # Linear metrics
            forecast = np.expm1(forecast)
            target = np.expm1(target)
            abs_miss = np.abs(forecast - target)
            total_abs_miss_train += abs_miss.sum()
            total_target_train += target.sum()
            logger.debug(f"Train batch pre-step loss = {loss}")
        loss_train /= c + 1
        male_train /= c + 1
        mape_train = total_abs_miss_train / total_target_train
        log_mape_train = total_log_abs_miss_train / total_log_target_train
        sw.add_scalar("train/epoch/loss", loss_train, epoch)
        sw.add_scalar("train/epoch/male", male_train, epoch)
        sw.add_scalar("train/epoch/mape", mape_train, epoch)
        sw.add_scalar("train/epoch/log_mape", log_mape_train, epoch)

        logging.info(
            f"EPOCH: {epoch:06d} | Epoch finished. Train Loss = {loss_train} – "
            f"MALE = {male_train} – MAPE = {mape_train}  – MAPLE = {log_mape_train} – "
            f"Global steps: {global_step}"
        )
        wandb.log(
            {
                "loss_dev": loss_dev,
                "male_dev": male_dev,
                "mape_dev": mape_dev,
                "log_mape_dev": log_mape_dev,
                "loss_train": loss_train,
                "male_train": male_train,
                "mape_train": mape_train,
                "log_mape_train": log_mape_train,
                "epoch": epoch,
            }
        )

        if epoch % 10 == 0:  # Save to wandb
            path = get_model_path(alias=alias)
            wandb.save(os.path.join(path, "*"))
            wandb.save(os.path.join(get_log_config_filepath(), "*"))
            wandb.save(os.path.join(get_tensorboard_path(), "*"))
