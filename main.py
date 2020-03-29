import logging.config
import os
import random
import torch

import numpy as np
import wandb
from tensorboardX import SummaryWriter

from src.common_paths import (
    get_log_config_filepath,
    get_model_path,
    get_tensorboard_path,
)

from src.data_tools import (
    shuffle_multiple,
    get_train_data_loader,
    get_dev_data_loader,
    get_data_cubes,
)
from src.general_utilities import get_custom_project_config, log_config
from src.model import build_architecture, run_validation_epoch, run_training_epoch

logging.config.fileConfig(get_log_config_filepath(), disable_existing_loggers=False)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    config = get_custom_project_config()
    alias = config["alias"]
    random_seed = config["random_seed"]
    sample = config["sample"]
    cuda = config["cuda"]
    batch_size = config["batch_size"]
    forecast_horizon = config["forecast_horizon"]
    learning_rate = config["learning_rate"]
    n_threads = config["n_threads"]
    log_config(config)
    wandb.init(project="cFavorita", config=config, id=alias, resume=alias)
    wandb.config.update(config)

    # Set random seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    # Load data
    df_master, df_master_static, cat_dict, cat_dict_static = get_data_cubes(sample)

    # Define batchers
    batcher_train = get_train_data_loader(
        df_time=df_master,
        df_static=df_master_static,
        batch_size=batch_size,
        forecast_horizon=forecast_horizon,
        n_jobs=n_threads,
    )
    batcher_dev = get_dev_data_loader(
        df_time=df_master,
        df_static=df_master_static,
        batch_size=batch_size,
        forecast_horizon=forecast_horizon,
        n_jobs=n_threads,
    )

    # Build model
    s2s = build_architecture(
        df_time=df_master,
        df_static=df_master_static,
        forecast_horizon=forecast_horizon,
        lr=learning_rate,
        cuda=cuda,
        alias=alias,
    )
    epoch, global_step, best_loss = s2s.load_checkpoint(best=False)

    wandb.watch(s2s)

    # Define summary writer
    summaries_path = os.path.join(get_tensorboard_path(), alias)
    sw = SummaryWriter(log_dir=summaries_path)
    logger.info(f"Summary writer instantiated at {summaries_path}")

    logger.info(f"Starting the training loop!")
    for epoch in range(epoch, 10000):  # Epochs loop
        # ! Validation phase
        logger.info(f"EPOCH: {epoch:06d} | Validation phase started...")
        is_best = False
        metrics_dev = run_validation_epoch(model=s2s, batcher=batcher_dev, cuda=cuda)

        # ! Model serialization
        if metrics_dev["loss"] < best_loss:
            is_best = True
            best_loss = metrics_dev["loss"]
        s2s.save_checkpoint(
            epoch=epoch, best_loss=best_loss, is_best=is_best, global_step=global_step
        )

        # ! Training phase
        logger.info(f"EPOCH: {epoch:06d} | Training phase started...")
        metrics_train = run_training_epoch(model=s2s, batcher=batcher_train, cuda=cuda)

        # ! Report
        for m in metrics_dev:
            sw.add_scalar(f"validation/epoch/{m}", metrics_dev[m], epoch)

        for m in metrics_train:
            sw.add_scalar(f"train/epoch/{m}", metrics_train[m], epoch)

        metrics_dev = {k + "_dev": v for k, v in metrics_dev.items()}
        metrics_train = {k + "_train": v for k, v in metrics_train.items()}
        metrics = {**metrics_dev, **metrics_train}

        logger.info(
            f"EPOCH: {epoch:06d} finished!"
            f"\n\tTraining | Loss = {metrics['loss_train']} | "
            f"\n\tValidation | Loss = {metrics['loss_dev']}"
        )

        wandb.log({**metrics, **{"epoch": epoch}})

        if epoch % 200 == 0:  # Save to wandb
            logger.info("Uploading state...")
            path = get_model_path(alias=alias)
            wandb.save(os.path.join(path, "*"))
            wandb.save(os.path.join(get_log_config_filepath(), "*"))
            wandb.save(os.path.join(get_tensorboard_path(), "*"))
            logger.info("Weight uploaded successfully!")
