import logging

import numpy as np
from collections import OrderedDict
from src.architecture import Seq2Seq
from src.constants import categorical_feats, numeric_feats

logger = logging.getLogger(__name__)


def build_architecture(df_time, df_static, forecast_horizon, lr, cuda, alias):
    # Calculate cardinalities
    cat_cardinalities_static = OrderedDict(
        [
            (col, len(np.unique(df_static[col])))
            for col in df_static.dtype.names
            if col in categorical_feats
        ]
    )
    cat_cardinalities_time = OrderedDict(
        [
            (col, len(np.unique(df_time[col])))
            for col in df_time.dtype.names
            if col in categorical_feats
        ]
    )
    # Feature groups definitions
    num_time_feats = np.intersect1d(df_time.dtype.names, numeric_feats)
    num_static_feats = np.intersect1d(df_static.dtype.names, numeric_feats)
    cat_time_feats = np.array(list(cat_cardinalities_time.keys()))
    cat_static_feats = np.array(list(cat_cardinalities_static.keys()))
    logger.info(f"Numeric time-dependent feats: {num_time_feats}")
    logger.info(f"Numeric static feats: {num_static_feats}")
    logger.info(f"Categorical time-dependent feats: {cat_time_feats}")
    logger.info(f"Categorical static feats: {cat_static_feats}")

    # Model delfinition
    logger.info("Building the architecture...")
    s2s = Seq2Seq(
        n_num_time_feats=len(num_time_feats),
        cardinalities_time=cat_cardinalities_time,
        cardinalities_static=cat_cardinalities_static,
        n_forecast_timesteps=forecast_horizon,
        lr=lr,
        cuda=cuda,
        name=alias,
    )
    logger.info("Architecture built successfully!")
    return s2s


def run_validation_epoch(model, batcher):
    task = "validate"
    metrics = _run_epoch(model=model, batcher=batcher, task=task)
    return metrics


def run_training_epoch(model, batcher):
    task = "train"
    metrics = _run_epoch(model=model, batcher=batcher, task=task)
    return metrics


def _run_epoch(model, batcher, task="validate"):
    if task == "validate":
        f = model.loss
    elif task == "train":
        f = model.step
    else:
        ValueError(f"Task specified not defined: {task}")
    loss_dev = 0
    male_dev = 0
    total_abs_miss_dev = 0
    total_log_abs_miss_dev = 0
    total_target_dev = 0
    total_log_target_dev = 0
    for (c, (ntb, ctb, csb, target)) in enumerate(batcher):
        loss, forecast = f(
            x_num_time=ntb, x_cat_time=ctb, x_cat_static=csb, target=target
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
    metrics = {
        "loss": loss_dev / (c + 1),
        "male": male_dev / (c + 1),
        "log_mape": total_log_abs_miss_dev / total_log_target_dev,
        "mape": total_abs_miss_dev / total_target_dev,
    }
    return metrics
