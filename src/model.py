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


def run_validation_epoch(model, batcher, cuda):
    task = "validate"
    metrics = _run_epoch(model=model, batcher=batcher, task=task, cuda=cuda)
    return metrics


def run_training_epoch(model, batcher, cuda):
    task = "train"
    metrics = _run_epoch(model=model, batcher=batcher, task=task, cuda=cuda)
    return metrics


def _run_epoch(model, batcher, task="validate", cuda=False):
    if task == "validate":
        f = model.loss
    elif task == "train":
        f = model.step
    else:
        ValueError(f"Task specified not defined: {task}")
    epoch_loss = 0
    epoch_male = 0
    total_weight = 0
    for (c, (ntb, ctb, csb, fwb, target, weight)) in enumerate(batcher):
        if cuda:
            ntb = ntb.cuda()
            ctb = ctb.cuda()
            csb = csb.cuda()
            # fwb = fwb.cuda()
            target = target.cuda()
            weight = weight.cuda()
        loss, forecast = f(
            x_num_time=ntb,
            x_cat_time=ctb,
            x_cat_static=csb,
            # x_fwd=fwb,
            target=target,
            weight=weight,
        )

        loss = loss.data.cpu().numpy()
        forecast = forecast.data.cpu().numpy()
        target = target.data.cpu().numpy()
        weight = weight.data.cpu().numpy()
        total_weight += weight.sum()

        assert forecast.shape == target.shape
        assert target.shape == weight.shape

        # Log metrics
        # Loss calculation (WRMSLE)
        sqrt_miss = loss * np.sqrt(weight.sum())  # Remove the denominator
        miss = sqrt_miss ** 2  # Remove the sqrt from the numerator
        epoch_loss += miss  # Accumulate weighted squared differences

        # WMALE calculation
        abs_miss = np.abs(forecast - target)  # Calculate abs differences
        epoch_male += np.sum(abs_miss * weight)  # Accumulate weighted absmiss

        # Linear metrics
        # forecast = np.expm1(forecast)
        # target = np.expm1(target)
        # abs_miss = np.abs(forecast - target)

    metrics = {
        "loss": np.sqrt(epoch_loss / total_weight),  # Divide by the denominator
        "male": epoch_male / total_weight,
    }
    return metrics
