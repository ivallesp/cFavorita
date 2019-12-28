import logging

import numpy as np

from src.architecture import Seq2Seq
from src.constants import categorical_feats, numeric_feats

logger = logging.getLogger(__name__)


def build_architecture(df_time, df_static, forecast_horizon, lr, cuda, alias):
    # Calculate cardinalities
    cat_cardinalities_static = {
        col: len(np.unique(df_static[col]))
        for col in df_static.dtype.names
        if col in categorical_feats
    }
    cat_cardinalities_time = {
        col: len(np.unique(df_time[col]))
        for col in df_time.dtype.names
        if col in categorical_feats
    }
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
