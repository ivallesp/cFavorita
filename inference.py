import logging.config
import os
import random
import torch
import argparse
import pandas as pd
#import pytorch_warmup as warmup
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
    get_live_data_loader,
    get_data_cubes,
    get_custom_data_loader
)
from src.general_utilities import get_custom_project_config, log_config
from src.model import build_architecture, run_validation_epoch, run_training_epoch

aliases = ['s2s_trim_655321_shorter0', 's2s_trim_655322_shorter0', 
           's2s_trim_655323_shorter0', 's2s_trim_655324_shorter0', 
           's2s_trim_655325_shorter0']


# Try loading aliases config
for alias in aliases:
    get_custom_project_config(alias)
    model_path=os.path.join("models", alias)
    assert os.path.exists(model_path), model_path

LAG = 3

print("Loading dataset")
df_master, df_master_static, cat_dict, cat_dict_static = get_data_cubes(False)

# Build the dataset
forecast_horizon = 16

for lag in range(1, LAG+1):
    master_f = df_master[:,:-(lag*forecast_horizon)][:,-forecast_horizon:]

    month = (((master_f['month']/2+.5)*11)+1).round().astype(int)
    day = (((master_f['day']/2+.5)*30)+1).round().astype(int)
    year = (master_f['year']*2+2015).round().astype(int)

    map_str = {v:k for k, v in cat_dict_static['store_nbr'].items()}
    store_nbr = np.array([map_str[s] for s in df_master_static['store_nbr']])
    store_nbr = np.tile(store_nbr, (forecast_horizon, 1)).transpose()

    map_itm = {v:k for k, v in cat_dict_static['item_nbr'].items()}
    item_nbr = np.array([map_itm[s] for s in df_master_static['item_nbr']])
    item_nbr = np.tile(item_nbr, (forecast_horizon, 1)).transpose()

    df_results = pd.DataFrame({'item_nbr': item_nbr.reshape(-1),
                               'store_nbr': store_nbr.reshape(-1),
                               'year': year.reshape(-1),
                               'month': month.reshape(-1),
                               'day': day.reshape(-1)})


    for alias in aliases:
        print(f'Computing alias {alias} - lag {lag}')
        config = get_custom_project_config(alias)
        random_seed = config["random_seed"]
        sample = config["sample"]
        cuda = config["cuda"]
        batch_size = config["batch_size"]
        forecast_horizon = config["forecast_horizon"]
        learning_rate = config["learning_rate"]
        n_threads = config["n_threads"]
        n_epochs = config["n_epochs"]
        n_history_ts = config["n_history_ts"]


        model = build_architecture(
            df_time=df_master,
            df_static=df_master_static,
            forecast_horizon=forecast_horizon,
            lr=learning_rate,
            cuda=cuda,
            alias=alias,
        )
        epoch, global_step, best_loss = model.load_checkpoint(best=True)

        _=model.eval()

        batcher_test = get_custom_data_loader(
            df_time=df_master,
            df_static=df_master_static,
            batch_size=batch_size*10,
            forecast_horizon=forecast_horizon,
            n_jobs=n_threads,
            n_history_ts=n_history_ts,
            lag = forecast_horizon*lag
        )

        # Predict
        target_mat = []
        forecast_mat = []
        weight_mat = []

        for (c, (ntb, ctb, csb, fwb, target, weight)) in enumerate(batcher_test):
            ntb = ntb.cuda()
            ctb = ctb.cuda()
            csb = csb.cuda()
            # fwb = fwb.cuda()
            target = target.cuda()
            weight = weight.cuda()

            loss, forecast = model.loss(
                x_num_time=ntb,
                x_cat_time=ctb,
                x_cat_static=csb,
                # x_fwd=fwb,
                target=target,
                weight=weight,
            )
            forecast_mat.append(forecast.cpu().data.numpy())
            target_mat.append(target.cpu().data.numpy())
            weight_mat.append(weight.cpu().data.numpy())

        # Append matrices
        target_mat = np.concatenate(target_mat, axis=1)
        forecast_mat = np.concatenate(forecast_mat, axis=1)
        weight_mat = np.concatenate(weight_mat, axis=1)

        target_mat = target_mat.transpose()
        forecast_mat = forecast_mat.transpose()
        weight_mat = weight_mat.transpose()

        # JUST A CHECK!
        r=master_f['unit_sales'].transpose()
        assert (r.transpose()-target_mat<1e-6).all() # Assert same order!

        assert target_mat.shape == forecast_mat.shape
        assert target_mat.shape == month.shape
        assert target_mat.shape == day.shape
        assert target_mat.shape == year.shape
        assert target_mat.shape == store_nbr.shape
        assert target_mat.shape == item_nbr.shape

        if 'target' in df_results.columns:
            assert (df_results[f'target'].values == target_mat.reshape(-1)).all()
        df_results[f'target'] = target_mat.reshape(-1)
        df_results[f'forecast_{alias}'] = forecast_mat.reshape(-1)
    df_results.to_csv(f'models/results_{alias}_lag_{lag}.csv', index=False)
