import logging
import os
import shutil

import numpy as np
import torch
from torch import nn

from src.common_paths import get_model_path
from src.constants import embedding_sizes

logger = logging.getLogger(__name__)


class Seq2Seq(nn.Module):
    def __init__(
        self,
        n_num_time_feats,
        cardinalities_time,
        cardinalities_static,
        n_forecast_timesteps,
        lr,
        cuda,
        name,
    ):
        super().__init__()
        self.name = name
        self.cat_time_feats = np.array(list(cardinalities_time.keys()))
        self.cat_static_feats = np.array(list(cardinalities_static.keys()))
        self.encoder = Encoder(
            n_num_time_feats=n_num_time_feats,
            categorical_cardinalities=cardinalities_time,
            cuda=cuda,
        )
        self.decoder = Decoder(
            n_forecast_timesteps=n_forecast_timesteps,
            categorical_cardinalities=cardinalities_static,
            cuda=cuda,
        )
        self.initialize_weights()
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=lr)
        if cuda:
            self.cuda()

    def forward(self, x_num_time, x_cat_time, x_cat_static):
        contextual_thought = self.encoder.forward(
            x_num_time=x_num_time,
            x_cat_time=x_cat_time,
            cat_time_names=self.cat_time_feats,
        )
        output = self.decoder.forward(
            x_cat_static=x_cat_static,
            # x_fwd=x_fwd,
            cat_static_names=self.cat_static_feats,
            state=contextual_thought,
        )
        return output

    def loss(self, x_num_time, x_cat_time, x_cat_static, target, weight):
        y_hat = self.forward(
            x_num_time=x_num_time,
            x_cat_time=x_cat_time,
            x_cat_static=x_cat_static,
            # x_fwd=x_fwd,
        )
        loss = torch_wrmse(target, y_hat, weight)
        return loss, y_hat

    def step(self, x_num_time, x_cat_time, x_cat_static, target, weight):
        loss, y_hat = self.loss(
            x_num_time=x_num_time,
            x_cat_time=x_cat_time,
            x_cat_static=x_cat_static,
            target=target,
            weight=weight,
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss, y_hat

    def save_checkpoint(self, epoch, global_step, best_loss, is_best=False):
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html
        state = {
            "model_params": self.state_dict(),
            "opt_params": self.optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_loss": best_loss,
        }
        path = get_model_path(alias=self.name)
        filepath = os.path.join(path, "last.dat")
        torch.save(state, filepath)
        if is_best:
            filepath_best = os.path.join(path, "best.dat")
            shutil.copyfile(filepath, filepath_best)

    def load_checkpoint(self, best=False):
        path = get_model_path(alias=self.name)
        if best:
            filepath = os.path.join(path, "best.dat")
        else:
            filepath = os.path.join(path, "last.dat")
        if os.path.exists(filepath):
            logger.info(f"Checkpoint found! Loading {filepath}")
            state = torch.load(filepath)
            self.load_state_dict(state["model_params"])
            self.optimizer.load_state_dict(state["opt_params"])
            epoch = state["epoch"]
            global_step = state["global_step"]
            best_loss = state["best_loss"]
            logger.info(f"Checkpoint loaded successfully.")
        else:
            logger.warn(f"Checkpoint not found at {filepath}. Training a new model...")
            epoch = 0
            global_step = 0
            best_loss = np.Inf
        logger.info(f"Model at ep={epoch}, g_step={global_step}, best_loss={best_loss}")
        return epoch, global_step, best_loss

    def initialize_weights(self):
        # https://discuss.pytorch.org/t/initializing-parameters-of-a-multi-layer-lstm/5791
        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight" in name:
                nn.init.kaiming_normal_(param)
            elif "emb" in name:
                nn.init.kaiming_normal_(param)
            else:
                logger.warn(f"Parameter {name} not initialized!")


def torch_rmse(actual, forecast):
    residuals = actual - forecast
    rmse = torch.sqrt(torch.mean((residuals) ** 2))
    return rmse


def torch_wrmse(actual, forecast, weight):
    # Assure no shapes mess-up
    assert weight.shape == forecast.shape
    assert actual.shape == forecast.shape
    residuals = actual - forecast
    sq_residuals = residuals ** 2
    w_sq_residuals = weight * sq_residuals
    wrmse = torch.sqrt(torch.sum(w_sq_residuals) / torch.sum(weight))
    return wrmse


class Encoder(nn.Module):
    def __init__(self, n_num_time_feats, categorical_cardinalities, cuda):
        super().__init__()
        self.cuda_ = cuda
        self.embs = nn.ModuleDict()
        for cat in categorical_cardinalities:
            self.embs[cat] = nn.Embedding(
                num_embeddings=categorical_cardinalities[cat],
                embedding_dim=embedding_sizes[cat],
                scale_grad_by_freq=False,
            )

        embs_sz = np.sum([embedding_sizes[c] for c in categorical_cardinalities.keys()])
        input_sz = int(embs_sz + n_num_time_feats)
        self.rnn_encoder = nn.LSTM(input_size=input_sz, hidden_size=128)

    def forward(self, x_num_time, x_cat_time, cat_time_names):
        emb_feats = []
        for i, cat_feat_name in enumerate(cat_time_names):
            emb_feats += [self.embs[cat_feat_name](x_cat_time[:, :, i].long())]
        time_features = torch.cat([x_num_time] + emb_feats, -1).squeeze()
        _, state = self.rnn_encoder(time_features)
        return state


class Decoder(nn.Module):
    def __init__(self, n_forecast_timesteps, categorical_cardinalities, cuda):
        super().__init__()
        self.cuda_ = cuda
        self.n_forecast_timesteps = n_forecast_timesteps
        self.n_recurrent_cells = 128
        self.rnn_decoder = nn.LSTM(input_size=1, hidden_size=self.n_recurrent_cells)
        self.embs = nn.ModuleDict()

        for cat in categorical_cardinalities:
            self.embs[cat] = nn.Embedding(
                num_embeddings=categorical_cardinalities[cat],
                embedding_dim=embedding_sizes[cat],
                scale_grad_by_freq=False,
            )

        embs_sz = np.sum([embedding_sizes[c] for c in categorical_cardinalities.keys()])
        thought_sz = self.n_recurrent_cells * 2
        context_thought_sz = thought_sz + embs_sz
        cd_h1 = nn.Linear(in_features=context_thought_sz, out_features=512)
        cd_h2 = nn.Linear(in_features=512, out_features=384)
        cd_h3 = nn.Linear(in_features=384, out_features=thought_sz)
        self.conditioning = nn.Sequential(
            cd_h1, nn.ReLU(True), cd_h2, nn.ReLU(True), cd_h3
        )
        td_h1 = nn.Linear(in_features=self.n_recurrent_cells, out_features=128)
        td_h2 = nn.Linear(in_features=128, out_features=1)
        self.time_distributed = nn.Sequential(td_h1, nn.ReLU(True), td_h2)

    def forward(self, x_cat_static, cat_static_names, state):
        # Mock the input of the decoder
        batch_size = x_cat_static.shape[0]
        assert x_cat_static.shape[-1] == len(cat_static_names)

        emb_feats = []
        for i, cat_feat_name in enumerate(cat_static_names):
            emb_feats += [self.embs[cat_feat_name](x_cat_static[:, i].long())]

        thought = torch.cat(state, -1).squeeze()
        context_thought = torch.cat(emb_feats + [thought], -1).squeeze()
        context_thought = self.conditioning(context_thought)
        context_thought = (
            context_thought[:, : self.n_recurrent_cells].unsqueeze(0).contiguous(),
            context_thought[:, self.n_recurrent_cells :].unsqueeze(0).contiguous(),
        )

        input_decoder = torch.zeros(
            self.n_forecast_timesteps, context_thought[0].shape[1], 1
        )
        if self.cuda_:
            input_decoder = input_decoder.cuda()
        input_decoder[0, :, :] = 1  # GO!
        # Concatenate the fwd looking features
        # input_decoder = torch.cat([input_decoder, x_fwd], dim=2)
        output, _ = self.rnn_decoder(input_decoder, context_thought)
        h = output.reshape(self.n_forecast_timesteps * batch_size, output.shape[-1])
        h = self.time_distributed(h)
        output = h.reshape(self.n_forecast_timesteps, batch_size, 1).squeeze()
        return output
