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
        self.n_rec_units_encoder = 128
        self.n_rec_units_decoder = 128
        self.name = name
        self.cat_time_feats = np.array(list(cardinalities_time.keys()))
        self.cat_static_feats = np.array(list(cardinalities_static.keys()))
        self.encoder = Encoder(
            n_num_time_feats=n_num_time_feats,
            categorical_cardinalities=cardinalities_time,
            n_rec_units=self.n_rec_units_encoder,
            cuda=cuda,
        )
        self.decoder = Decoder(
            n_forecast_timesteps=n_forecast_timesteps,
            categorical_cardinalities=cardinalities_static,
            n_rec_units=self.n_rec_units_decoder,
            n_rec_units_encoder=self.n_rec_units_encoder,
            cuda=cuda,
        )
        self.initialize_weights()
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=lr)
        if cuda:
            self.cuda()

    def forward(self, x_num_time, x_cat_time, x_cat_static):
        output, thought = self.encoder.forward(
            x_num_time=x_num_time,
            x_cat_time=x_cat_time,
            cat_time_names=self.cat_time_feats,
        )
        output = self.decoder.forward(
            x_cat_static=x_cat_static,
            cat_static_names=self.cat_static_feats,
            outputs_encoder=output,
            state=thought,
        )
        return output

    def loss(self, x_num_time, x_cat_time, x_cat_static, target):
        y_hat = self.forward(
            x_num_time=x_num_time, x_cat_time=x_cat_time, x_cat_static=x_cat_static
        )
        loss = torch_rmse(target, y_hat)
        return loss, y_hat

    def step(self, x_num_time, x_cat_time, x_cat_static, target):
        loss, y_hat = self.loss(
            x_num_time=x_num_time,
            x_cat_time=x_cat_time,
            x_cat_static=x_cat_static,
            target=target,
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


class Encoder(nn.Module):
    def __init__(self, n_num_time_feats, categorical_cardinalities, n_rec_units, cuda):
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
        self.rnn_encoder = nn.LSTM(input_size=input_sz, hidden_size=n_rec_units)

    def forward(self, x_num_time, x_cat_time, cat_time_names):
        emb_feats = []
        for i, cat_feat_name in enumerate(cat_time_names):
            emb_feats += [self.embs[cat_feat_name](x_cat_time[:, :, i].long())]
            # TODO: make all the tensor long to enhance efficiency

        time_features = torch.cat([x_num_time] + emb_feats, -1).squeeze()
        output, state = self.rnn_encoder(time_features)
        return output, state


class Decoder(nn.Module):
    def __init__(
        self,
        n_forecast_timesteps,
        categorical_cardinalities,
        n_rec_units,
        n_rec_units_encoder,
        cuda,
    ):
        super().__init__()
        # Just add for consistency of structure with attention model
        del n_rec_units_encoder
        self.cuda_ = cuda
        self.n_forecast_timesteps = n_forecast_timesteps
        self.n_rec_units = n_rec_units
        self.rnn_decoder = nn.LSTM(input_size=1, hidden_size=self.n_rec_units)
        self.embs = {}

        for cat in categorical_cardinalities:
            self.embs[cat] = nn.Embedding(
                num_embeddings=categorical_cardinalities[cat],
                embedding_dim=embedding_sizes[cat],
                scale_grad_by_freq=False,
            )

        embs_sz = np.sum([embedding_sizes[c] for c in categorical_cardinalities.keys()])
        thought_sz = self.n_rec_units * 2
        context_thought_sz = thought_sz + embs_sz
        cd_h1 = nn.Linear(in_features=context_thought_sz, out_features=512)
        cd_h2 = nn.Linear(in_features=512, out_features=384)
        cd_h3 = nn.Linear(in_features=384, out_features=thought_sz)
        self.conditioning = nn.Sequential(
            cd_h1, nn.ReLU(True), cd_h2, nn.ReLU(True), cd_h3
        )
        td_h1 = nn.Linear(in_features=self.n_rec_units, out_features=128)
        td_h2 = nn.Linear(in_features=128, out_features=1)
        self.time_distributed = nn.Sequential(td_h1, nn.ReLU(True), td_h2)

    def forward(self, x_cat_static, cat_static_names, state, outputs_encoder):
        # Just add for consistency of structure with attention model
        del outputs_encoder
        batch_size = x_cat_static.shape[0]
        assert x_cat_static.shape[-1] == len(cat_static_names)

        emb_feats = []
        for i, cat_feat_name in enumerate(cat_static_names):
            emb_feats += [self.embs[cat_feat_name](x_cat_static[:, i].long())]
            # TODO: make all the tensor long to enhance efficiency

        thought = torch.cat(state, -1).squeeze()
        context_thought = torch.cat(emb_feats + [thought], -1).squeeze()
        context_thought = self.conditioning(context_thought)
        context_thought = (
            context_thought[:, : self.n_rec_units].unsqueeze(0).contiguous(),
            context_thought[:, self.n_rec_units :].unsqueeze(0).contiguous(),
        )

        input_decoder = torch.zeros(
            self.n_forecast_timesteps, context_thought[0].shape[1], 1
        )
        if self.cuda_:
            input_decoder = input_decoder.cuda()
        input_decoder[0, :, :] = 1  # GO!
        output, _ = self.rnn_decoder(input_decoder, context_thought)
        h = output.reshape(self.n_forecast_timesteps * batch_size, output.shape[-1])
        h = self.time_distributed(h)
        output = h.reshape(self.n_forecast_timesteps, batch_size, 1).squeeze()
        return output


class AttentionDecoder(nn.Module):
    def __init__(
        self,
        n_forecast_timesteps,
        categorical_cardinalities,
        n_rec_units,
        n_rec_units_encoder,
        cuda,
    ):
        super().__init__()
        self.cuda_ = cuda
        self.n_forecast_timesteps = n_forecast_timesteps
        self.n_rec_units = n_rec_units
        self.rnn_decoder = nn.LSTM(
            input_size=n_rec_units_encoder, hidden_size=self.n_rec_units
        )
        self.embs = {}

        for cat in categorical_cardinalities:
            self.embs[cat] = nn.Embedding(
                num_embeddings=categorical_cardinalities[cat],
                embedding_dim=embedding_sizes[cat],
                scale_grad_by_freq=False,
            )
            # Register the parameter for updating it (bc. not set as attribute directly)
            self.register_parameter("emb_mat_" + cat, self.embs[cat].weight)

        embs_sz = np.sum([embedding_sizes[c] for c in categorical_cardinalities.keys()])
        thought_sz = self.n_rec_units * 2
        context_thought_sz = thought_sz + embs_sz
        cd_h1 = nn.Linear(in_features=context_thought_sz, out_features=512)
        cd_h2 = nn.Linear(in_features=512, out_features=384)
        cd_h3 = nn.Linear(in_features=384, out_features=thought_sz)
        self.conditioning = nn.Sequential(
            cd_h1, nn.ReLU(True), cd_h2, nn.ReLU(True), cd_h3
        )

        n_feats = n_rec_units_encoder + self.n_rec_units * 2
        att_h1 = nn.Linear(in_features=n_feats, out_features=128,)
        att_h2 = nn.Linear(in_features=128, out_features=1)
        self.attention = nn.Sequential(att_h1, nn.ReLU(True), att_h2)

        td_h1 = nn.Linear(in_features=self.n_rec_units, out_features=128)
        td_h2 = nn.Linear(in_features=128, out_features=1)
        self.time_distributed = nn.Sequential(td_h1, nn.ReLU(True), td_h2)

    def forward(self, x_cat_static, cat_static_names, state, outputs_encoder):
        # Output_encoder shapes
        encoder_timesteps = outputs_encoder.shape[0]
        encoder_batch_size = outputs_encoder.shape[1]
        encoder_output_size = outputs_encoder.shape[2]

        batch_size = x_cat_static.shape[0]
        assert x_cat_static.shape[-1] == len(cat_static_names)

        emb_feats = []
        for i, cat_feat_name in enumerate(cat_static_names):
            emb_feats += [self.embs[cat_feat_name](x_cat_static[:, i].long())]
            # TODO: make all the tensor long to enhance efficiency

        thought = torch.cat(state, -1).squeeze()
        context_thought = torch.cat(emb_feats + [thought], -1).squeeze()
        context_thought = self.conditioning(context_thought)
        context_thought = (
            context_thought[:, : self.n_rec_units].unsqueeze(0).contiguous(),
            context_thought[:, self.n_rec_units :].unsqueeze(0).contiguous(),
        )

        outputs = []
        state = context_thought
        for i in range(self.n_forecast_timesteps):  # Forecast loop
            # Calculate Attention weights with state and outputs
            state_ = torch.cat([s[0] for s in state], 1).unsqueeze(0)  # Cat state (c,h)
            state_ = torch.cat(
                [state_] * encoder_timesteps, 0
            )  # Repeat state encoder timesteps
            att_input = torch.cat(
                [outputs_encoder, state_], -1
            )  # Cat state and encoder outputs
            # Join timesteps and batch_size dimensions
            n_feats = encoder_output_size + self.n_rec_units * 2
            att_input = att_input.reshape(-1, n_feats)
            # Calculate attention weights one by one
            w_attention = self.attention(att_input).squeeze()
            # Recover timesteps and batch_size dimensions
            w_attention = w_attention.reshape(encoder_timesteps, encoder_batch_size)
            # Apply the Softmax over the timesteps dimension
            w_attention = torch.nn.Softmax(dim=0)(w_attention)

            # Multiply the weights by the outputs_encoder
            w_attention = w_attention.unsqueeze(-1)  # Adapt the dims for recycling
            input_ = (outputs_encoder * w_attention).sum(0, keepdims=True)
            output, state = self.rnn_decoder(input_, state)
            outputs.append(output)
        output = torch.cat(outputs, 0)
        h = output.reshape(self.n_forecast_timesteps * batch_size, output.shape[-1])
        h = self.time_distributed(h)
        output = h.reshape(self.n_forecast_timesteps, batch_size, 1).squeeze()
        return output
