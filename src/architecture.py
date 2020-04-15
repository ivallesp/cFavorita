import logging
import os
import shutil
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from src.common_paths import get_model_path
from src.constants import embedding_sizes

logger = logging.getLogger(__name__)


class Transformer(nn.Module):
    def __init__(
        self,
        n_num_time_feats,
        cardinalities_time,
        cardinalities_static,
        n_forecast_timesteps,
        lr,
        cuda,
        name,
        dropout,
    ):
        super().__init__()
        self.cuda_ = cuda
        self.name = name
        self.cat_time_feats = np.array(list(cardinalities_time.keys()))
        self.cat_static_feats = np.array(list(cardinalities_static.keys()))

        ######
        self.embs_time = nn.ModuleDict()
        for cat in cardinalities_time:
            self.embs_time[cat] = nn.Embedding(
                num_embeddings=cardinalities_time[cat],
                embedding_dim=embedding_sizes[cat],
                scale_grad_by_freq=False,
            )

        embs_sz = np.sum([embedding_sizes[c] for c in cardinalities_time.keys()])
        input_sz = int(embs_sz + n_num_time_feats)
        pos_emb_size = 20
        d_model = input_sz + pos_emb_size
        self.encoder = EncoderTransformer(
            d_model=d_model,
            pos_emb_size=pos_emb_size,
            n_input_feats=input_sz,
            dropout=dropout,
            N=6,
            cuda=cuda,
        )
        ######

        ######
        self.embs_cat = nn.ModuleDict()

        for cat in cardinalities_static:
            self.embs_cat[cat] = nn.Embedding(
                num_embeddings=cardinalities_static[cat],
                embedding_dim=embedding_sizes[cat],
                scale_grad_by_freq=False,
            )

        embs_sz = np.sum([embedding_sizes[c] for c in cardinalities_static.keys()])
        input_sz = int(embs_sz + 1)
        self.decoder = DecoderTransformer(
            d_model=d_model,
            pos_emb_size=pos_emb_size,
            n_input_feats=input_sz,
            n_output_feats=1,
            dropout=dropout,
            N=6,
            cuda=cuda,
        )
        ######

        self.initialize_weights()
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=lr)
        if cuda:
            self.cuda()

    def forward(self, x_num_time, x_cat_time, x_cat_static, y, output_encoder=None):
        # Encoder
        if output_encoder is None:
            emb_feats = []
            for i, cat_feat_name in enumerate(self.cat_time_feats):
                emb_feats += [self.embs_time[cat_feat_name](x_cat_time[:, :, i].long())]
            time_features = torch.cat([x_num_time] + emb_feats, -1)
            output_encoder = self.encoder(time_features)

        # Decoder
        assert x_cat_static.shape[-1] == len(self.cat_static_feats)

        emb_feats = []
        for i, cat_feat_name in enumerate(self.cat_static_feats):
            emb_feats += [self.embs_cat[cat_feat_name](x_cat_static[:, i].long())]
        emb_feats = torch.cat(emb_feats, -1).squeeze()

        # Right shift y to use always info from the prev. time step
        y = torch.cat([torch.zeros_like(y[[0]]), y[:-1]], 0)
        y = y[:, :, None]

        # Concatenate the features of the embedding and the output
        y = torch.cat([y, emb_feats[None, :].repeat(y.shape[0], 1, 1)], -1)
        output_decoder = self.decoder(y=y, h=output_encoder)
        assert output_decoder.shape[-1] == 1
        return output_encoder, output_decoder[:, :, 0]

    def loss(
        self,
        x_num_time,
        x_cat_time,
        x_cat_static,
        target,
        weight,
        y,
        autoregressive=False,
    ):
        if autoregressive:

            y_hat = torch.zeros_like(y[[0]])
            forecasting_horizon = y.shape[0]
            output_encoder = None  # Cache encoder output for the autoregressive loop

            for i in range(forecasting_horizon):
                with torch.no_grad():
                    output_encoder, y_hat_last = self.forward(
                        x_num_time=x_num_time,
                        x_cat_time=x_cat_time,
                        x_cat_static=x_cat_static,
                        y=y_hat,
                        output_encoder=output_encoder,
                    ).detach()[-1:]
                    y_hat = torch.cat([y_hat, y_hat_last], axis=0)
            y_hat = y_hat[1:]

        else:
            _, y_hat = self.forward(
                x_num_time=x_num_time,
                x_cat_time=x_cat_time,
                x_cat_static=x_cat_static,
                y=y
                # x_fwd=x_fwd,
            )
        loss = torch_wrmse(target, y_hat, weight)
        return loss, y_hat

    def step(self, x_num_time, x_cat_time, x_cat_static, target, weight, y):
        loss, y_hat = self.loss(
            x_num_time=x_num_time,
            x_cat_time=x_cat_time,
            x_cat_static=x_cat_static,
            target=target,
            weight=weight,
            y=y,
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
            if "layer_norm" in name:
                logger.warn(f"Parameter {name} initialization skipped!")
                continue
            elif "bias" in name:
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


class EncoderTransformer(nn.Module):
    def __init__(self, d_model, pos_emb_size, n_input_feats, dropout, N=6, cuda=False):

        super().__init__()
        self.cuda_ = cuda
        self.pos_emb_size = pos_emb_size
        # Make sure d_model >> pos_emb_size
        self.input_ff = nn.Linear(n_input_feats, d_model - pos_emb_size)
        layers = []
        for _ in range(N):
            layer = EncoderBlock(
                n_heads_attention=8,
                d_in_attention=d_model,
                d_out_attention=d_model,
                dropout=dropout,
            )
            layers.append(layer)
        layers = nn.ModuleList(layers)
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # Generate positional embedding
        pos_emb = generate_positional_embedding(
            length=x.shape[0],  # Number of time steps
            channels=self.pos_emb_size,  # Number of features
        )
        if self.cuda_:
            pos_emb = pos_emb.cuda()
        pos_emb = pos_emb.repeat(1, x.shape[1], 1)  # Broadcast in batch dimension
        # Project input space to adjust to d_model size
        x = self.input_ff(x)
        # Concat pos embedding to input (in the original paper, this is a sum)
        x = torch.cat([x, pos_emb], axis=-1)

        return self.layers(x)


class DecoderTransformer(nn.Module):
    def __init__(
        self,
        d_model,
        pos_emb_size,
        n_input_feats,
        n_output_feats,
        dropout,
        N=6,
        cuda=False,
    ):

        super().__init__()
        self.cuda_ = cuda
        self.pos_emb_size = pos_emb_size
        # Make sure d_model >> pos_emb_size
        self.input_ff = nn.Linear(n_input_feats, d_model - pos_emb_size)
        layers = []
        for _ in range(N):
            layer = DecoderBlock(
                n_heads_attention=8,
                d_in_attention=d_model,
                d_out_attention=d_model,
                dropout=dropout,
            )
            layers.append(layer)
        self.output_ff = nn.Linear(d_model, n_output_feats)

        self.layers = nn.ModuleList(layers)

    def forward(self, y, h):
        # Generate positional embedding
        pos_emb = generate_positional_embedding(
            length=y.shape[0],  # Number of time steps
            channels=self.pos_emb_size,  # Number of features
        )
        if self.cuda_:
            pos_emb = pos_emb.cuda()
        pos_emb = pos_emb.repeat(1, h.shape[1], 1)  # Broadcast in batch dimension
        # Project input space to adjust to d_model size
        y = self.input_ff(y)
        # Concat pos embedding to input (in the original paper, this is a sum)
        y = torch.cat([y, pos_emb], axis=-1)
        for layer in self.layers:
            y = layer(y, h)
        y = self.output_ff(y)
        return y


class EncoderBlock(nn.Module):
    def __init__(self, n_heads_attention, d_in_attention, d_out_attention, dropout):
        super().__init__()
        self.attention = MultiHeadAttention(
            n_heads=n_heads_attention,
            d_in=d_in_attention,
            d_out=d_out_attention,
            mask_out_future=False,
            dropout=dropout,
        )
        self.ff = FeedForwardTransformerLayer(
            expansion_factor=4, input_size=d_in_attention, dropout=dropout
        )

    def forward(self, x):
        # Multi-Head Attention layer
        h = self.attention(query=x, key=x, value=x)  # self-attention
        # FF layer
        h = self.ff(h)
        return h


class DecoderBlock(nn.Module):
    def __init__(self, n_heads_attention, d_in_attention, d_out_attention, dropout):
        super().__init__()
        self.attention_dec = MultiHeadAttention(
            n_heads=n_heads_attention,
            d_in=d_in_attention,
            d_out=d_out_attention,
            mask_out_future=True,  # Make attention causal
            # TODO: Try if it improves by only masking the first layer, and unmasking the subsequent ones
            dropout=dropout,
        )
        self.attention_enc_dec = MultiHeadAttention(
            n_heads=n_heads_attention,
            d_in=d_in_attention,
            d_out=d_out_attention,
            mask_out_future=False,
            dropout=dropout,
        )
        self.ff = FeedForwardTransformerLayer(
            expansion_factor=4, input_size=d_in_attention, dropout=dropout
        )

    def forward(self, y, h):
        """[summary]

        Args:
            y ([type]): decoder previous outputs
            h ([type]): encoder hidden state

        Returns:
            [type]: [description]
        """
        # Decoder multi-head Attention layer
        h = self.attention_dec(query=y, key=y, value=y)  # self-attention.
        # Encoder-decoder multi-head Attention layer
        h = self.attention_enc_dec(query=h, key=h, value=h)  # self-attention.
        # FF layer
        h = self.ff(h)
        return h


class FeedForwardTransformerLayer(nn.Module):
    def __init__(self, expansion_factor, input_size, dropout):
        super().__init__()
        self.l1 = nn.Linear(input_size, input_size * expansion_factor)  # Expansion
        self.l2 = nn.Linear(input_size * expansion_factor, input_size)  # Contraction
        nn.init.xavier_normal_(self.l1.weight)
        nn.init.xavier_normal_(self.l2.weight)
        self.layer_norm = nn.LayerNorm(input_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        h = F.relu(self.l1(x))
        h = self.l2(h)
        h = self.dropout(h)
        h = self.layer_norm(h + residual)
        return h


class ScaledDotProductAttention(nn.Module):
    def __init__(self, mask_out_future, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.mask_out_future = mask_out_future

    def forward(self, query, key, value):
        # QK dot product calculation
        sim = torch.einsum("ijk,ljk->jil", query, key)  # Batch, tsQ, tsK
        # Equivalent of query.transpose(0, 1).matmul(key.transpose(0, 1).transpose(1,2))
        # Scaling
        sim = sim / math.sqrt(key.shape[-1])

        # Mask to make it causal (only look at current and previous elements)
        if self.mask_out_future:
            mask = torch.triu(torch.ones_like(sim) * (-np.Inf), diagonal=1)
            sim = sim + mask

        # Softmax
        a = F.softmax(sim, dim=-1)

        a = self.dropout(a)  # Original place. TODO: try moving it downwards

        # Weighted average
        context = torch.einsum("ijk,kil->jil", a, value)
        # Equivalent of a.matmul(value.transpose(0,1)).transpose(0,1)

        return context


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_in, d_out, mask_out_future, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.Wq = nn.Linear(d_in, d_out * n_heads)
        self.Wk = nn.Linear(d_in, d_out * n_heads)
        self.Wv = nn.Linear(d_in, d_out * n_heads)
        nn.init.xavier_normal_(self.Wq.weight)
        nn.init.xavier_normal_(self.Wk.weight)
        nn.init.xavier_normal_(self.Wv.weight)

        self.attention = ScaledDotProductAttention(
            mask_out_future=mask_out_future, dropout=dropout
        )
        self.layer_norm = nn.LayerNorm(d_out)
        self.linear = nn.Linear(d_out * n_heads, d_in)
        nn.init.xavier_normal_(self.linear.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        residual = query
        n_ts_q, batch_sz, n_feats_q = query.shape
        n_ts_k, batch_sz, n_feats_k = key.shape
        n_ts_v, batch_sz, n_feats_v = value.shape
        n_heads = self.n_heads

        # Linear transformations; expand to n_heads
        # (TS, BS, F) -> (TS, BS, H*F)
        query = self.Wq(query)
        key = self.Wk(key)
        value = self.Wv(value)

        # Shapes adjustment
        # (TS, BS, H*F) -> (TS, BS*H, F)
        query = query.view(n_ts_q, batch_sz * n_heads, n_feats_q)
        key = key.view(n_ts_k, batch_sz * n_heads, n_feats_k)
        value = value.view(n_ts_v, batch_sz * n_heads, n_feats_v)

        # Scaled dot product
        c = self.attention(query, key, value)

        # Shapes adjustment
        c = c.reshape(n_ts_q, batch_sz, -1)
        c = self.linear(c)

        # Dropout
        c = self.dropout(c)

        # Residual + LN
        c = self.layer_norm(c + residual)
        return c


def generate_positional_embedding(length, channels):
    # Adapted from https://github.com/tensorflow/tensor2tensor
    max_timescale = 10e4
    position = torch.arange(length)
    num_timescales = channels // 2
    log_timescale_increment = math.log(float(max_timescale)) / (num_timescales - 1)
    inv_timescales = torch.exp(torch.arange(num_timescales) * -log_timescale_increment)
    scaled_time = position[:, None] * inv_timescales[None]
    signal = torch.cat([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    signal = torch.reshape(signal, [1, length, channels])
    signal = signal.transpose(0, 1)  # Pytorch format, time at the beginning
    return signal
