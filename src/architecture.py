import torch
from torch import nn
import numpy as np
from src.constants import embedding_sizes


class Seq2Seq(nn.Module):
    def __init__(
        self,
        n_num_time_feats,
        cardinalities_time,
        cardinalities_static,
        n_forecast_timesteps,
        lr,
    ):
        super().__init__()
        self.encoder = Encoder(
            n_num_time_feats=n_num_time_feats,
            categorical_cardinalities=cardinalities_time,
        )
        self.decoder = Decoder(
            n_forecast_timesteps=n_forecast_timesteps,
            categorical_cardinalities=cardinalities_static,
        )
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=lr)

    def forward(
        self, x_num_time, x_cat_time, x_cat_static, cat_time_names, cat_static_names
    ):
        contextual_thought = self.encoder.forward(
            x_num_time=x_num_time, x_cat_time=x_cat_time, cat_time_names=cat_time_names
        )
        output = self.decoder.forward(
            x_cat_static=x_cat_static,
            cat_static_names=cat_static_names,
            state=contextual_thought,
        )
        return output

    def loss(
        self,
        x_num_time,
        x_cat_time,
        x_cat_static,
        cat_time_names,
        cat_static_names,
        target,
    ):
        y_hat = self.forward(
            x_num_time=x_num_time,
            x_cat_time=x_cat_time,
            x_cat_static=x_cat_static,
            cat_time_names=cat_time_names,
            cat_static_names=cat_static_names,
        )
        loss = torch_rmse(target, y_hat)
        return loss, y_hat

    def step(
        self,
        x_num_time,
        x_cat_time,
        x_cat_static,
        cat_time_names,
        cat_static_names,
        target,
    ):
        loss, y_hat = self.loss(
            x_num_time=x_num_time,
            x_cat_time=x_cat_time,
            x_cat_static=x_cat_static,
            cat_time_names=cat_time_names,
            cat_static_names=cat_static_names,
            target=target,
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss, y_hat


def torch_rmse(actual, forecast):
    residuals = actual - forecast
    rmse = torch.sqrt(torch.mean((residuals) ** 2))
    return rmse


class Encoder(nn.Module):
    def __init__(self, n_num_time_feats, categorical_cardinalities):
        super().__init__()
        self.embs = {}
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
            # TODO: make all the tensor long to enhance efficiency

        time_features = torch.cat([x_num_time] + emb_feats, -1).squeeze()
        _, state = self.rnn_encoder(time_features)
        return state


class Decoder(nn.Module):
    def __init__(self, n_forecast_timesteps, categorical_cardinalities):
        super().__init__()
        self.n_forecast_timesteps = n_forecast_timesteps
        self.n_recurrent_cells = 128
        self.rnn_decoder = nn.LSTM(input_size=1, hidden_size=self.n_recurrent_cells)
        self.embs = {}

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
        # TODO: Adapt forward looking features
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
            context_thought[:, : self.n_recurrent_cells].unsqueeze(0).contiguous(),
            context_thought[:, self.n_recurrent_cells :].unsqueeze(0).contiguous(),
        )

        input_decoder = torch.zeros(
            self.n_forecast_timesteps, context_thought[0].shape[1], 1
        ).cuda()  # TODO: Parametrize
        input_decoder[0, :, :] = 1  # GO!
        output, _ = self.rnn_decoder(input_decoder, context_thought)
        h = output.reshape(self.n_forecast_timesteps * batch_size, output.shape[-1])
        h = self.time_distributed(h)
        output = h.reshape(self.n_forecast_timesteps, batch_size, 1).squeeze()
        return output
