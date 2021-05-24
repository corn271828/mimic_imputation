import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from tmip.gpo import GPO

def positional_encoding_1d(d_model, length):
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                          -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe

class MultiModalImputer(nn.Module):
  def __init__(self, series_dim, adm_dim, output_dim, series_length, num_layers=2, hidden_size=16, rnn_class='gru'):
    super(MultiModalImputer, self).__init__()
    self.series_dim = series_dim

    # Admission model
    self.adm_mlp = MLP(adm_dim, hidden_size, hidden_size)

    # Series model
    self.datum_emb_size = 8
    self.d_model = hidden_size if rnn_class in {'gru'} else 2*hidden_size

    self.series_input = nn.ModuleList([nn.Linear(1, self.datum_emb_size, bias=False) for _ in range(series_dim)])
    self.mask_embedding = nn.Embedding(series_dim+1, self.datum_emb_size, padding_idx=0)
    self.pos_encoding = positional_encoding_1d(self.datum_emb_size, series_length).unsqueeze(0)

    self.series_rnn = nn.GRU(self.datum_emb_size*series_dim, self.d_model*series_dim, num_layers, batch_first=True, bidirectional=True, dropout=0.5)

    self.output = nn.ModuleList([PositionWiseFFN(hidden_size*3, output_dim) for _ in range(series_dim)])

  def forward(self, x, mask):
    series, adm = x
    N, L = series.size()[:2]

    # compute adm embedding
    adm_output = self.adm_mlp(adm).unsqueeze(1)
    adm_output = adm_output.expand([N, L, -1])

    # compute series embedding
    input_array = []
    series_array = torch.chunk(series, self.series_dim, dim=-1)
    mask_array = torch.chunk(mask, self.series_dim, dim=-1)
    for module_id, (series_item, mask_item) in enumerate(zip(series_array, mask_array)):
      mask_emb = self.mask_embedding((mask_item.squeeze()*(module_id+1)).long())
      seq_emb = self.series_input[module_id](series_item)
      input_array.append(mask_emb + seq_emb)
    series_rnn_input = torch.cat(input_array, dim=-1)

    # rnn forward
    series_rnn_output, _ = self.series_rnn(series_rnn_input)

    series_logits = []
    series_rnn_array = torch.chunk(series_rnn_output, self.series_dim, dim=-1)
    for module_id, series_rnn_item in enumerate(series_rnn_array):
      series_logit = self.output[module_id](torch.cat([series_rnn_item, adm_output], dim=-1))
      series_logits.append(series_logit)

    return torch.cat(series_logits, dim=-1)

class LinearNet(nn.Module):
  def __init__(self, input_dim, output_dim):
    super(LinearNet, self).__init__()

    self.fc1 = nn.Linear(input_dim, output_dim)

  def forward(self, x):
    return self.fc1(x)

class MLP(nn.Module):
  def __init__(self, input_dim, output_dim, hidden_size=512):
    super(MLP, self).__init__()

    self.input_dropout = nn.Dropout(0.1)
    self.fc1 = nn.Linear(input_dim, hidden_size)
    self.fc2 = nn.Linear(hidden_size, output_dim)
    self.dropout = nn.Dropout(0.5)

  def forward(self, x):
    h = self.dropout(self.fc1(self.input_dropout(x)))
    return self.fc2(F.relu(h, inplace=True))

class PositionWiseFFN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=512, dropout=0.5):
        super(PositionWiseFFN, self).__init__()
        self._dropout = dropout
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim),
        )

    def forward(self, X):
        ffn = self.ffn(X)

        return ffn

    def init_weight(self):
        for idx in range(len(self.ffn)):
            if hasattr(self.ffn[idx], "weight"):
                nn.init.uniform_(self.ffn[idx].weight, -0.1, 0.1)

class MultiModalNet(nn.Module):
  def __init__(self, series_dim, adm_dim, output_dim, num_layers=2, hidden_size=32, rnn_class='gru'):
    super(MultiModalNet, self).__init__()

    self.adm_dropout = nn.Dropout(0.1)
    self.series_dropout = nn.Dropout(0.1)

    if rnn_class == 'gru':
      self.d_model = hidden_size
      self.series_rnn = nn.GRU(series_dim, self.d_model, num_layers, batch_first=True, bidirectional=True, dropout=0.5)
    elif rnn_class == 'transformer':
      self.d_model = 2*hidden_size
      encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=4, dim_feedforward=2*hidden_size, dropout=0.1)
      encoder_norm = nn.LayerNorm(self.d_model)

      self.series_input = nn.Sequential(
            nn.Linear(series_dim, self.d_model),
            nn.Dropout(0.5),
          )
      self.series_rnn = nn.TransformerEncoder(encoder_layer, num_layers, encoder_norm)

    self.adm_mlp = MLP(adm_dim, hidden_size, hidden_size)
    # self.output = MLP(3*hidden_size, output_dim)
    self.output = PositionWiseFFN(3*hidden_size, output_dim)
    self.pool = GPO(32, 32)

  def forward(self, x):
    series, adm = x
    N, L = series.size()[:2]

    # compute adm embedding
    adm_output = self.adm_mlp(adm)

    # compute series embedding
    if isinstance(self.series_rnn, nn.GRU):
      series_emb, _ = self.series_rnn(series)
    elif isinstance(self.series_rnn, nn.TransformerEncoder):
      pos_encoding = positional_encoding_1d(self.d_model, L).unsqueeze(0).expand([N, L, -1])
      pos_encoding = pos_encoding.transpose(0, 1).to(series.device)
      series_emb = self.series_input(series.transpose(0, 1))
      series_emb = self.series_rnn(series_emb + pos_encoding)
      series_emb = series_emb.transpose(0, 1)
    else:
      raise ValueError('Unknown rnn model')

    series_output, _ = self.pool(series_emb)
    output = self.output(torch.cat([series_output, adm_output], dim=-1))

    return output

class Discriminator(nn.Module):
  def __init__(self, series_dim, adm_dim, series_length):
    super(Discriminator, self).__init__()
    self.input_dim = series_dim * series_length + adm_dim
    self.network = PositionWiseFFN(self.input_dim, 1)

  def forward(self, x):
    series, adm = x
    N, L = series.size()[:2]
    model_input = torch.cat([series.view(N, -1), adm], axis=-1)
    return self.network(model_input)
