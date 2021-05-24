import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam

import utils

def mask_data(X, miss_rate, random_seed):
  np.random.seed(random_seed)
  mask = np.random.uniform(size=X.shape) < miss_rate
  Xhat = X.copy()
  Xhat[mask] = 0

  return Xhat, mask

class GANImputer(object):
  def __init__(self, feature_length, miss_rate, category_mask,
      max_iter=1000, batch_size=128, base_lr=0.001, verbose=True):

    self.miss_rate = miss_rate
    self.batch_size = batch_size
    # assuming gpu by default
    self.device = torch.device('cuda')

    self.g_loss = utils.Averager()
    self.d_loss = utils.Averager()

    self.output_dim = 1

    self.g_loss_fn = nn.SmoothL1Loss(reduction='none')
    self.d_loss_fn = nn.BCEWithLogitsLoss()
    self.max_iter = max_iter

    self.g_model = Generator(feature_length, 1)
    self.d_model = PositionWiseFFN(feature_length, 1)

    self.g_model.to(self.device)
    self.d_model.to(self.device)
    self.g_optimizer = Adam(self.g_model.parameters(), lr=base_lr, weight_decay=0.00001)
    self.d_optimizer = Adam(self.d_model.parameters(), lr=base_lr, weight_decay=0.00001)

    self.category_mask = torch.BoolTensor(category_mask).to(self.device)

  def fit(self, X, y):
    self.start_training()

    bsz = self.batch_size
    N = y.shape[0]
    for i in range(self.max_iter):
      start = i*bsz % N
      end = min(start+bsz, N)

      real_batch = self.prepare_batch(X, start, end)
      X_batch, mask_batch = mask_data(real_batch, self.miss_rate, random_seed=np.random.randint(9999))

      # effective batch size
      ebsz = real_batch.shape[0]
      # convert numpy to tensor
      X_batch = self.prepare_data(X_batch)
      mask_batch = self.prepare_data(mask_batch)
      real_batch = self.prepare_data(real_batch)

      y_batch = self.prepare_data(self.prepare_batch(y, start, end))
      y_pred = self.g_model(X_batch, mask_batch)

      # 1. Train discrminator
      fake_batch = y_pred.detach() * mask_batch + X_batch

      for _ in range(5):
        self.d_optimizer.zero_grad()
        d_loss = 0
        d_loss += self.d_loss_fn(self.d_model(real_batch), torch.ones([ebsz, 1]).to(self.device))
        d_loss += self.d_loss_fn(self.d_model(fake_batch), torch.zeros([ebsz, 1]).to(self.device))

        d_loss.backward()
        self.d_optimizer.step()
        self.d_loss.add(d_loss.item())

      # 2. Train generator
      self.g_optimizer.zero_grad()
      fake_batch = y_pred * mask_batch + X_batch

      g_loss = 0
      # Data loss
      data_loss = self.g_loss_fn(y_pred, y_batch) * mask_batch
      g_loss += data_loss.sum() / mask_batch.sum()
      g_loss += self.d_loss_fn(self.d_model(fake_batch), torch.zeros([ebsz, 1]).to(self.device))

      g_loss.backward()
      self.g_optimizer.step()
      self.g_loss.add(g_loss.item())

      if (i + 1) % 50 == 0:
        print('step#{} generator loss={:.4f}, discriminator loss={:.4f}'.format(i+1, self.g_loss.item(), self.d_loss.item()))

  def transform(self, X_eval, mask_eval):
    self.start_evaluating()

    X = self.prepare_data(X_eval)
    mask = self.prepare_data(mask_eval)

    N = X.shape[0]
    with torch.no_grad():
      y_eval = self.g_model(X, mask)
      cat_mask = self.category_mask.unsqueeze(0).expand([N, 15])
      y_eval[cat_mask] = (y_eval[cat_mask] > 0.5).float()

    y_output = y_eval.cpu().detach().numpy() * mask_eval + X_eval
    return y_output

  def start_training(self):
    self.g_model.train()
    self.d_model.train()

  def start_evaluating(self):
    self.g_model.eval()
    self.d_model.eval()

  def prepare_data(self, data):
    output_data = torch.Tensor(data)
    output_data = output_data.to(self.device)

    return output_data

  def prepare_batch(self, data, start, end):
    return data[start:end, :]

class Generator(nn.Module):
  def __init__(self, column_num, output_dim, hidden_size=16):
    super(Generator, self).__init__()
    self.column_num = column_num

    # module-specific constant
    datum_emb_size = 8

    self.data_input = nn.ModuleList([nn.Linear(1, datum_emb_size, bias=False) for _ in range(column_num)])
    self.mask_embedding = nn.Embedding(column_num+1, datum_emb_size, padding_idx=0)

    self.xinfo_network = PositionWiseFFN(datum_emb_size*column_num, column_num*hidden_size)
    self.output = nn.ModuleList([PositionWiseFFN(hidden_size, output_dim) for _ in range(column_num)])

  def forward(self, x, mask):
    N, L = x.size()

    # compute data embedding
    input_array = []
    data_array = torch.chunk(x, self.column_num, dim=-1)
    mask_array = torch.chunk(mask, self.column_num, dim=-1)
    for module_id, (data_item, mask_item) in enumerate(zip(data_array, mask_array)):
      mask_emb = self.mask_embedding((mask_item.squeeze()*(module_id+1)).long())
      data_emb = self.data_input[module_id](data_item)
      input_array.append(mask_emb + data_emb)
    xinfo_input = torch.cat(input_array, dim=-1)

    # rnn forward
    xinfo_output = self.xinfo_network(xinfo_input)

    data_logits = []
    xinfo_array = torch.chunk(xinfo_output, self.column_num, dim=-1)
    for module_id, data_item in enumerate(xinfo_array):
      data_logits.append(self.output[module_id](data_item))

    return torch.cat(data_logits, dim=-1)

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
