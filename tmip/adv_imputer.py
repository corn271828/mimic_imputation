import torch
import torch.nn as nn
from torch.optim import Adam
from typing import List, Tuple

import numpy as np
from sklearn import preprocessing

import tmip.evaluation as evaluation
import tmip.dataset as dataset
import tmip.layers as layers
import tmip.utils as utils

SERIES_DIM=15
ADM_DIM=5
class AdversarialImputer(object):
  def __init__(self, model_seed, series_feature_length, miss_rate,
      max_iter=10000, batch_size=256, base_lr=0.001, verbose=True):

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

    self.g_model = layers.MultiModalImputer(
                      SERIES_DIM,
                      ADM_DIM,
                      self.output_dim,
                      series_feature_length,
                    )
    self.d_model = layers.Discriminator(
                      SERIES_DIM,
                      ADM_DIM,
                      series_feature_length,
                    )

    self.g_model.to(self.device)
    self.d_model.to(self.device)
    self.g_optimizer = Adam(self.g_model.parameters(), lr=base_lr, weight_decay=0.00001)
    self.d_optimizer = Adam(self.d_model.parameters(), lr=base_lr)

    self.gen_interval = 1
    self.disc_interval = 5

  def fit(self, X_train, y_train, X_eval, y_eval):
    self.start_training()

    bsz = self.batch_size
    N = y_train[0].shape[0]
    for i in range(self.max_iter):
      start = i*bsz % N
      end = min(start+bsz, N)

      real_batch = self.prepare_batch(X_train, start, end)
      X_batch, mask_batch = self.mask_data(real_batch, random_seed=np.random.randint(9999), padding_value=0)
      X_batch = self.prepare_data(X_batch)

      y_batch = self.prepare_data(self.prepare_batch(y_train, start, end)[0])
      mask_batch = self.prepare_data(mask_batch)
      real_batch = self.prepare_data(real_batch)

      # effective batch size
      ebsz = mask_batch.shape[0]

      # 1. Train discrminator
      if i % self.disc_interval == 0:
        self.d_optimizer.zero_grad()

        # forward data
        with torch.no_grad():
          y_pred = self.g_model(X_batch, mask_batch)

        fake_batch = y_pred.detach() * mask_batch + X_batch[0]

        d_loss = 0
        d_model_input = [
            torch.cat([real_batch[0], fake_batch], dim=0),
            torch.cat([real_batch[1], real_batch[1]], dim=0),
        ]
        d_model_output = self.d_model(d_model_input)
        d_model_label = torch.cat([torch.ones([ebsz, 1]), torch.zeros([ebsz, 1])]).to(self.device)
        d_loss += self.d_loss_fn(d_model_output, d_model_label)

        d_loss.backward()
        self.d_optimizer.step()
        self.d_loss.add(d_loss.item())

      # 2. Train generator
      if i % self.gen_interval == 0:
        self.g_optimizer.zero_grad()
        # model forward
        y_pred = self.g_model(X_batch, mask_batch)

        fake_batch = y_pred * mask_batch + X_batch[0]

        g_loss = 0
        # Data loss
        data_loss = self.g_loss_fn(y_pred, y_batch) * mask_batch
        g_loss += data_loss.sum() / mask_batch.sum()
        g_loss += 0.1* self.d_loss_fn(self.d_model([fake_batch, real_batch[1]]), torch.ones([ebsz, 1]).to(self.device))

        g_loss.backward()
        self.g_optimizer.step()
        self.g_loss.add(g_loss.item())

      if (i + 1) % 100 == 0:
        self.start_evaluating()
        X_eval_temp, mask_eval = self.mask_data(X_eval, random_seed=np.random.randint(9999), padding_value=0)
        y_output = self.predict(X_eval_temp, mask_eval)
        mse_eval = (y_output * mask_eval - y_eval[0] * mask_eval)**2
        mse_eval = mse_eval.sum() / mask_eval.sum()
        print('step#{} generator loss={:.4f}, discriminator loss={:.4f}, eval_mse={:.4f}'.format(i+1, self.g_loss.item(), self.d_loss.item(), mse_eval))
        self.start_training()

  def predict(self, X_eval, mask_eval):
    self.start_evaluating()

    X_eval = self.prepare_data(X_eval)
    mask_eval = self.prepare_data(mask_eval)

    N = X_eval[0].shape[0]
    with torch.no_grad():
      y_eval = self.g_model(X_eval, mask_eval)
      y_eval = y_eval*mask_eval + X_eval[0]

    y_output = y_eval.cpu().detach().numpy()
    return y_output

  def start_training(self):
    self.g_model.train()
    self.d_model.train()

  def start_evaluating(self):
    self.g_model.eval()
    self.d_model.eval()

  def mask_data(self, data, random_seed, padding_value=0):
    miss_rate = self.miss_rate
    X, adm = data
    np.random.seed(random_seed)
    mask = np.random.uniform(size=X.shape) < miss_rate
    Xhat = X.copy()
    Xhat[mask] = padding_value

    return [Xhat, adm], mask

  def prepare_data(self, data, padding_value=0):
    if isinstance(data, List) or isinstance(data, Tuple):
      output_data = []
      for datum in data:
        # if nan make zero
        datum[np.isnan(datum)] = padding_value
        output_datum = torch.Tensor(datum)
        output_datum = output_datum.to(self.device)
        output_data.append(output_datum)
    else:
      output_data = torch.Tensor(data)
      output_data = output_data.to(self.device)

    return output_data

  def prepare_batch(self, data, start, end):
    if isinstance(data, List) or isinstance(data, Tuple):
      output_data = []
      for datum in data:
        output_data.append(datum[start:end, :])
    else:
      output_data = data[start:end, :]

    return output_data
