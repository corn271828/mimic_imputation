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
class Imputer(object):
  def __init__(self, model_seed, series_feature_length, miss_rate,
      max_iter=5000, batch_size=256, base_lr=0.001, verbose=True):

    self.miss_rate = miss_rate
    self.batch_size = batch_size
    # assuming gpu by default
    self.device = torch.device('cuda')
    self.verbose = verbose

    self.train_loss = utils.Averager()

    self.output_dim = 1

    self.loss_fn = nn.SmoothL1Loss(reduction='none')
    self.max_iter = max_iter

    self.model = layers.MultiModalImputer(
                      SERIES_DIM,
                      ADM_DIM,
                      self.output_dim,
                      series_feature_length
                    )

    self.model.to(self.device)
    self.optimizer = Adam(self.model.parameters(), lr=base_lr, weight_decay=0.00001)

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

  def fit(self, X_train, y_train, X_eval, y_eval):
    self.start_training()

    bsz = self.batch_size
    N = y_train[0].shape[0]
    for i in range(self.max_iter):
      start = i*bsz % N
      end = min(start+bsz, N)
      X_batch = self.prepare_batch(X_train, start, end)
      X_batch, mask_batch = self.mask_data(X_batch, random_seed=np.random.randint(9999), padding_value=0)
      X_batch = self.prepare_data(X_batch)

      y_batch = self.prepare_data(self.prepare_batch(y_train, start, end)[0])
      mask_batch = self.prepare_data(mask_batch)

      # Forward step
      self.optimizer.zero_grad()
      y_pred = self.model(X_batch, mask_batch)

      loss = self.loss_fn(y_pred, y_batch) * mask_batch
      loss = loss.sum() / mask_batch.sum()

      # Optimization step
      loss.backward()
      self.optimizer.step()
      self.train_loss.add(loss.item())

      # Logging
      if self.verbose and (i + 1) % 50 == 0:
        self.start_evaluating()
        X_eval_, mask_eval = self.mask_data(X_eval, random_seed=np.random.randint(9999), padding_value=0)
        y_output = self.predict(X_eval_, mask_eval)
        mse_eval = (y_output * mask_eval - y_eval[0] * mask_eval)**2
        mse_eval = mse_eval.sum() / mask_eval.sum()
        print('step#{} loss={:.4f}, eval_metrics={}'.format(i+1, self.train_loss.item(), mse_eval))
        self.start_training()

  def predict(self, X_eval, mask_eval):
    self.start_evaluating()

    X_eval = self.prepare_data(X_eval)
    mask_eval = self.prepare_data(mask_eval)
    with torch.no_grad():
      y_eval = self.model(X_eval, mask_eval)
      y_eval = y_eval*mask_eval + X_eval[0]

    y_output = y_eval.cpu().detach().numpy()
    return y_output

  def start_training(self):
    self.model.train()

  def start_evaluating(self):
    self.model.eval()

  def mask_data(self, data, random_seed, padding_value=0):
    miss_rate = self.miss_rate
    X, adm = data
    np.random.seed(random_seed)
    mask = np.random.uniform(size=X.shape) < miss_rate
    Xhat = X.copy()
    Xhat[mask] = padding_value

    return [Xhat, adm], mask
