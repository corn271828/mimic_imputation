import torch
import torch.nn as nn
from torch.optim import Adam
from typing import List, Tuple

import numpy as np
from sklearn import preprocessing

import tmip.evaluation as evaluation
import tmip.layers as layers
import tmip.utils as utils

SERIES_DIM=15
ADM_DIM=5
class Trainer(object):
  def __init__(self, model_name, task_name, model_seed, series_feature_num,
      max_iter=12500, batch_size=256, base_lr=0.001, verbose=True):

    self.batch_size = batch_size
    # assuming gpu by default
    self.device = torch.device('cuda')
    self.verbose = verbose

    self.train_loss = utils.Averager()

    self.task_name = task_name
    self.output_dim = 1
    if task_name in {'mor'}:
      pos_weight = torch.Tensor([10.0]).to(self.device)
      self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
      self.max_iter = max_iter // 5
      self.y_transform = utils.IdentityTransform()
    elif task_name in {'icd9'}:
      self.output_dim = 20
      self.loss_fn = nn.BCEWithLogitsLoss()
      self.max_iter = max_iter // 2
      self.y_transform = utils.IdentityTransform()
    else:
      # self.loss_fn = nn.SmoothL1Loss()
      self.loss_fn = nn.MSELoss()
      self.max_iter = max_iter // 5
      self.y_transform = utils.IdentityTransform()

    if model_name in 'mlp':
      input_dim = series_feature_num * SERIES_DIM + ADM_DIM
      self.model = layers.MLP(input_dim, output_dim=self.output_dim)
    elif model_name in 'linear':
      input_dim = series_feature_num * SERIES_DIM + ADM_DIM
      self.model = layers.LinearNet(input_dim, output_dim=self.output_dim)
    elif model_name in 'ffn':
      input_dim = series_feature_num * SERIES_DIM + ADM_DIM
      self.model = layers.PositionWiseFFN(input_dim, output_dim=self.output_dim)
    elif model_name in 'mmdn':
      self.model = layers.MultiModalNet(SERIES_DIM, ADM_DIM, output_dim=self.output_dim)

    self.model.to(self.device)
    self.optimizer = Adam(self.model.parameters(), lr=base_lr, weight_decay=0.00001)

  def prepare_data(self, data):
    if isinstance(data, List) or isinstance(data, Tuple):
      output_data = []
      for datum in data:
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
      return output_data
    else:
      return data[start:end, :]

  def fit(self, X_train, y_train, X_eval, y_eval):
    self.model.train()

    self.y_transform.fit(y_train[:, np.newaxis])
    y_train = self.y_transform.transform(y_train[:, np.newaxis]).squeeze()

    X_train = self.prepare_data(X_train)
    y_train = self.prepare_data(y_train)

    if y_train.dim() == 1:
      y_train = y_train.unsqueeze(-1)

    bsz = self.batch_size
    N = y_train.size(0)
    for i in range(self.max_iter):
      start = i*bsz % N
      end = min(start+bsz, N)
      X_batch = self.prepare_batch(X_train, start, end)
      y_batch = self.prepare_batch(y_train, start, end)

      self.optimizer.zero_grad()
      y_pred = self.model(X_batch)
      loss = self.loss_fn(y_pred, y_batch)

      loss.backward()
      self.optimizer.step()

      self.train_loss.add(loss.item())
      if self.verbose and (i + 1) % 500 == 0:
        y_output = self.predict(X_eval)
        metrics = evaluation.eval_metrics(self.task_name, y_output, y_eval)
        print('step#{} loss={:.4f}, eval_metrics={}'.format(i+1, self.train_loss.item(), metrics))
        self.model.train()

  def predict(self, X_eval):
    self.model.eval()

    X_eval = self.prepare_data(X_eval)
    with torch.no_grad():
      y_eval = self.model(X_eval)

    y_output = y_eval.cpu().detach().numpy()
    return self.y_transform.inverse_transform(y_output)
