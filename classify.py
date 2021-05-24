import os
import ipdb
import pickle
import argparse
import numpy as np

import tmip.dataset as dataset
import tmip.evaluation as evaluation

from tmip.classifier import Trainer

attribute_names = [
    'age',
    'acquired immunodeficiency syndrome',
    'hematologic malignancy',
    'metastatic cancer',
    'admission type',
]

INPUT_FFEATURE_FILEPATH = 'data/processed/{}hrs.data.pkl'
def _commandline_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--feature_type', type=int, default=24, choices=[24, 48])
  parser.add_argument('--task_name', required=True, choices=['mor', 'los', 'icd9'])
  parser.add_argument('--model_name', required=True, choices=['mmdn', 'linear', 'mlp', 'ffn'])
  parser.add_argument('--impute_methods', type=str, nargs='+', default=['bayesian', 'adv', 'seq'])
  parser.add_argument('--miss_rate', type=float, default=0.2)
  parser.add_argument('--seed', type=int, default=1001)
  parser.add_argument('--model_seed', type=int, default=1)

  return parser

def main(args):
  input_features = INPUT_FFEATURE_FILEPATH.format(args.feature_type)
  with open(input_features, 'rb') as fd:
    data = pickle.load(fd)

  imputed_data_filepath = '{:.2f}_{:4d}_{}.pkl'.format(args.miss_rate, args.seed, args.feature_type)
  print('Preparing data....')
  X_train_raw, y_train, X_val_raw, y_val = dataset.prepare_classification_data(data,
      task_name=args.task_name,
      seed=args.seed,
      miss_rate=args.miss_rate,
      imputed_filepath=os.path.join('imputed', '{}_'+imputed_data_filepath),
      impute_methods=args.impute_methods,
      )

  if args.model_name in {'linear', 'mlp', 'ffn'}:
    processor = dataset.Preprocessor('normal')
  elif args.model_name in {'mmdn'}:
    processor = dataset.Preprocessor('sequential')

  processor.fit(X_train_raw)
  X_train = processor.preprocess(X_train_raw)
  X_val   = processor.preprocess(X_val_raw['perfect'])

  print('Fitting model....')
  learner = Trainer(args.model_name, args.task_name, args.model_seed, args.feature_type)
  learner.fit(X_train, y_train, X_val, y_val)

  print('Predicting on perfect eval data....')
  yhat_val = learner.predict(X_val)
  metrics = evaluation.eval_metrics(args.task_name, yhat_val, y_val, verbose=True)
  print(metrics)

  print('Predicting on missing eval data....')
  X_val = processor.preprocess(X_val_raw['missing'])
  yhat_val = learner.predict(X_val)
  metrics = evaluation.eval_metrics(args.task_name, yhat_val, y_val, verbose=True)
  print(metrics)

  mape = 0
  print('Predicting on statistically imputed eval data....')
  X_val = processor.preprocess(X_val_raw['mean_imputed'])
  yhat_val = learner.predict(X_val)
  metrics = evaluation.eval_metrics(args.task_name, yhat_val, y_val, verbose=True)
  print(metrics)
  mse = evaluation._evalMSE(X_val_raw['perfect'][0], X_val_raw['mean_imputed'][0])
  print('Direct Imputation Outcome: MSE={:.2f}, MAPE={:.2f}'.format(mse, mape))

  for impute_method in args.impute_methods:
    key = impute_method + '_imputed'
    if X_val_raw.get(key, None):
      print('Predicting on {} model imputed eval data....'.format(impute_method))
      X_val = processor.preprocess(X_val_raw[key])
      yhat_val = learner.predict(X_val)
      metrics = evaluation.eval_metrics(args.task_name, yhat_val, y_val, verbose=True)
      print(metrics)

      mse = evaluation._evalMSE(X_val_raw['perfect'][0], X_val_raw[key][0])
      print('Direct Imputation Outcome: MSE={:.2f}, MAPE={:.2f}'.format(mse, mape))


if __name__ == '__main__':
  parser = _commandline_parser()
  args = parser.parse_args()

  with ipdb.launch_ipdb_on_exception():
    main(args)
