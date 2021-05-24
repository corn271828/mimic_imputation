import os
import pickle
import argparse
import numpy as np

import tmip.dataset as dataset
import tmip.evaluation as evaluation

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from sklearn.linear_model import BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import ExtraTreesRegressor


ADM_DIM=5
SERIES_DIM=15
INPUT_FFEATURE_FILEPATH = 'data/processed/{}hrs.data.pkl'
def _commandline_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--feature_type', type=int, default=24, choices=[24, 48])
  parser.add_argument('--model_name', type=str, default='bayesian', choices=['knn', 'randomforest', 'bayesian'])
  parser.add_argument('--miss_rate', type=float, default=0.2)
  parser.add_argument('--seeds', type=int, nargs='+', default=[1001,1002,1003,1004,1005])

  return parser

def prepare_estimator(args):
  if args.model_name == 'knn':
    estimator = KNeighborsRegressor(n_neighbors=5)
  elif args.model_name == 'randomforest':
    estimator = ExtraTreesRegressor(n_estimators=10, random_state=0)
  elif args.model_name == 'bayesian':
    estimator = BayesianRidge()

  return estimator

def _evalMSE(labels, preds):
  error = labels - preds
  return (error ** 2).sum(axis=-1).mean()

def main(args):
  input_features = INPUT_FFEATURE_FILEPATH.format(args.feature_type)
  with open(input_features, 'rb') as fd:
    data = pickle.load(fd)

  print('Preparing data....')
  X_train_raw, Y_train_raw, X_test_raw, Y_test_raw = dataset.prepare_imputation_data(data)

  X_train_series, _ = dataset._mask_data(X_train_raw[0], args.miss_rate, 0, padding_value=np.nan)
  X_test_series, _ = dataset._mask_data(X_test_raw[0], args.miss_rate, 1, padding_value=np.nan)

  processor = dataset.Preprocessor('normal')
  processor.fit(X_train_raw)
  X_train = processor.preprocess([X_train_series, X_train_raw[1]])
  X_test  = processor.preprocess([X_test_series, X_test_raw[1]])
  Y_test  = processor.preprocess(Y_test_raw)

  # Imputation Model
  print('Preparing model....')
  est = prepare_estimator(args)
  imp = IterativeImputer(max_iter=5, random_state=0, estimator=est, verbose=True)

  print('Fitting....')
  imp.fit(X_train)

  X_test_imputed = imp.transform(X_test)
  X_test_output = processor.postprocess_series(X_test_imputed)
  print('Global Mean Squared Error: {}'.format(_evalMSE(X_test_output, Y_test[:, :-ADM_DIM])))

  for data_seed in args.seeds:
    _, _, X_output_raw, _ = dataset.prepare_classification_data(data,
        task_name='mor',
        seed=data_seed,
        miss_rate=args.miss_rate,
        padding_value=np.nan
        )
    X_output = processor.preprocess(X_output_raw['missing'])
    X_output_mask = np.isnan(X_output[0])
    X_output_imputed = imp.transform(X_output)
    X_output_imputed = processor.postprocess_series(X_output_imputed)

    output_filename = '{}_{:.2f}_{:4d}_{}.pkl'.format(args.model_name, args.miss_rate, data_seed, args.feature_type)
    with open(os.path.join('imputed', output_filename), 'wb') as fd:
      pickle.dump(X_output_imputed, fd)

if __name__ == '__main__':
  parser = _commandline_parser()
  args = parser.parse_args()

  import ipdb
  with ipdb.launch_ipdb_on_exception():
    main(args)

