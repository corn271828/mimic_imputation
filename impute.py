import os
import pickle
import argparse
import numpy as np
import ipdb

import tmip.dataset as dataset
import tmip.evaluation as evaluation

from tmip.imputer import Imputer
from tmip.adv_imputer import AdversarialImputer

INPUT_FFEATURE_FILEPATH = 'data/processed/{}hrs.data.pkl'
def _commandline_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--feature_type', type=int, default=24, choices=[24, 48])
  parser.add_argument('--model_name', default='seq', choices=['adv', 'seq'])
  parser.add_argument('--model_seed', type=int, default=1)
  parser.add_argument('--seeds', type=int, nargs='+', default=[1001,1002,1003,1004,1005])
  parser.add_argument('--miss_rate', type=float, default=0.2)

  return parser


def main(args):
  input_features = INPUT_FFEATURE_FILEPATH.format(args.feature_type)
  with open(input_features, 'rb') as fd:
    data = pickle.load(fd)

  print('Preparing data....')
  X_train_raw, Y_train_raw, X_test_raw, Y_test_raw = dataset.prepare_imputation_data(data)

  processor = dataset.Preprocessor('sequential')
  processor.fit(X_train_raw)

  X_train = processor.preprocess(X_train_raw)
  X_test  = processor.preprocess(X_test_raw)
  Y_train = processor.preprocess(Y_train_raw)
  Y_test  = processor.preprocess(Y_test_raw)

  print('Fitting model....')
  if args.model_name == 'seq':
    learner = Imputer(args.model_seed, args.feature_type, args.miss_rate)
  elif args.model_name == 'adv':
    learner = AdversarialImputer(args.model_seed, args.feature_type, args.miss_rate)
  learner.fit(X_train, Y_train, X_test, Y_test)

  for data_seed in args.seeds:
    _, _, X_output_raw, _ = dataset.prepare_classification_data(data,
        task_name='mor',
        seed=data_seed,
        miss_rate=args.miss_rate,
        padding_value=np.nan
        )
    X_output = processor.preprocess(X_output_raw['missing'])
    X_output_mask = np.isnan(X_output[0])
    X_output_imputed = learner.predict(X_output, X_output_mask)
    X_output_imputed = processor.postprocess_series(X_output_imputed)

    output_filename = '{}_{:.2f}_{:4d}_{}.pkl'.format(args.model_name, args.miss_rate, data_seed, args.feature_type)
    with open(os.path.join('imputed', output_filename), 'wb') as fd:
      pickle.dump(X_output_imputed, fd)


if __name__ == '__main__':
  parser = _commandline_parser()
  args = parser.parse_args()

  with ipdb.launch_ipdb_on_exception():
    main(args)
