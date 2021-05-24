import os
import pickle
import argparse
import numpy as np


def _commandline_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--output_filepath', type=str, default='data/processed/')
  parser.add_argument('--split', type=str, default='24hrs')

  return parser

def main(args):
  filepath = os.path.join('data', args.split, 'data.npz')

  data = np.load(filepath, allow_pickle=True)
  series_data = data['ep_tdata']
  series_mask = data['ep_tdata_masking']

  mortality_label = data['y_mor']
  icd9_label = data['y_icd9']
  len_stays_label = data['y_los'] / 60.0 # convert to hours

  adm_features = data['adm_features_all']
  adm_features[:, 0] /= 365.25 # use age in years

  with open(os.path.join('data/processed', args.split+'.data.pkl'), 'wb') as fd:
    pickle.dump(dict(series_data=series_data,
                     series_mask=series_mask,
                     label_mortality=mortality_label,
                     label_icd9=icd9_label,
                     label_los=len_stays_label,
                     noseries_data=adm_features), fd)

if __name__ == '__main__':
  parser = _commandline_parser()
  args = parser.parse_args()


  main(args)
