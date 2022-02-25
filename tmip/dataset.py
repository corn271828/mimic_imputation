import os
import pickle
import numpy as np
from sklearn import preprocessing

import tmip.utils as utils

ADM_DIM=5
SERIES_DIM=15
def _load_imputed_data(series_data, miss_rate, seed):
  series_data, series_mask = _mask_data(series_data, miss_rate, seed, padding_value=-1)

  series_data[series_mask] = np.nan
  output_data = _preprocess_series_data(series_data)

  return output_data

def _mask_data(X, miss_rate, random_seed, padding_value=np.nan, verbose=True):
  np.random.seed(random_seed)
  mask = np.random.uniform(size=X.shape) < miss_rate
  Xhat = X.copy()
  Xhat[mask] = padding_value

  if verbose:
    print('Missing data ratio: {:.2f} %'.format(mask.sum() / np.prod(X.shape) * 100))

  return Xhat, mask

def _generate_train_val_split(N, seed, trainval_ratio=0.8):
  # Split train/val dat
  np.random.seed(seed) # set random seed
  permuted_ids = np.random.permutation(N)
  train_ids, val_ids = permuted_ids[:int(N * trainval_ratio)], permuted_ids[int(N * trainval_ratio):]

  return train_ids, val_ids

def _impute_adm_data(adm_data):
  # remove imputed data from original series data
  adm_mean = np.nanmean(adm_data, axis=0)
  xs, ys = np.where(np.isnan(adm_data))
  for x, y in zip(xs.tolist(), ys.tolist()):
    adm_data[x, y] = adm_mean[y]

  return adm_data

def _impute_series_data(series_data):
  original_shape = series_data.shape
  series_data = series_data.reshape(-1, SERIES_DIM)
  # remove imputed data from original series data
  series_mean = np.nanmean(series_data, axis=0)
  xs, ys = np.where(np.isnan(series_data))
  for x, y in zip(xs.tolist(), ys.tolist()):
    series_data[x, y] = series_mean[y]

  return series_data.reshape(original_shape)

def _preprocess_series_data(series_data):
  ''' Takes in a 3-dimensional data structure, 
  replaces outliers with np.nan, then replaces 
  nans with the mean along axis=0.
  
  Keyword arguments:
  series_data -- 3-dimensional numpy array
  '''
  
  # filtering outlier
  series_data[series_data == 9999999] = np.nan

  # remove imputed data from original series data
  per_patient_mean = np.nanmean(series_data, axis=0)
  xs, ys, zs = np.where(np.isnan(series_data))
  for x, y, z in zip(xs.tolist(), ys.tolist(), zs.tolist()):
    series_data[x, y, z] = per_patient_mean[y, z]

  return series_data

def prepare_imputation_data(data, seed=1001):
  series_data = data['series_data'].copy()
  series_mask = data['series_mask'].copy()
  normal_data = data['noseries_data'].copy()

  # Pre-process data, remove nan fields
  series_data = _preprocess_series_data(series_data)

  series_label = series_data.copy()
  normal_label = normal_data.copy()

  # Generate train/val splits
  train_ids, val_ids = _generate_train_val_split(series_data.shape[0], seed)

  train_data = [series_data[train_ids], normal_data[train_ids]]
  train_label = [series_label[train_ids], normal_label[train_ids]]

  val_data = [series_data[val_ids], normal_data[val_ids]]
  val_label = [series_label[val_ids], normal_label[val_ids]]

  return train_data, train_label, val_data, val_label

def prepare_classification_data(
    data,
    task_name,
    seed=1001,
    miss_rate=0.2,
    padding_value=-1,
    imputed_filepath='',
    impute_methods=[]):
  series_data = data['series_data']
  series_mask = data['series_mask']
  normal_data = data['noseries_data']

  # Pre-process data, remove nan fields
  series_data = _preprocess_series_data(series_data)

  if 'mor' in task_name:
    label = data['label_mortality']
  elif 'los' in task_name:
    label = data['label_los']
  elif 'icd9' in task_name:
    label = data['label_icd9']

  # Generate train/val splits
  train_ids, val_ids = _generate_train_val_split(series_data.shape[0], seed)

  train_data = [series_data[train_ids], normal_data[train_ids]]
  train_label = label[train_ids]
  val_label = label[val_ids]

  val_series_data = series_data[val_ids]
  val_adm_data = normal_data[val_ids]

  # Make val datasets
  val_datasets = dict()

  # 1. Perfect data
  val_datasets['perfect'] = [val_series_data, val_adm_data.copy()]

  # 2. Missing data
  masked_val_series_data, series_mask = _mask_data(val_series_data, miss_rate, seed, padding_value=padding_value)
  # masked_val_adm_data, adm_mask = _mask_data(val_adm_data, miss_rate, seed, padding_value=padding_value)
  # val_datasets['missing'] = [masked_val_series_data.copy(), masked_val_adm_data.copy()]
  val_datasets['missing'] = [masked_val_series_data.copy(), val_adm_data.copy()]

  # 3. Stats imputed data
  # masked_val_adm_data[adm_mask] = np.nan
  masked_val_series_data[series_mask] = np.nan
  val_datasets['mean_imputed'] = [_impute_series_data(masked_val_series_data), val_adm_data.copy()]

  # 4. Model Imputed Data
  for impute_method in impute_methods:
    infilled_filepath = imputed_filepath.format(impute_method)
    if os.path.exists(infilled_filepath):
      with open(infilled_filepath, 'rb') as fd:
        imputed_val_series_data = pickle.load(fd)
        if len(imputed_val_series_data.shape) == 2:
          N = imputed_val_series_data.shape[0]
          imputed_val_series_data = imputed_val_series_data.reshape([N, -1, SERIES_DIM])

      val_datasets['{}_imputed'.format(impute_method)] = [imputed_val_series_data, val_adm_data.copy()]

  return train_data, train_label, val_datasets, val_label

class Preprocessor():
  def __init__(self, _format='normal'):
    assert _format in {'normal', 'sequential'}, 'Unsupported data preprocessor'
    self.format = _format

    self.series_max = None
    self.series_min = None
    self.series_scalar = None

    self.adm_max = None
    self.adm_min = None
    self.adm_scaler = None

  def fit(self, data):
    series_data, adm_data = data

    N, L = series_data.shape[:-1]

    # series_data = series_data.reshape(N*L, -1)
    series_data = series_data.reshape(N, -1)

    # fitting scaler
    self.series_scaler = preprocessing.StandardScaler().fit(series_data)
    self.adm_scaler = preprocessing.StandardScaler().fit(adm_data)

    self.series_max = np.nanmax(series_data, axis=0, keepdims=True)
    self.series_min = np.nanmin(series_data, axis=0, keepdims=True)

    self.adm_max = np.nanmax(adm_data, axis=0, keepdims=True)
    self.adm_min = np.nanmin(adm_data, axis=0, keepdims=True)

  def preprocess(self, data):
    assert self.series_scaler is not None
    assert self.adm_scaler is not None

    series_data, adm_data = data.copy()

    N, L = series_data.shape[:-1]
    # series_data = series_data.reshape(N*L, -1)
    series_data = series_data.reshape(N, -1)

    # transform with scaler
    series_data = self.series_scaler.transform(series_data).reshape(N, -1)
    adm_data = self.adm_scaler.transform(adm_data).reshape(N, -1)

    if self.format in {'normal'}:
      output = np.concatenate([series_data.reshape(N, -1), adm_data], axis=1)
    elif self.format in {'sequential'}:
      output = [series_data.reshape(N, L, -1), adm_data]

    return output

  def postprocess_series(self, data):
    assert self.series_scaler is not None

    if self.format in {'normal'}:
      N = data.shape[0]
      series_data = data[:, :-ADM_DIM].copy()
      series_data = self.series_scaler.inverse_transform(series_data)
      series_data = series_data.clip(self.series_min, self.series_max)
    elif self.format in {'sequential'}:
      N, L = data.shape[:2]
      # series_data = data.copy().reshape([N*L, SERIES_DIM])
      series_data = data.copy().reshape([N, L*SERIES_DIM])
      series_data = self.series_scaler.inverse_transform(series_data)
      series_data = series_data.clip(self.series_min, self.series_max)
      series_data = series_data.reshape([N, L, SERIES_DIM])
    else:
      raise NotImplementedError('Oops...')

    return series_data

def SAPSIITransform(X):
    '''
    [('GCS', 0)], 'mengcz_vital_ts': [('SysBP_Mean', 1), ('HeartRate_Mean', 2), ('TempC_Mean', 3)],
    'mengcz_pao2fio2_ts': [('PO2', 4), ('FIO2', 5)], 'mengcz_urine_output_ts': [('UrineOutput', 6)],
    'mengcz_labs_ts': [('BUN_min', 7), ('WBC_min', 8), ('BICARBONATE_min', 9), ('SODIUM_min', 10),
    ('POTASSIUM_min', 11), ('BILIRUBIN_min', 12)]
    age: 0, aids: 1, he,: 2, mets: 3, admissiontype: 4
    '''
    serial = np.copy(X[0])
    non_serial = np.copy(X[1])

    for admid in range(non_serial.shape[0]):
        # non_serial
        age, aids, hem, mets, admissiontype = non_serial[admid][0], non_serial[admid][1], non_serial[admid][2], non_serial[admid][3], non_serial[admid][4]

        try:
            age = age
            if age < 40:
                non_serial[admid][0] = 0.0
            elif age < 60:
                non_serial[admid][0] = 7.0
            elif age < 70:
                non_serial[admid][0] = 12.0
            elif age < 75:
                non_serial[admid][0] = 15.0
            elif age < 80:
                non_serial[admid][0] = 16.0
            elif age >= 80:
                non_serial[admid][0] = 18.0
        except:
            non_serial[0] = 0.0

        try:
            if aids == 1:
                non_serial[admid][1] = 17.0
            else:
                non_serial[admid][1] = 0.0
        except:
            non_serial[admid][1] = 0.0

        try:
            if hem == 1:
                non_serial[admid][2] = 10.0
            else:
                non_serial[admid][2] = 0.0
        except:
            non_serial[admid][2] = 0.0

        try:
            if mets == 1:
                non_serial[admid][3] = 9.0
            else:
                non_serial[admid][3] = 0.0
        except:
            non_serial[admid][3] = 0.0

        try:
            if admissiontype == 0: # medical
                non_serial[admid][4] = 6.0
            elif admissiontype == 1: # sche
                non_serial[admid][4] = 0.0
            elif admissiontype == 2: # unsche
                non_serial[admid][4] = 8.0
        except:
            non_serial[admid][4] = 0.0

        # serial
        for t in range(serial[admid].shape[0]):
            gcs = serial[admid][t][0]
            sbp = serial[admid][t][1]
            hr = serial[admid][t][2]
            bt = serial[admid][t][3]
            pfr = serial[admid][t][4]
            uo = serial[admid][t][5]
            sunl = serial[admid][t][6]
            wbc = serial[admid][t][7]
            sbl = serial[admid][t][8]
            sl = serial[admid][t][9]
            pl = serial[admid][t][10]
            bl = serial[admid][t][11]

            try:
                if hr < 40:
                    serial[admid][t][2] = 11.0
                elif hr >= 160:
                    serial[admid][t][2] = 7.0
                elif hr >= 120:
                    serial[admid][t][2] = 4.0
                elif hr < 70:
                    serial[admid][t][2] = 2.0
                elif hr >= 70 and hr < 120:
                    serial[admid][t][2] = 0.0
                else:
                    serial[admid][t][2] = 0.0
            except:
                serial[admid][t][2] = 0.0

            try:
                if sbp < 70:
                    serial[admid][t][1] = 13.0
                elif sbp < 100:
                    serial[admid][t][1] = 5.0
                elif sbp >= 200:
                    serial[admid][t][1] = 2.0
                elif sbp >= 100 and sbp < 200:
                    serial[admid][t][1] = 0.0
                else:
                    serial[admid][t][1] = 0.0
            except:
                serial[admid][t][1] = 0.0

            try:
                if bt < 39.0:
                    serial[admid][t][3] = 0.0
                elif bt >= 39.0:
                    serial[admid][t][3] = 3.0
                else:
                    serial[admid][t][3] = 0.0
            except:
                serial[admid][t][3] = 0.0

            try:
                if pfr < 100:
                    serial[admid][t][4] = 11.0
                elif pfr < 200:
                    serial[admid][t][4] = 9.0
                elif pfr >= 200:
                    serial[admid][t][4] = 6.0
                else:
                    serial[admid][t][4] = 0.0
            except:
                serial[admid][t][4] = 0.0

            try:
                if uo < 500:
                    serial[admid][t][5] = 11.0
                elif uo < 1000:
                    serial[admid][t][5] = 4.0
                elif uo >= 1000:
                    serial[admid][t][5] = 0.0
                else:
                    serial[admid][t][5] = 0.0
            except:
                serial[admid][t][5] = 0.0

            try:
                if sunl < 28.0:
                    serial[admid][t][6] = 0.0
                elif sunl < 83.0:
                    serial[admid][t][6] = 6.0
                elif sunl >= 84.0:
                    serial[admid][t][6] = 10.0
                else:
                    serial[admid][t][6] = 0.0
            except:
                serial[admid][t][6] = 0.0

            try:
                if wbc < 1.0:
                    serial[admid][t][7] = 12.0
                elif wbc >= 20.0:
                    serial[admid][t][7] = 3.0
                elif wbc >= 1.0 and wbc < 20.0:
                    serial[admid][t][7] = 0.0
                else:
                    serial[admid][t][7] = 0.0
            except:
                serial[admid][t][7] = 0.0

            try:
                if pl < 3.0:
                    serial[admid][t][10] = 3.0
                elif pl >= 5.0:
                    serial[admid][t][10] = 3.0
                elif pl >= 3.0 and pl < 5.0:
                    serial[admid][t][10] = 0.0
                else:
                    serial[admid][t][10] = 0.0
            except:
                serial[admid][t][10] = 0.0

            try:
                if sl < 125:
                    serial[admid][t][9] = 5.0
                elif sl >= 145:
                    serial[admid][t][9] = 1.0
                elif sl >= 125 and sl < 145:
                    serial[admid][t][9] = 0.0
                else:
                    serial[admid][t][9] = 0.0
            except:
                serial[admid][t][9] = 0.0

            try:
                if sbl < 15.0:
                    serial[admid][t][8] = 5.0
                elif sbl < 20.0:
                    serial[admid][t][8] = 3.0
                elif sbl >= 20.0:
                    serial[admid][t][8] = 0.0
                else:
                    serial[admid][t][8] = 0.0
            except:
                serial[admid][t][8] = 0.0

            try:
                if bl < 4.0:
                    serial[admid][t][11] = 0.0
                elif bl < 6.0:
                    serial[admid][t][11] = 4.0
                elif bl >= 6.0:
                    serial[admid][t][11] = 9.0
                else:
                    serial[admid][t][11] = 0.0
            except:
                serial[admid][t][11] = 0.0

            try:
                if gcs < 3:
                    serial[admid][t][0] = 0.0
                elif gcs < 6:
                    serial[admid][t][0] = 26.0
                elif gcs < 9:
                    serial[admid][t][0] = 13.0
                elif gcs < 11:
                    serial[admid][t][0] = 7.0
                elif gcs < 14:
                    serial[admid][t][0] = 5.0
                elif gcs >= 14 and gcs <= 15:
                    serial[admid][t][0] = 0.0
                else:
                    serial[admid][t][0] = 0.0
            except:
                serial[admid][t][0] = 0.0

    return [serial, non_serial]
