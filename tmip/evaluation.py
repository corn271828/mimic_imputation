from sklearn import metrics
import numpy as np

def eval_metrics(task_name, pred, gt, verbose=False):
  if task_name == 'mor':
    hard_pred = (pred > 0)
    val_acc = np.mean(gt == hard_pred)
    val_auroc = metrics.roc_auc_score(gt, pred)
    val_ap = metrics.average_precision_score(gt, pred)
    if verbose:
      print('Accuracy: {:.2f}'.format(val_acc*100))
      print('ROCAUC: {:.2f}'.format(val_auroc*100))
      print('AP: {:.2f}'.format(val_ap*100))

    return [val_acc, val_auroc, val_ap]
  elif task_name == 'icd9':
    hard_pred = (pred > 0)
    val_acc = np.mean(gt == hard_pred)
    val_auroc = metrics.roc_auc_score(gt, pred)
    val_ap = metrics.average_precision_score(gt, pred)
    if verbose:
      print('Accuracy: {:.2f}'.format(val_acc*100))
      print('ROCAUC: {:.2f}'.format(val_auroc*100))
      print('AP: {:.2f}'.format(val_ap*100))

    return [val_acc, val_auroc, val_ap]
  else:
    val_mse = metrics.mean_squared_error(gt, pred)
    if verbose:
      print('MSE: {:.2f}'.format(val_mse))

    return [val_mse]

def _evalMSE(labels, preds):
  error = labels - preds
  return (error ** 2).sum(axis=-1).mean()

def _evalMAPE(labels, preds):
  return np.nanmean(np.abs((labels - preds)/labels))
