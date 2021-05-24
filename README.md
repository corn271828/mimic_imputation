# How to use this Code

### Data Imputation 


1. Running baseline imputation method.

```
python simple_impute.py
```

There are the following arguments for the ``simple_impute.py``, which uses the sklearn libary.

```
usage: simple_impute.py [-h] [--feature_type {24,48}]
                        [--model_name {knn,randomforest,bayesian}]
                        [--miss_rate MISS_RATE] [--seeds SEEDS [SEEDS ...]]

optional arguments:
  -h, --help            show this help message and exit
  --feature_type {24,48}
  --model_name {knn,randomforest,bayesian}
  --miss_rate MISS_RATE
  --seeds SEEDS [SEEDS ...]
```

- ``--feature_type`` represents the format of feature we would like a model to use, wehther it is 24 hours or 48 hours of the ICU data.
- ``--model_name {knn,randomforest,bayesian}`` represents the imputation model we want to use. ``bayesian`` corresponds to the MICE (implemented in sklearn) with the Bayesian Ridge Estimator. Other estimators are also included but they are not efficient enough to run on the entire MIMIC-III data.
- ``--model_seed MODEL_SEED`` represents random seed.
- ``--miss_rate MISS_RATE`` represents the missing rate used in the MIMIC-III dataset. 


2. Running nueral imputation method.

```
python impute.py 
```

There are the following arguments for the ``impute.py``, which uses pytorch library.

```
usage: impute.py [-h] [--feature_type {24,48}] [--model_name {adv,seq}]
                 [--model_seed MODEL_SEED] [--seeds SEEDS [SEEDS ...]]
                 [--miss_rate MISS_RATE]

optional arguments:
  -h, --help            show this help message and exit
  --feature_type {24,48}
  --model_name {adv,seq}
  --model_seed MODEL_SEED
  --seeds SEEDS [SEEDS ...]
  --miss_rate MISS_RATE
```

- ``--feature_type`` represents the format of feature we would like a model to use, wehther it is 24 hours or 48 hours of the ICU data.
- ``--model_name {adv,seq}`` represents the imputation model we want to use. ``adv`` and ``seq`` corresponds to the adversarial imputer, and the neural imputer referred in the paper, respectively.
- ``--model_seed MODEL_SEED`` represents random seed.
- ``--miss_rate MISS_RATE`` represents the missing rate used in the MIMIC-III dataset. 

### Data Classification

Running classification algorithm.

```
python classify.py
```

There are the following arguments for the ``classify.py``, which uses the sklearn libary.


```
usage: classify.py [-h] [--feature_type {24,48}] --task_name {mor,los,icd9}
                   --model_name {mmdn,mlp}
                   [--miss_rate MISS_RATE] [--seed SEED]
                   [--model_seed MODEL_SEED]

optional arguments:
  -h, --help            show this help message and exit
  --feature_type {24,48}
  --task_name {mor,los,icd9}
  --model_name {mmdn,mlp}
  --miss_rate MISS_RATE
  --seed SEED
  --model_seed MODEL_SEED
```

- ``--feature_type`` represents the format of feature we would like a model to use, wehther it is 24 hours or 48 hours of the ICU data.
- ``--task_name {mor,los,icd9}`` represents the mortality prediction, length of stay regression, ICD9 group prediction.
- ``--model_name {mmdn,mlp}`` represents the imputation model we want to use. ``mmdn`` and ``mlp`` corresponds to the MLP+BiGRU and MLP classifier referred in the paper, respectively.
- ``--miss_rate MISS_RATE`` represents the missing rate used in the MIMIC-III dataset. 
- ``--seed SEED`` corresponds to the random seed used for generating the data, needs to be consistent with the seed in ``impute.py`` and ``simple_impute.py``
- ``--model_seed MODEL_SEED`` represents random seed in model state.


