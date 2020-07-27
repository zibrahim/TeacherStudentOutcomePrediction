from collections import Counter, defaultdict
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import f1_score

import keras.backend as K

lstm_dict={}
lstm_dict['activation'] = ['sigmoid']
lstm_dict['batch'] = ['all']
lstm_dict['data'] = ['balanced', 'unchanged']
lstm_dict['dropout'] = [0.0, 0.2, 0.4, 0.6]
lstm_dict['layers'] = [1, 2]
lstm_dict['masking'] = [True]
lstm_dict['optimizer'] = ['RMSprop', 'Adagrad']
lstm_dict['outcome'] = ['ITUAdmission7Days', 'ITUAdmission14Days', 'ITUAdmission30Days',
                         'Mortality7Days','Mortality14Days','Mortality30Days']
lstm_dict['units'] = [1, 4, 8, 16, 32, 64, 128]

lstm_running_params = {}
lstm_running_params['monitor_checkpoint'] = ['val_matthews', 'max']
lstm_running_params['monitor_early_stopping'] = ['val_matthews', 'max']
lstm_running_params['early_stopping'] = True
lstm_running_params['save_checkpoint'] = True
lstm_running_params['seed'] = 42
lstm_running_params['n_splits'] = 10
lstm_running_params['n_epochs'] = 1000
lstm_running_params['patience'] = 200

def stratified_group_k_fold ( X, y, groups, k, seed=None) :
    labels_num = np.max(y) + 1
    y_counts_per_group = defaultdict(lambda : np.zeros(labels_num))
    y_distr = Counter()
    for label, g in zip(y, groups) :
        y_counts_per_group[g][label] += 1
        y_distr[label] += 1

    y_counts_per_fold = defaultdict(lambda : np.zeros(labels_num))
    groups_per_fold = defaultdict(set)

    def eval_y_counts_per_fold ( y_counts, fold ) :
        y_counts_per_fold[fold] += y_counts
        std_per_label = []
        for label in range(labels_num) :
            label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(k)])
            std_per_label.append(label_std)
        y_counts_per_fold[fold] -= y_counts
        return np.mean(std_per_label)

    groups_and_y_counts = list(y_counts_per_group.items())
    random.Random(seed).shuffle(groups_and_y_counts)

    for g, y_counts in sorted(groups_and_y_counts, key=lambda x : -np.std(x[1])) :
        best_fold = None
        min_eval = None
        for i in range(k) :
            fold_eval = eval_y_counts_per_fold(y_counts, i)
            if min_eval is None or fold_eval < min_eval :
                min_eval = fold_eval
                best_fold = i
        y_counts_per_fold[best_fold] += y_counts
        groups_per_fold[best_fold].add(g)

    all_groups = set(groups)
    for i in range(k) :
        train_groups = all_groups - groups_per_fold[i]
        test_groups = groups_per_fold[i]

        train_indices = [i for i, g in enumerate(groups) if g in train_groups]
        test_indices = [i for i, g in enumerate(groups) if g in test_groups]

        yield train_indices, test_indices

def get_distribution ( y_vals ) :
    y_distr = Counter(y_vals)
    y_vals_sum = sum(y_distr.values())
    return [f'{y_distr[i] / y_vals_sum:.2%}' for i in range(np.max(y_vals) + 1)]

def generate_balanced_arrays(X_train, y_train):
 while True:
  positive = np.where(y_train==1)[0].tolist()
  negative = np.random.choice(np.where(y_train==0)[0].tolist(),size = len(positive), replace = False)
  balance = np.concatenate((positive, negative), axis=0)
  np.random.shuffle(balance)
  input = X_train.iloc[balance, :]
  target = y_train.iloc[balance]
  yield input, target

def generate_trajectory_timeseries(df, baseline_columns, static_columns, timeseries_columns, id_col, outcome_columns):
    for i, j in zip(timeseries_columns, baseline_columns):
        df[i] = df[i] - df[j]

    new_df = df[timeseries_columns]
    new_df.insert(0, id_col, df[id_col])
    new_df[outcome_columns] = df[outcome_columns]
    new_df[static_columns] = df[static_columns]

    return new_df

def impute(df, impute_columns):
    imp = IterativeImputer(max_iter=10, random_state=0)
    imp.fit(df[impute_columns])
    df[impute_columns] = imp.transform(df[impute_columns])

    return df[impute_columns]

def scale(df, scale_columns):

    scaler = MinMaxScaler()
    normalized_df = pd.DataFrame(scaler.fit_transform(df[scale_columns]))
    normalized_df.columns = scale_columns

    return normalized_df
