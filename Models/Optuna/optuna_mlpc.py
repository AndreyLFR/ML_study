import optuna
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import numpy as np


def optune_optimize(trial):
    """
    Реализация подбора гиперпараметров с использованием фреймворка optuna. Запускать стоит через файл optuna_mlpc.py
    :param trial:
    :return:
    """
    solver = trial.suggest_categorical('solver', ['sgd', 'adam', 'lbfgs'])
    max_iter = trial.suggest_int('max_iter', 100, 2000, 100)
    hidden_layer_sizes = trial.suggest_int('hidden_layer_sizes', 1, 15, 1)
    random_state = trial.suggest_int('random_state', 0, 9, 1)
    alpha = trial.suggest_float('alpha', 0.0001, 1, step=0.0001)
    model = MLPClassifier(solver=solver,
                          max_iter=max_iter,
                          hidden_layer_sizes=hidden_layer_sizes,
                          random_state=random_state,
                          alpha=alpha).fit(X_train, y_train)
    score = metrics.roc_auc_score(y_test, model.predict(X_test))
    return score

with (open('X_train', 'rb') as f_xtrain,
      open('y_train', 'rb') as f_ytrain,
      open('X_test', 'rb') as f_xtest,
      open('y_test', 'rb') as f_ytest):
        X_train = np.load(f_xtrain)
        y_train = np.load(f_ytrain)
        X_test = np.load(f_xtest)
        y_test = np.load(f_ytest)

study_gb = optuna.create_study(study_name='nn', direction='maximize')
study_gb.optimize(optune_optimize, n_trials=10, n_jobs=-1)
print(f'Параметры модели {study_gb.best_params}')
print(f'Лучший score {study_gb.best_value}')