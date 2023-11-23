import optuna
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
import numpy as np


def optune_optimize(trial):
    """
    Реализация подбора гиперпараметров с использованием фреймворка optuna. Запускать стоит через файл optuna_gb.py
    :param trial:
    :return:
    """
    max_depth = trial.suggest_int('max_depth', 1, 50, 1)
    subsample = trial.suggest_float('subsample', 0.1, 1.0, step=0.1)
    n_estimators = trial.suggest_int('n_estimators', 50, 1500, 50)
    min_samples_split = trial.suggest_float('min_samples_split', 0.1, 1.0, step=0.1)
    min_samples_leaf = trial.suggest_float('min_samples_leaf', 0.01, 1.00, step=0.01)
    min_weight_fraction_leaf = trial.suggest_float('min_weight_fraction_leaf', 0.1, 0.5, step=0.1)
    min_impurity_decrease = trial.suggest_float('min_impurity_decrease', 0.0, 3.0, step=0.1)
    max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 2, 150, 1)
    learning_rate = trial.suggest_float('learning_rate', 0.0, 1.0, step=0.1)
    loss = trial.suggest_categorical('loss', ['log_loss', 'exponential'])

    model = GradientBoostingClassifier(max_depth=max_depth,
                                       subsample=subsample,
                                       n_estimators=n_estimators,
                                       min_samples_split=min_samples_split,
                                       min_samples_leaf=min_samples_leaf,
                                       min_weight_fraction_leaf=min_weight_fraction_leaf,
                                       min_impurity_decrease=min_impurity_decrease,
                                       max_leaf_nodes=max_leaf_nodes,
                                       learning_rate=learning_rate,
                                       loss=loss,
                                       random_state=42
                                        )
    model.fit(X_train, y_train)
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

study_gb = optuna.create_study(study_name='GB', direction='maximize')
study_gb.optimize(optune_optimize, n_trials=10, n_jobs=-1)
print(f'Параметры модели {study_gb.best_params}')
print(f'Лучший score {study_gb.best_value}')