import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV
from time import process_time
from Models.model_validation import results

def m_rf(X_train, y_train, X_test, y_test):
    """
    Метод обучает модель RandomForestClassifier с использованием инструментов подбора гиперпараметров GridSearchCV
    и RandomizedSearchCV
    :param X_train: обучающий массив np значений атрибутов
    :param y_train: обучающий массив target
    :param X_test: тестовый массив np значений атрибутов
    :param y_test: тестовый массив target
    :return:
    """
    start_time = process_time()
    rf = RandomForestClassifier()
    grid_space={'max_depth': [3, 5, 10, None],
                  'n_estimators': [10, 100, 200],
                  'max_features': [1, 3, 5, 7],
                  'min_samples_leaf': [1, 2, 3],
                  'min_samples_split': [1, 2, 3]
               }
    grid = GridSearchCV(rf, param_grid=grid_space, cv=3, scoring='accuracy')
    model_grid = grid.fit(X_train, y_train)
    y_pred_grid = model_grid.predict(X_test)
    print('Best grid search hyperparameters are: '+str(model_grid.best_params_))
    print('Best grid search score is: '+str(model_grid.best_score_))

# random search cv
    rs_space={'max_depth': list(np.arange(10, 100, step=10)) + [None],
              'n_estimators': np.arange(10, 500, step=20),
              'max_features': randint(1, 14),
              'criterion': ['gini', 'entropy', 'log_loss'],
              'min_samples_leaf': randint(1, 10),
              'min_samples_split': np.arange(2, 20, step=2)
          }


    rf_random = RandomizedSearchCV(rf, rs_space, n_iter=500, scoring='accuracy', n_jobs=-1, cv=3)
    model_random = rf_random.fit(X_train, y_train)
    y_pred_random = model_random.predict(X_test)
    print('Best random search hyperparameters are: '+str(model_random.best_params_))
    print('Best random search score is: '+str(model_random.best_score_))

    finish_time = process_time()
    rf_time = finish_time - start_time
    print('Random forest random')
    results(y_test, y_pred_random, rf_time)
    print('Random forest grid')
    results(y_test, y_pred_grid, rf_time)

def m_rf_second(X_train, y_train, X_test, y_test):
    """
    Метод обучает модель RandomForestClassifier без использования инструментов подбора гиперпараметров
    :param X_train: обучающий массив np значений атрибутов
    :param y_train: обучающий массив target
    :param X_test: тестовый массив np значений атрибутов
    :param y_test: тестовый массив target
    :return:
    """
    start_time = process_time()
    rf = RandomForestClassifier()
    model = rf.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    finish_time = process_time()
    rf_time = finish_time - start_time
    print('Random forest')
    results(y_test, y_pred, rf_time)