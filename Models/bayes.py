from sklearn.naive_bayes import GaussianNB
from time import process_time
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
from Models.model_validation import results
import numpy as np


def m_bayes(X_train, y_train, X_test, y_test):
    """
    Метод обучает модель Наивный Байесесовский классификатор с использованием инструментов подбора гиперпараметров GridSearchCV
    и RandomizedSearchCV
    :param X_train: обучающий массив np значений атрибутов
    :param y_train: обучающий массив target
    :param X_test: тестовый массив np значений атрибутов
    :param y_test: тестовый массив target
    :return:
    """
    start_time = process_time()
    bayes = GaussianNB()
    grid_space = {'var_smoothing': np.logspace(0, -9, num=100)}
    grid = GridSearchCV(estimator=bayes, param_grid=grid_space, cv=5, verbose=1, scoring='accuracy')
    model_grid = grid.fit(X_train, y_train)
    y_pred_grid = model_grid.predict(X_test)
    print('Best grid search hyperparameters are: ' + str(model_grid.best_params_))
    print('Best grid search score is: ' + str(model_grid.best_score_))

    rs_space={'var_smoothing': np.logspace(0, -9, num=100)}
    gb_random = RandomizedSearchCV(bayes, rs_space, n_iter=1000, scoring='accuracy', n_jobs=-1, cv=5)
    model_random = gb_random.fit(X_train, y_train)
    y_pred_random = model_random.predict(X_test)
    print('Best random search hyperparameters are: '+str(model_random.best_params_))
    print('Best random search score is: '+str(model_random.best_score_))

    finish_time = process_time()
    bayes_time = finish_time - start_time
    print('Bayes random')
    results(y_test, y_pred_random, bayes_time)
    print('Bayes grid')
    results(y_test, y_pred_grid, bayes_time)


def m_bayes_second(X_train, y_train, X_test, y_test):
    """
    Метод обучает модель Наивный Байесесовский классификатор без использования инструментов подбора гиперпараметров
    :param X_train: обучающий массив np значений атрибутов
    :param y_train: обучающий массив target
    :param X_test: тестовый массив np значений атрибутов
    :param y_test: тестовый массив target
    :return:
    """
    start_time = process_time()
    bayes = GaussianNB().fit(X_train, y_train)
    y_pred = bayes.predict(X_test)
    finish_time = process_time()
    bayes_time = finish_time - start_time
    print('Bayes')
    results(y_test, y_pred, bayes_time)