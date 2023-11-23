from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
from time import process_time
from Models.model_validation import results
import numpy as np


def m_svm(X_train, y_train, X_test, y_test):
    """
    Метод обучает модель SVC с использованием инструментов подбора гиперпараметров GridSearchCV
    и RandomizedSearchCV
    :param X_train: обучающий массив np значений атрибутов
    :param y_train: обучающий массив target
    :param X_test: тестовый массив np значений атрибутов
    :param y_test: тестовый массив target
    :return:
    """
    start_time = process_time()
    svc = SVC()
    grid_space = {'C': np.arange(4, 60, 4),
                  'gamma': np.arange(0.1, 1, 0.2),
                  'kernel': ['linear', 'rbf'],
                  'cache_size': [50, 200, 300]}
    grid = GridSearchCV(svc, param_grid=grid_space, cv=None, n_jobs=-1, scoring='accuracy', verbose=2,
                        refit=True, error_score='raise', pre_dispatch='2*n_jobs')
    model_grid = grid.fit(X_train, y_train)
    y_pred_grid = model_grid.predict(X_test)
    print('Best grid search hyperparameters are: ' + str(model_grid.best_params_))
    print('Best grid search score is: ' + str(model_grid.best_score_))

    rs_space = {'C': uniform(2, 10),
                'gamma': uniform(0.1, 1)}
    gb_random = RandomizedSearchCV(svc, param_distributions=rs_space, n_iter=20, scoring='accuracy', n_jobs=-1, cv=3,
                                   random_state=42, verbose=2)
    model_random = gb_random.fit(X_train, y_train)
    y_pred_random = model_random.predict(X_test)
    print('Best random search hyperparameters are: '+str(model_random.best_params_))
    print('Best random search score is: '+str(model_random.best_score_))

    finish_time = process_time()
    svm_time = finish_time - start_time
    print('SVM random')
    results(y_test, y_pred_random, svm_time)
    print('SVM grid')
    results(y_test, y_pred_grid, svm_time)


def m_svm_second(X_train, y_train, X_test, y_test):
    """
    Метод обучает модель SVM — дискриминативный классификатор без использования инструментов подбора гиперпараметров
    :param X_train: обучающий массив np значений атрибутов
    :param y_train: обучающий массив target
    :param X_test: тестовый массив np значений атрибутов
    :param y_test: тестовый массив target
    :return:
    """
    start_time = process_time()
#    svm = SVC(gamma='auto', probability=True).fit(X_train, y_train)
    svm = SVC().fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    finish_time = process_time()
    svm_time = finish_time - start_time
    print('SVM')
    results(y_test, y_pred, svm_time)