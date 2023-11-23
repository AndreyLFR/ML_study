from sklearn.neighbors import KNeighborsClassifier
from time import process_time
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
import numpy as np
from Models.model_validation import results


def m_knn(X_train, y_train, X_test, y_test):
    """
    Метод обучает модель KNeighborsClassifier с использованием инструментов подбора гиперпараметров GridSearchCV
    и RandomizedSearchCV
    :param X_train: обучающий массив np значений атрибутов
    :param y_train: обучающий массив target
    :param X_test: тестовый массив np значений атрибутов
    :param y_test: тестовый массив target
    :return:
    """
    start_time = process_time()
    knn = KNeighborsClassifier()
    grid_space = {'n_neighbors': np.arange(1, 10),
                  'metric': ['euclidean', 'cityblock'],
                  'weights': ['uniform', 'distance'],
                  'leaf_size': [1, 10, 20, 30, 40, 50]}
    grid = GridSearchCV(knn, param_grid=grid_space, cv=None, n_jobs=-1, scoring='accuracy', verbose=2,
                        refit=True, error_score='raise', pre_dispatch='2*n_jobs')
    model_grid = grid.fit(X_train, y_train)
    y_pred_grid = model_grid.predict(X_test)
    print('Best grid search hyperparameters are: ' + str(model_grid.best_params_))
    print('Best grid search score is: ' + str(model_grid.best_score_))

    rs_space = {'n_neighbors': range(1, 5), 'weights': ['uniform', 'distance']}
    gb_random = RandomizedSearchCV(knn, param_distributions=rs_space, n_iter=20, scoring='accuracy', n_jobs=-1, cv=3,
                                   random_state=42, verbose=2)
    model_random = gb_random.fit(X_train, y_train)
    y_pred_random = model_random.predict(X_test)
    print('Best random search hyperparameters are: '+str(model_random.best_params_))
    print('Best random search score is: '+str(model_random.best_score_))

    finish_time = process_time()
    knn_time = finish_time - start_time
    print('KNN random')
    results(y_test, y_pred_random, knn_time)
    print('KNN grid')
    results(y_test, y_pred_grid, knn_time)


def m_knn_second(X_train, y_train, X_test, y_test):
    """
    Метод обучает модель KNeighborsClassifier без использования инструментов подбора гиперпараметров
    :param X_train: обучающий массив np значений атрибутов
    :param y_train: обучающий массив target
    :param X_test: тестовый массив np значений атрибутов
    :param y_test: тестовый массив target
    :return:
    """
    start_time = process_time()
    #neigh = KNeighborsClassifier(n_neighbors=16)
    knn = KNeighborsClassifier().fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    finish_time = process_time()
    neigh_time = finish_time - start_time
    print('KNN')
    results(y_test, y_pred, neigh_time)