from sklearn.neural_network import MLPClassifier
from time import process_time
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
import numpy as np
from Models.model_validation import results

def m_mlpc(X_train, y_train, X_test, y_test):
    """
    Метод обучает модель MLPClassifier с использованием инструментов подбора гиперпараметров GridSearchCV
    и RandomizedSearchCV
    :param X_train: обучающий массив np значений атрибутов
    :param y_train: обучающий массив target
    :param X_test: тестовый массив np значений атрибутов
    :param y_test: тестовый массив target
    :return:
    """
    start_time = process_time()
    nn = MLPClassifier()
    grid_space = {'solver': ['sgd', 'adam', 'lbfgs'],
                  'max_iter': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000 ],
                  'alpha': 10.0 ** -np.arange(1, 10),
                  'hidden_layer_sizes': np.arange(10, 15),
                  'random_state': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                  }
    grid = GridSearchCV(nn, param_grid=grid_space, cv=5, scoring='accuracy')
    model_grid = grid.fit(X_train, y_train)
    y_pred_grid = model_grid.predict(X_test)
    print('Best grid search hyperparameters are: ' + str(model_grid.best_params_))
    print('Best grid search score is: ' + str(model_grid.best_score_))

    rs_space = {
        'hidden_layer_sizes': [(randint.rvs(100, 600, 1), randint.rvs(100, 600, 1),),
                               (randint.rvs(100, 600, 1),)],
        'activation': ['tanh', 'relu', 'logistic'],
        'solver': ['sgd', 'adam', 'lbfgs'],
        'alpha': uniform(0.0001, 0.9),
        'learning_rate': ['constant', 'adaptive']}

    gb_random = RandomizedSearchCV(nn, rs_space, n_iter=1000, scoring='accuracy', n_jobs=-1, cv=5)
    model_random = gb_random.fit(X_train, y_train)
    y_pred_random = model_random.predict(X_test)
    # random search results
    print('Best random search hyperparameters are: '+str(model_random.best_params_))
    print('Best random search score is: '+str(model_random.best_score_))

    finish_time = process_time()
    nn_time = finish_time - start_time
    print('MLPC random')
    results(y_test, y_pred_random, nn_time)
    print('MLPC grid')
    results(y_test, y_pred_grid, nn_time)


def m_mlpc_second(X_train, y_train, X_test, y_test):
    """
    Метод обучает модель MLPClassifier без использования инструментов подбора гиперпараметров
    :param X_train: обучающий массив np значений атрибутов
    :param y_train: обучающий массив target
    :param X_test: тестовый массив np значений атрибутов
    :param y_test: тестовый массив target
    :return:
    """
    start_time = process_time()
    params = {'solver': 'lbfgs', 'max_iter': 1800, 'hidden_layer_sizes': 4, 'random_state': 1, 'alpha': 0.4699}
    nn = MLPClassifier(**params)
    #activation='logistic', solver='adam', hidden_layer_sizes=(100, 100)
    nn.fit(X_train, y_train)
    y_pred = nn.predict(X_test)
    finish_time = process_time()
    nn_time = finish_time - start_time
    print('MLPC')
    results(y_test, y_pred, nn_time)