from sklearn.linear_model import LogisticRegression
from time import process_time
from Models.model_validation import results
from sklearn.model_selection import GridSearchCV
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV
import numpy as np


def m_lr(X_train, y_train, X_test, y_test):
    """
    Метод обучает модель LogisticRegression с использованием инструментов подбора гиперпараметров GridSearchCV
    и RandomizedSearchCV
    :param X_train: обучающий массив np значений атрибутов
    :param y_train: обучающий массив target
    :param X_test: тестовый массив np значений атрибутов
    :param y_test: тестовый массив target
    :return:
    """
    start_time = process_time()
    clf = LogisticRegression()
    parameters = {'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                  'penalty': ['l2', 'l1'],
                  'C': [0.001, 0.01, 0.1, 1, 10, 100],
                  'max_iter': [3000, 5000, 8000, 10000, 12000],
                  'random_state': [30, 60, 90, 120, 150]
                  }

    grid_search = GridSearchCV(estimator=clf,
                               param_grid=parameters,
                               scoring='accuracy',
                               cv=10,
                               verbose=0)

    model_grid = grid_search.fit(X_train, y_train)
    y_pred_grid = model_grid.predict(X_test)
    print('Best grid search hyperparameters are: '+str(model_grid.best_params_))
    print('Best grid search score is: '+str(model_grid.best_score_))

    rs_space = {'penalty': ['l1', 'l2', 'elasticnet', None],
                'dual': [False],
                'solver': ['lbfgs', 'liblinear', 'newton - cg', 'newton - cholesky', 'sag', 'saga'],
                'C': randint(1, 5),
                'random_state': list(np.arange(0, 1000, step=100)) + [None],
                'max_iter': np.arange(1000, 15000, step=1000),
                'multi_class': ['auto', 'ovr', 'multinomial']
                }
    clf_random = RandomizedSearchCV(clf, rs_space, n_iter=500, scoring='accuracy', n_jobs=-1, cv=3)
    model_random = clf_random.fit(X_train, y_train)
    y_pred_random = model_random.predict(X_test)
    # random  search results
    print('Best random search hyperparameters are: ' + str(model_random.best_params_))
    print('Best random search score is: ' + str(model_random.best_score_))

    finish_time = process_time()
    clf_time = finish_time - start_time
    print('Logistic Regression random')
    results(y_test, y_pred_random, clf_time)
    print('Logistic Regression grid')
    results(y_test, y_pred_grid, clf_time)


def m_lr_second(X_train, y_train, X_test, y_test):
    """
    Метод обучает модель LogisticRegression без использования инструментов подбора гиперпараметров
    :param X_train: обучающий массив np значений атрибутов
    :param y_train: обучающий массив target
    :param X_test: тестовый массив np значений атрибутов
    :param y_test: тестовый массив target
    :return:
    """
    start_time = process_time()
    params = {'C': 2, 'dual': False, 'max_iter': 13000, 'multi_class': 'multinomial', 'penalty': None, 'random_state': 300, 'solver': 'lbfgs'}
    clf = LogisticRegression(**params)
    model = clf.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    finish_time = process_time()
    clf_time = finish_time - start_time
    print('Logistic Regression')
    results(y_test, y_pred, clf_time)