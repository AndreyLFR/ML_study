from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
import numpy as np
from time import process_time
from Models.model_validation import results

def m_lda(X_train, y_train, X_test, y_test):
    """
    Метод обучает модель LinearDiscriminantAnalysis с использованием инструментов подбора гиперпараметров GridSearchCV
    и RandomizedSearchCV
    :param X_train: обучающий массив np значений атрибутов
    :param y_train: обучающий массив target
    :param X_test: тестовый массив np значений атрибутов
    :param y_test: тестовый массив target
    :return:
    """
    start_time = process_time()
    lda = LDA()
    grid_space = {'solver': ['svd', 'lsqr', 'eigen'],
                  'shrinkage': [0.0, 0.3, 0.6, 0.9, 1.0, 'auto'],
                  'n_components': [1, 2, 3, 4, 5],
                  'tol': [0.0, 0.3, 0.6, 0.9, 1.0]
                  }
    grid = GridSearchCV(lda, param_grid=grid_space, cv=5, scoring='accuracy')
    model_grid = grid.fit(X_train, y_train)
    y_pred_grid = model_grid.predict(X_test)
    print('Best grid search hyperparameters are: ' + str(model_grid.best_params_))
    print('Best grid search score is: ' + str(model_grid.best_score_))

    rs_space={'shrinkage': list(np.arange(0, 1.0, step=0.1)) + ['auto'],
              'n_components': np.arange(1, 50, step=1),
              'tol': uniform(0.0, 1.0),
              'solver': ['svd', 'lsqr', 'eigen'],
              }
    gb_random = RandomizedSearchCV(lda, rs_space, n_iter=1000, scoring='accuracy', n_jobs=-1, cv=5)
    model_random = gb_random.fit(X_train, y_train)
    y_pred_random = model_random.predict(X_test)
    # random search results
    print('Best random search hyperparameters are: '+str(model_random.best_params_))
    print('Best random search score is: '+str(model_random.best_score_))

    finish_time = process_time()
    gb_time = finish_time - start_time
    print('LDA random')
    results(y_test, y_pred_random, gb_time)
    print('LDA grid')
    results(y_test, y_pred_grid, gb_time)


def m_lda_second(X_train, y_train, X_test, y_test):
    """
    Метод обучает модель LinearDiscriminantAnalysis без использования инструментов подбора гиперпараметров
    :param X_train: обучающий массив np значений атрибутов
    :param y_train: обучающий массив target
    :param X_test: тестовый массив np значений атрибутов
    :param y_test: тестовый массив target
    :return:
    """
    start_time = process_time()
    lda = LDA().fit(X_train, y_train)
    y_pred = lda.predict(X_test)
    finish_time = process_time()
    lda_time = finish_time - start_time
    print('LDA')
    results(y_test, y_pred, lda_time)