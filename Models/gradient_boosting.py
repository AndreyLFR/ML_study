from time import process_time
from Models.model_validation import results
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from scipy.stats import randint, uniform
from Models.model_validation import save_model
import numpy as np

def m_gb(X_train, y_train, X_test, y_test):
    """
    Метод обучает модель GradientBoostingClassifier с использованием инструментов подбора гиперпараметров GridSearchCV
    и RandomizedSearchCV
    :param X_train: обучающий массив np значений атрибутов
    :param y_train: обучающий массив target
    :param X_test: тестовый массив np значений атрибутов
    :param y_test: тестовый массив target
    :return:
    """
    start_time = process_time()
    gb = GradientBoostingClassifier()
    grid_space = {'max_depth': [7, 8, 9, 10, None],
                  'subsample': [0.1, 0.8, 0.9, 0.95, 1.0],
                  'n_estimators': [700, 850, 900, 950, 1200, 1300],
                  'min_samples_split': [0.01, 0.1, 0.3, 0.5, 1.0],
                  'min_samples_leaf': [0.01, 0.1, 0.3, 0.5, 0.99],
                  'min_weight_fraction_leaf': [0, 0.05, 0.1, 0.2, 0.5],
                  'min_impurity_decrease': [0, 0.05, 0.1, 0.7, 3],
                  'max_leaf_nodes': [2, 6, 17, 51, 150],
                  'learning_rate': [0.05, 0.075, 0.1, 0.15, 0.2],
                  'loss': ['exponential']
                  }
    grid = GridSearchCV(gb, param_grid=grid_space, cv=5, scoring='accuracy')
    model_grid = grid.fit(X_train, y_train)
    y_pred_grid = model_grid.predict(X_test)
    print('Best grid search hyperparameters are: ' + str(model_grid.best_params_))
    print('Best grid search score is: ' + str(model_grid.best_score_))

    rs_space = {'max_depth': list(np.arange(1, 100, step=1)) + [None],
                'subsample': uniform(0.1, 1.0),
                'n_estimators': np.arange(10, 700, step=10),
                'min_samples_split': uniform(0.01, 1.0),
                'min_samples_leaf': uniform(0.01, 0.99),
                'min_weight_fraction_leaf': uniform(0.0, 0.9),
                'min_impurity_decrease': uniform(0.0, 3.0),
                'max_leaf_nodes': randint(1, 300),
                'learning_rate': uniform(0.0, 0.5),
                'loss': ['log_loss', 'exponential'],
                }

    gb_random = RandomizedSearchCV(gb, rs_space, n_iter=300, scoring='accuracy', n_jobs=-1, cv=5)
    model_random = gb_random.fit(X_train, y_train)
    y_pred_random = model_random.predict(X_test)
    print('Best random search hyperparameters are: '+str(model_random.best_params_))
    print('Best random search score is: '+str(model_random.best_score_))

    finish_time = process_time()
    gb_time = finish_time - start_time
    print('Gradient Boosting random')
    results(y_test, y_pred_random, gb_time)
    print('Gradient Boosting grid')
    results(y_test, y_pred_grid, gb_time)
    save_model(name_m='gb_random', model=model_random)
    save_model(name_m='gb_grid', model=model_grid)


def m_gb_second(X_train, y_train, X_test, y_test):
    """
    Метод обучает модель GradientBoostingClassifier без использования инструментов подбора гиперпараметров
    :param X_train: обучающий массив np значений атрибутов
    :param y_train: обучающий массив target
    :param X_test: тестовый массив np значений атрибутов
    :param y_test: тестовый массив target
    :return:
    """
    start_time = process_time()
#    params = {'loss': 'exponential', 'max_depth': 10, 'n_estimators': 900}
    gb = GradientBoostingClassifier().fit(X_train, y_train)
    y_pred = gb.predict(X_test)
    save_model(name_m='gb_best', model=gb)
    finish_time = process_time()
    gb_time = finish_time - start_time
    print('Gradient boosting')
    results(y_test, y_pred, gb_time)