import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import ADASYN
from collections import Counter


def auto_cleaning_nan(df):
    '''
    Принимает df и анализирует количество nan в разрезе столбцов. Там где много nan удаляет столбец.
    :param df:
    :return: df c исключенными столбцами, в которых много nan
    '''
    df_notha = df.notna().sum()
    na_col = np.unique(df_notha, return_counts=True)
    plt.figure(figsize=(20, 8))
    diagram_NaN = sns.barplot(x=na_col[0], y=na_col[1])
    diagram_NaN.tick_params(labelsize=5)
    plt.show()
    start = df.shape[1]
    print('-----')
    for i in df.iloc[:, 3:].columns:
        if df[i].notna().sum() < 0.75 * len(df):
            print(f'Удален столбец {i}, так как NaN более 25% от всех значений')
            del df[i]
    finish = df.shape[1]
    print(f'По результатам автоматического исключения столбцов'
          f' с большим количеством nan количество столбцов сократилось с {start} до {finish}')
    return df

def auto_fill_gap(df):
    '''
    Заполнение пропусков методом К-ближайших соседей
    :param input_df:
    :return: df без пропусков
    '''
    knn = KNNImputer(n_neighbors=8)
    knn.fit(df.iloc[:, 4:].values)
    data_df = knn.transform(df.iloc[:, 4:].values)
    output_df = pd.DataFrame(data=data_df, columns=list(df)[4:])
    output_df['Дефолт'] = df['Дефолт']
    output_df['Прогноз_ДКР_ДККР'] = df['Прогноз_ДКР_ДККР']
    output_df['ИНН'] = df['ИНН']
    output_df['Наименование'] = df['Наименование']
    return output_df

def auto_reformat_inf(df):
    '''
    В результате деления на 0 при определении расчетных коэффициентов они принимают значение inf.
    Функция заменяет значение inf на -999, 999
    :param df:
    :return:
    '''
    df.replace([-np.inf, np.inf], [-9999, 9999], inplace=True)
    return df

def multicollenism_del(df):
    """
    Метод удаляет атрибуты, которые имеют мультиколлиниарность, оставляя атрибут с большим влиянием на target
    :param df:
    :return: df c исключенными атрибутами
    """
    cor = df.iloc[:, :-2].corr()
    features = list(cor.columns)
    list_columns_del = []
    for i in cor.columns:
        for j in cor.index:
            if features.index(j) < features.index(i) and abs(cor.loc[i, j]) > 0.8 and i != 'Дефолт' and j != 'Дефолт':
                list_columns_del.append(j) if abs(cor.loc['Дефолт', i]) > abs(cor.loc['Дефолт', j]) else list_columns_del.append(i)
    # удаляю атрибуты с мультиколлениарностью и с наименьшим влиянием на target
    for k in set(list_columns_del):
        print(f'столбец {k} удален, так как выявлена мультиколлениарность')
        del df[k]
    return df

def normalize_df(df):
    """
    Нормализация атрибутов в диапазоне от 0 до 1
    :param df:
    :return: df с нормализацией атрибутов
    """
    df_x = df.iloc[:, :-3]
    scaler = MinMaxScaler()
    scaler.fit(df_x)
    df_norm = pd.DataFrame(data=scaler.transform(df_x), columns=df_x.columns)
    df_norm['Дефолт'] = df['Дефолт']
    df_norm['ИНН'] = df['ИНН']
    df_norm['Наименование'] = df['Наименование']
    return df_norm

def search_log_reg(df):
    """
    Метод исключает атрибуты не имеющие логистическую связь с target
    :param df:
    :return: массив numpy c атрибутами, имеющими логистическую корреляцию с target
    """
    input_df = df.iloc[:, :-2]
    X = input_df.iloc[:, :-2].values
    print(X.shape)
    y = input_df.iloc[:, -1].values
    log_r = LogisticRegression(penalty='l2', max_iter=3000).fit(X, y)
    model = SelectFromModel(log_r, prefit=True, max_features=100)
    X_new = model.transform(X)
    print(X_new.shape)
    return X_new

def rebalancing(X_train, y_train):
    """
    Ребалансировка алгоритмами undersample, ADASYN
    :param X_train:
    :param y_train:
    :return: X, y
    """
    print(f'До ребалансировки {Counter(y_train)}')
    undersample = RandomUnderSampler(sampling_strategy=0.85)
    X_, y_ = undersample.fit_resample(X_train, y_train)
    print(f'После ребалансировки undersample {Counter(y_)}')
    ada = ADASYN(sampling_strategy='minority')
    X_res, y_res = ada.fit_resample(X_, y_)
    print(f'После ребалансировки ADASYN {Counter(y_res)}')
    return X_res, y_res

def rebalancing_second(X_train, y_train):
    """
    Осуществляет балансировку классов 0 и 1. Класс 1 увеличиваем в 2 раза путем копирования для улучшения обучения
    :param X_train:
    :param y_train:
    :return: отбалансированные массивы targeta и атрибутов
    """
    # ребалансировка классов
    train = pd.DataFrame(data=X_train)
    train['Дефолт'] = y_train
    train_1 = train[train['Дефолт'] == 1]
    train_0 = train[train['Дефолт'] == 0]
    print(f'до ребалансировки количество объектов 0 класса {train_0.shape[0]}, 1 класса {train_1.shape[0]}')
    train_0 = train_0.sample(train_1.shape[0] * 2, random_state=0)
    train_1 = pd.concat([train_1, train_1.copy()])
    train_bal = pd.concat([train_1, train_0])
    print(f'после ребалансировки количество объектов 0 класса {train_0.shape[0]}, 1 класса {train_1.shape[0]}')
    X_train_balancing = train_bal.iloc[:, :-1].values
    y_train_balancing = train_bal['Дефолт']
    return X_train_balancing, y_train_balancing