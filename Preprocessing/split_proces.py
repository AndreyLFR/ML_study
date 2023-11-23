from sklearn.model_selection import train_test_split
from Preprocessing.auto_proces import rebalancing, multicollenism_del, normalize_df, search_log_reg


def split_df(df, flag):
    """
    dataframe делится на обучающую и тестовую выборку.
    в дальнейшем проводится ребалансировка методов rebalancing()
    :param df:
    :param flag: зависит от маршрута - осуществляется нормализация и исключение мультиколлиниарности или нет
    :return:
    """
    if flag % 2 == 0:
        df_multicol = multicollenism_del(df=df)
        df = normalize_df(df=df_multicol)
        X = search_log_reg(df=df)
        y = df['Дефолт'].values
    else:
        X = df.iloc[:, :-3].values
        y = df['Дефолт'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=11)
    print(f'Размер обучающей выборки {len(X_train)}, размер тестовой выборки {len(X_test)}')
    print(f'Количество объектов 1 класса в обучающей выборке {y_train.sum()}, '
          f'количество объектов 1 класса в тестовой {y_test.sum()}')
    X_train_balancing, y_train_balancing = rebalancing(X_train=X_train, y_train=y_train)
    return X_train_balancing, X_test, y_train_balancing, y_test