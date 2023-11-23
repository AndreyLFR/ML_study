import pandas as pd
import numpy as np
import math
from constants import LOGIC_DEL_COL, DICT_TYPES_COL, COL_NAN_TO_NULL, COL_ONLY_POSITIVE
from print_func import print_del_col, print_nan


def logic_del_col(df):
    """
    Удаляю столбцы из логики. Перечень наименований столбцов хранится в константах
    :param df:
    :return: df с удаленными столбцами
    """
    print_del_col(LOGIC_DEL_COL)
    return df.drop(columns=LOGIC_DEL_COL, axis=1)


def logic_format_NA(df):
    """
    Заменяю пустые значения на 0 исходя из логики значений
    :param df:
    :return: df, где nan заменен на 0
    """
    print_nan(df)
    df[COL_NAN_TO_NULL] = df[COL_NAN_TO_NULL].fillna(0)
    return df


def logic_format_types(df):
    """
    Типы данных уточняю в соответствии с содержимым
    :param df:
    :return:
    """
    for i in range(0, len(DICT_TYPES_COL)-1):
        df[DICT_TYPES_COL[i]] = pd.to_numeric(df[DICT_TYPES_COL[i]], errors='coerce')
    print(f'-----\n'
          f'Уточняю типы данных - float меняю на int в следующих столбцах \n'
          f'{DICT_TYPES_COL}')
    return df.astype(DICT_TYPES_COL[-1])


def logic_reformat_SITE(df):
    """
    Параметр Сайт в сети интернет меняю на есть сайт или нет сайта в системе СПАРК
    :param df:
    :return: df в котором атрибут сайт имеет только 2 значения yes_site, np_site
    """
    df['Сайт в сети Интернет'] = df['Сайт в сети Интернет'].notnull().astype('int')
    df['Сайт в сети Интернет'] = df['Сайт в сети Интернет'].apply(lambda x: 'yes_site' if x == 1 else 'no_site')
    print('------\n'
          'Меняю данные в столбце Сайт в сети Интернет на есть сайт и нет сайта')
    return df


def logic_reformat_INDUSTRY(df):
    """
    Открытый перечень отраслей меняем на Розничная торговля, Оптовая торговля, СХ, производство, иная
    :param df:
    :return: df со сгруппированными отраслями
    """
    ref_col = 'Вид деятельности/отрасль'
    for i in df.index:
        if 'озничная' in df.at[i, ref_col] and 'орговля' in df.at[i, ref_col]:
            df.loc[i, ref_col] = 'Розничная торговля'
        elif 'птовая' in df.at[i, ref_col] and 'орговля' in df.at[i, ref_col]:
            df.loc[i, ref_col] = 'Оптовая торговля'
        elif 'астениеводство' in df.at[i, ref_col] or 'ивотноводство' in df.at[i, ref_col] or 'ыращивание' in df.at[i, ref_col]:
            df.loc[i, ref_col] = 'Сельское хозяйство'
        elif 'роизводство' in df.at[i, ref_col]:
            df.loc[i, ref_col] = 'Производство'
        else:
            df.loc[i, ref_col] = 'Иная'
    return df


def logic_del_minus(df):
    """
    Исходя из логики исправляю ошибки входяших данных -
    удаляю строки в которых отрицательные значения не могут быть объяснены логикой.
    :param df:
    :return:
    """
    start = df.shape[0]
    for col in COL_ONLY_POSITIVE:
        df = df.loc[df[col].isna() | (df[col] >= 0)]
        finish = df.shape[0]
    print(f'-----\n Удалено {start - finish} из {start - 1} строк, которые имели отрицательное значения в столбцах: \n'
          f' {COL_ONLY_POSITIVE}')
    return df