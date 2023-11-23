import pandas as pd
import numpy as np
from Data.Sources.Macroeconomics_data import dict_macroeconomics_data as dict_md, years_list_macroeconomics as list_years

def get_df_targets(path_file):
    """
    Получаем данные из xls файла. Исключаем дубликаты. Меняем отчетные даты у контрагентом с target 0
    :param path_file: путь к файлу с target
    :return: датафрейм с компаний с target
    """
    df = pd.read_excel(path_file)
    df.drop_duplicates(subset=['ИНН'], inplace=True)
    return df


def get_df_spark(path_file):
    """
    Исключаю дублирование названия столбцов
    :param path_file: путь к файлу, в которой выложены отчеты СПАРК
    :return: датафремй
    """
    df = pd.read_excel(path_file)
    df.drop_duplicates(subset=['Код налогоплательщика'], inplace=True)
    return df


def df_add_attribute_macro(df):
    """
    В датафремй добавляюься макроэкономические данные.
    Обогащаю таблицу данными курсовой валатильности
    Обогащаю таблицу данными динамики ВВП
    Обогащаю таблицу данными об уровне безработицы
    Обогащаю таблицу данными об уровне реальной заработной платы
    Обогащаю таблицу данными по ключевой ставке Банка России
    Данные следует обновить в случае изменения годов выгрузки отчетности контрагентов.
    :param df: входящий датафрейм с таргетом
    :return: входящий датафрейм с макроэкономическими атрибутами
    """
    for name_column, values in dict_md.items():
        _list = [values[i]/values[i-1] for i in range(1, len(values))] if len(values) == 8 else values
        data = {'Отчетный год': list_years[1:], name_column: _list}
        _df = pd.DataFrame(data)
        df = pd.merge(df, _df, on='Отчетный год', how='left')
    return df