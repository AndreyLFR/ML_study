import pandas as pd
from constants import KOEF_AMORTIZATION

def add_attributes(df):
    """
    Добавляются переменные EBITDA, СОК, Аналитический СК, Аналитический СК АППГ,
    Динамика выручки, Динамика выручки АППГ, Долг к выручке, Динамика долга и т.д.
    :param df:
    :return: df дополнен расчетными коэффициентами
    """
    # добавляю расчетные показатели
    df['EBITDA'] = df['EBIT'] + KOEF_AMORTIZATION * df['ОС']
    df['СОК'] = df['Капитал'] + df['ДКиЗ'] - df['ВНА'] - df['КФЛ']
    df['Аналитический СК'] = df['Капитал'] - df['ДФЛ'] - df['КФЛ']
    df['Аналитический СК АППГ'] = df['Капитал АППГ'] - df['ДФЛ АППГ'] - df['КФЛ АППГ']
    df['Динамика выручки'] = df['Выручка'] / df['Выручка, АППГ']
    df['Долг к выручке'] = df['Совокупный долг'] / df['Выручка']
    df['Долг к выручке к АППГ'] = df['Совокупный долг, АППГ'] / df['Выручка, АППГ']
    df['Динамика долга'] = df['Долг к выручке'] / df['Долг к выручке к АППГ']
    df['Долг к EBITDA'] = df['Совокупный долг'] / df['EBITDA']
    df['%% к EBIT'] = df['Проценты к уплате'] / df['EBIT']
    df['EBITDA к выручке'] = df['EBITDA'] / df['Выручка']
    df['Пол или отр СОК'] = [1 if element >= 0 else 0 for element in df['СОК']]
    df['Капитал к долгу'] = df['Капитал'] / df['Совокупный долг']
    df['Аналит капитал к долгу'] = df['Аналитический СК'] / df['Совокупный долг']
    df['Капитал к ВБ'] = df['Капитал'] / df['Активы']
    df['Аналит капитал к ВБ'] = df['Аналитический СК'] / df['Активы']
    df['Динамика обор запасов АППГ'] = df['Обор запасы'] / df['Обор запасы АППГ']
    df['Динамика обор ДЗ АППГ'] = df['Обор ДЗ'] / df['Обор ДЗ АППГ']
    df['Динамика обор КЗ АППГ'] = df['Обор КЗ'] / df['Обор КЗ АППГ']
    df['Иски к капиталу'] = (df['Сумма незавершенных исков в роли ответчика, тыс. RUB'] * 1000) / df['Капитал']
    df['ОС к ВБ'] = df['ОС'] / df['Активы']
    df['Динамика капитала'] = df['Капитал'] / df['Капитал АППГ']
    df['Динамика аналит капитала'] = df['Аналитический СК'] / df['Аналитический СК АППГ']
    df['Динамика ВВП в долл'] = df['Динамика ВВП'] / df['Динамика курса долл к руб']
    return df

def add_reformat_owner(df):
    """
    Преобразую список собственников в 2 переменные: категория собственника и количество собственников
    :param df:
    :return:
    """
    legal_form = [', АО', ', ООО', ', ЗАО', ', ПАО']
    fio_ending = ['вна', 'вич', 'Оглы']
    number_owners_list = []
    category_owners_list = []
    for owner in df['Совладельцы, Приоритетный'].tolist():
        if isinstance(owner, str):
            number_owners_list.append(str(len(owner.split('('))-1))
            if (legal_form[0] in owner) or (legal_form[1] in owner) or (legal_form[2] in owner) or (legal_form[3] in owner):
                category_owners_list.append('собственник ЮЛ')
            elif (fio_ending[0] not in owner) and (fio_ending[1] not in owner) and (fio_ending[2] not in owner):
                category_owners_list.append('иностранец, субъект или другой')
            else:
                category_owners_list.append('только ФЛ')
        else:
            category_owners_list.append(owner)
            number_owners_list.append(owner)
    df['Собственник'] = category_owners_list
    df['Количество собственников'] = number_owners_list
    df.drop(columns=['Совладельцы, Приоритетный'], axis=1, inplace=True)
    return df