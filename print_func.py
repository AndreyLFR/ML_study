from constants import COL_NAN_TO_NULL

def print_info_df(df):
    print(f'DF содержит {df.shape[0]-1} строк и {df.shape[1]-1} столбцов.\n'
          f'DF содержит следующие столбцы\n {list(df.columns)}')

def print_del_col(LOGIC_DEL_COL):
    print(f'-----\n'
          f'Исходя из экономического смысла следующие переменные не оказывают влияние на target.'
          f'Удаляем столбцы {LOGIC_DEL_COL}')

def print_nan(df):
    print(f'-----\n'
          f'Незаполненные значения имеют следующие переменные\n'
          f'{df.isnull().sum()}.\n'
          f'В столбцах \n{COL_NAN_TO_NULL}\n значения Nan меняю на 0 из экономического смысла')
