import numpy as np
import pandas as pd
from time import process_time
from Data.func_data import get_df_targets, get_df_spark, df_add_attribute_macro
from Data.func_calc_attributes import add_attributes, add_reformat_owner
from Preprocessing.logic_proces import logic_del_col, logic_format_types, logic_format_NA, logic_reformat_SITE,\
    logic_del_minus, logic_reformat_INDUSTRY
from Preprocessing.auto_proces import auto_cleaning_nan, auto_fill_gap, auto_reformat_inf
from Preprocessing.split_proces import split_df
from print_func import print_info_df
from constants import LIST_INDUSTRY_DEL
from Models.random_forest import m_rf, m_rf_second
from Models.logistic_regression import m_lr, m_lr_second
from Models.mlpc import m_mlpc, m_mlpc_second
from Models.gradient_boosting import m_gb, m_gb_second
from Models.lda import m_lda, m_lda_second
from Models.bayes import m_bayes, m_bayes_second
from Models.svm import m_svm, m_svm_second
from Models.knn import m_knn, m_knn_second
from Models.model_validation import load_model, results

from Input_func.input_func import input_work_algorithm


track_number = input_work_algorithm()

if track_number == 1 or track_number == 101:
    if track_number == 1:
        path_file_target = 'Data/Sources/Targets.xlsx'
        path_file_spark = 'Data/Sources/ML_study_SPARK.xlsx'
        path_result = 'Data/Sources/united_data.xlsx'
        path_pp_logic = 'Data/Sources/pp_logic_data.xlsx'
        path_pp_auto = 'Data/Sources/pp_auto_data.xlsx'
    else:
        path_file_target = 'Data/Sources/Target_test.xlsx'
        path_file_spark = 'Data/Sources/Test_spark.xlsx'
        path_result = 'Data/Sources/united_data_test.xlsx'
        path_pp_logic = 'Data/Sources/pp_logic_data_test.xlsx'
        path_pp_auto = 'Data/Sources/pp_auto_data_test.xlsx'
    #объединяю все входящие данные в один df
    df = get_df_targets(path_file=path_file_target)
    df_spk = get_df_spark(path_file=path_file_spark)
    df = df.merge(df_spk, how='left', left_on='ИНН', right_on='Код налогоплательщика')
    #df = df[df['Отчетный год'] != 2022]
    df = df_add_attribute_macro(df=df)
    df.to_excel(path_result, index=False)
    print_info_df(df)

    # предпроцессинг логикой, экспертной оценкой
    df = pd.read_excel(path_result)
    df = logic_del_col(df=df)
    df = df.loc[~df['Вид деятельности/отрасль'].isin(LIST_INDUSTRY_DEL)]
    df = logic_format_NA(df=df)
    df = logic_format_types(df=df)
    df = logic_del_minus(df=df)
    df = logic_reformat_SITE(df=df)
    df = logic_reformat_INDUSTRY(df=df)
    df.to_excel(path_pp_logic, index=False)

    #добавляю расчетные показатели
    df = pd.read_excel(path_pp_logic)
    df = add_attributes(df=df)
    df = add_reformat_owner(df=df)
    df.to_excel('Data/Sources/enrich_data.xlsx', index=False)

    df = pd.read_excel('Data/Sources/enrich_data.xlsx')
    df = auto_cleaning_nan(df) if track_number == 1 else df
    df = pd.get_dummies(data=df, drop_first=True,
                        columns=['Сайт в сети Интернет',
                                 'Вид деятельности/отрасль',
                                 'Регион регистрации',
                                 'Собственник',
                                 'Количество собственников'],
                        )
    df = auto_reformat_inf(df=df)
    df = auto_fill_gap(df)
    df.to_excel(path_pp_auto, index=False)

    if track_number == 101:
        df_train = pd.read_excel('Data/Sources/pp_auto_data.xlsx')
        df_test = pd.DataFrame(columns=list(df_train))

        for col_name in list(df_train):
            if col_name in df.columns:
                df_test[col_name] = df[col_name]
            else:
                df_test[col_name] = df_test[col_name].fillna(0)

        X = df_test.iloc[:, :-3].values
        y = df_test['Дефолт'].values
        y_dkr = df['Прогноз_ДКР_ДККР']
        start_time = process_time()
        model = load_model('gb_best')
        y_pred = model.predict(X)

        dict_ = {}
        names_cols = df['Наименование'].values
        if len(y_pred) == len(names_cols):
            for i in range(len(y_pred)):
                key = names_cols[i]
                value = y_pred[i]
                dict_[key] = value
        finish_time = process_time()
        test_time = finish_time - start_time
        print('TEST_TEST_TEST')
        results(y, y_pred, test_time)
        print('Result DKKR DKR')
        results(y, y_dkr, test_time)
        print(dict_)

elif track_number in [10, 11]:
    df = pd.read_excel('Data/Sources/pp_auto_data.xlsx')
    X_train, X_test, y_train, y_test = split_df(df=df, flag=track_number)
    with (open('Models/Optuna/X_train', 'wb') as f_X_train,
          open('Models/Optuna/y_train', 'wb') as f_y_train,
          open('Models/Optuna/X_test', 'wb') as f_X_test,
          open('Models/Optuna/y_test', 'wb') as f_y_test):
        np.save(f_X_train, X_train)
        np.save(f_y_train, y_train)
        np.save(f_X_test, X_test)
        np.save(f_y_test, y_test)
        print('Массивы сохранены в файлы, теперь можно запускать optuna перейдя в папку OPTUNA')

elif 20 < track_number < 100:
    path_pp_auto = 'Data/Sources/pp_auto_data.xlsx'
    df = pd.read_excel(path_pp_auto)
    X_train, X_test, y_train, y_test = split_df(df=df, flag=track_number)
    #Logistic Regression
    if track_number in [21, 22]:
        m_lr_second(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    elif track_number in [23, 24]:
        m_lr(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    #Random Forest Classifier
    elif track_number in [31, 32]:
        m_rf_second(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    elif track_number in [33, 34]:
        m_rf(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    #MLPClassifier
    elif track_number in [41, 42]:
        m_mlpc_second(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    elif track_number in [43, 44]:
        m_mlpc(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    #GradientBoostingClassifier
    elif track_number in [51, 52]:
        m_gb_second(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    elif track_number in [53, 54]:
        m_gb(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    #LinearDiscriminantAnalysis
    elif track_number in [61, 62]:
        m_lda_second(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    elif track_number in [63, 64]:
        m_lda(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    #GaussianNB
    elif track_number in [71, 72]:
        m_bayes_second(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    elif track_number in [73, 74]:
        m_bayes(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    #SVM (SVC)
    elif track_number in [81, 82]:
        m_svm_second(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    elif track_number in [83, 84]:
        m_svm(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    #KNeighborsClassifier
    elif track_number in [91, 92]:
        m_knn_second(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    elif track_number in [93, 94]:
        m_knn(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)


