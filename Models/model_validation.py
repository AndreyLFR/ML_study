from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import joblib


def results(y_test, y_pred, time=0):
    report = classification_report(y_test, y_pred, target_names=['0', '1'])
    print(report)
    print('\nПлощадь под ROC-кривой - ' + str(round(roc_auc_score(y_test, y_pred), 4)))
    if time != 0: print('\nВремя работы кода - ' + str(round(time, 4)) + ' сек')


def save_model(name_m, model):
    joblib_file = f'Models/StorageModels/{name_m}.pkl'
    joblib.dump(model, joblib_file)
    print(f'Save model {name_m}')


def load_model(name_m):
    joblib_model = joblib.load(f'Models/StorageModels/{name_m}.pkl')
    return joblib_model