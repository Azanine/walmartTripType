
import time
from collections import Counter
from itertools import product
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
from sklearn.ensemble import RandomForestClassifier, \
                             GradientBoostingClassifier, \
                             BaggingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.model_selection import cross_val_score

PATH = "D:/Documents/Machine Learning/Walmart Trip Type"
DATA_IMPORT_PATH = PATH + "/data/"
RETRAIN = True
RETRAIN_ALL = False
k_fold = True


def read_data(train_n_rows=None, test_n_rows=None):
    """
    Read train and test data
    """
    train = pd.read_csv(DATA_IMPORT_PATH + "train.csv", nrows=train_n_rows)
    test = pd.read_csv(DATA_IMPORT_PATH + "test.csv", nrows=test_n_rows)
    print("files read")
    return train, test


def export_sub(prediction, filename):
    """
    Export predictions to submittable file
    """
    sub = pd.read_csv(PATH + "/data/sample_submission.csv")
    prediction = pd.DataFrame(data=prediction, index=None, columns=sub.columns[1:])
    prediction = pd.concat([sub["VisitNumber"], prediction], axis=1)
    prediction.to_csv(PATH+"/predictions/" + filename + ".csv", index=False)
    return 1


def detect_outliers(df, n, features):
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than n outliers according
    to the Tukey method.
    """
    outlier_indices = []

    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col], 75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1

        # outlier step
        outlier_step = 1.5 * IQR

        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index

        # append the found outlier indices for col to the list of outlier indices
        outlier_indices.extend(outlier_list_col)

    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)

    return multiple_outliers


def split_data(train, test):
    """
    strip dataframes of data that are not used in training
    """
    return train.iloc[:, 2:], test.iloc[:, 1:], train['TripType']

def k_fold_comb():
    """
    get all possible combos of params in a nested dictionary
    """
    params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.9, 1, 1.1, 1.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [4, 5, 6, 7, 8],
        'n_estimators': [50, 100, 150, 200]
        }

    value_combos = list(product(*params.values()))
    keys = list(params.keys())

    final_combo_dict = {}
    for i in range(0, len(value_combos)):
        combo_dict = {}
        for j in range(0, len(keys)):
            combo_dict[keys[j]] = value_combos[i][j]
        final_combo_dict[i] = combo_dict
    joblib.dump(final_combo_dict, PATH + "/temp/dict.pkl")

    return final_combo_dict










