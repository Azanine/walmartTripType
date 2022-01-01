import time
import joblib
import numpy as np
import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
from sklearn.ensemble import RandomForestClassifier, \
    GradientBoostingClassifier, \
    BaggingClassifier, AdaBoostClassifier
from sklearn.model_selection import cross_val_score

PATH = "D:/Documents/Machine Learning/Walmart Trip Type"
DATA_IMPORT_PATH = PATH + "/data/"
RETRAIN = True
RETRAIN_ALL = False
k_fold = True

def tune(x_train_df, y_train_df, retune_params=False):
    """
    get best parameters for the final model
    """
    num_eval = 1
    param_hyperopt = {
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(1)),
        'max_depth': scope.int(hp.quniform('max_depth', 3, 15, 1)),
        'min_child_weight': scope.int(hp.quniform('min_child_weight', 1, 10, 1)),
        'n_estimators': scope.int(hp.quniform('n_estimators', 25, 200, 25)),
        'gamma': hp.quniform('gamma', 0.5, 8, 0.5),
         }

    trials, best_params = hyperopt(param_hyperopt, num_eval, x_train_df, y_train_df)
    return best_params


def train_tune_model(x_train, y_train, x_val, y_val, x_train_sample, y_train_sample):
    """
    train the model with the best parameters(hyperopt)
    """
    load_params = True
    if load_params:
        best_params = joblib.load(PATH+"/hyperopt/tuned_params")
    else:
        best_params = tune(x_train_sample, y_train_sample)

    model = xgb_model(x_train, y_train, x_val, y_val, params=best_params)
    return model


def xgb_model(x_train, y_train, x_val=None, y_val=None, params=None):
    """
    train XGBoost model
    """
    start = time.time()
    chonda = ""
    if params:
        # params["verbosity"] = 2
        model = xgb.XGBClassifier(**params)
        chonda = "hyperopt_model"
    else:
        model = xgb.XGBClassifier(objective='multi:softmax',
                                  booster='gbtree',
                                  eval_metric='mlogloss',
                                  num_class='37',
                                  importance_type='weight',
                                  verbosity=1,
                                  n_jobs=-1)
    print("model fit starting")

    model.fit(x_train, y_train)

    joblib.dump(model, PATH + "/models/xgb" + chonda + "outliers.pkl")

    if (x_val is not None) & (y_val is not None):
        print("##### Results")
        print("Test Score: ", model.score(x_val, y_val.values))
    print("XGBoost model train time: ", time.time() - start)
    return model


def hyperopt(param_space, x_train_df, y_train_df, num_eval):
    """
    return best parameters for XGBoost model and Trials object,
    Triaols object containes every step of hyperopt procedure
    """
    start = time.time()

    def objective_function(params):
        clf = xgb.XGBClassifier(**params)
        score = cross_val_score(clf, x_train_df, y_train_df, cv=2).mean()
        return {'loss': -score, 'status': STATUS_OK}

    trials = Trials()
    best_param = fmin(objective_function,
                      param_space,
                      algo=tpe.suggest,
                      max_evals=num_eval,
                      trials=trials,
                      rstate=np.random.RandomState(1))

    joblib.dump(best_param, PATH+"/hyperopt/tuned_params")
    loss = [x['result']['loss'] for x in trials.trials]

    print("")
    print("##### Results")
    print("Score best parameters: ", min(loss) * -1)
    print("Best parameters: ", best_param)
    print("Time elapsed: ", time.time() - start)
    print("Parameter combinations evaluated: ", num_eval)

    return trials, best_param


def gradient_model(x_train, x_test, y_train):
    if False or RETRAIN_ALL:
        model = GradientBoostingClassifier(n_estimators=20,
                                           random_state=1111,
                                           max_depth=5,
                                           learning_rate=0.03,
                                           verbose=10)
        model.fit(x_train, y_train)
        joblib.dump(model, PATH + "/models/gradient_model.pkl")
    else:
        model = joblib.load(PATH + "/models/gradient_model.pkl")
        print("gradient_model load")
    predictions = model.predict_proba(x_test)
    return predictions, model


def forest_model(x_train, x_test, y_train):
    if False or RETRAIN_ALL:
        model = RandomForestClassifier(n_estimators=160,
                                       max_depth=8, random_state=1111,
                                       criterion='entropy')
        model.fit(x_train, y_train)
        joblib.dump(model, PATH + "/models/forest_model.pkl")
    else:
        model = joblib.load(PATH + "/models/forest_model.pkl")
        print("forest_model load")
    predictions = model.predict_proba(x_test)
    return predictions, model


def forest_ada_model(x_train, x_test, y_train):
    if False or RETRAIN_ALL:
        print("model train")
        model = RandomForestClassifier(n_estimators=160,
                                       max_depth=8, random_state=1111,
                                       criterion='entropy', verbose=10, n_jobs=-1)
        model = AdaBoostClassifier(base_estimator=model, n_estimators=25)
        model.fit(x_train, y_train)
        joblib.dump(model, PATH + "/models/forest_ada_model.pkl")
    else:
        model = joblib.load(PATH + "/models/forest_ada_model.pkl")
        print("forest_ada_model load")
    predictions = model.predict_proba(x_test)
    return predictions, model


def forest_calibrated(x_train, x_test, y_train):
    if False or RETRAIN_ALL:
        model = RandomForestClassifier(n_estimators=60,
                                       max_depth=8, random_state=1111,
                                       criterion='entropy', )
        model.fit(x_train, y_train)
        joblib.dump(model, PATH + "/models/forest_calibrated.pkl")
    else:
        model = joblib.load(PATH + "/models/forest_calibrated.pkl")
        print("forest_calibrated load")
    predictions = model.predict_proba(x_test)
    return predictions, model


def forest_bagging(x_train, x_test, y_train):
    if False or RETRAIN_ALL:
        model = RandomForestClassifier(n_estimators=150,
                                       max_depth=8, random_state=1111,
                                       criterion='entropy', )
        model = BaggingClassifier(base_estimator=model, max_features=0.80,
                                  n_jobs=-1, n_estimators=50)
        model.fit(x_train, y_train)
        joblib.dump(model, PATH + "/models/forest_bagging.pkl")
    else:
        model = joblib.load(PATH + "/models/forest_bagging.pkl")
        print("forest_bagging load")
    predictions = model.predict_proba(x_test)
    return predictions, model