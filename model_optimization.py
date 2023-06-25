import optuna
import sys
import numpy as np
import scipy

from optuna.samplers import TPESampler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(trial, train_x, train_y, clf_name, seed):
    estimator = define_new_model_estimator(clf_name, trial, seed)
    # -- Make a pipeline
    # pipeline = make_pipeline(estimator)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    # if clf_name == 'svm':
    # scaler = StandardScaler()
    # train_x = scaler.fit_transform(train_x, train_y)
    # -- Evaluate the score by cross-validation
    score = cross_val_score(estimator, train_x, train_y, scoring='balanced_accuracy', cv=cv, n_jobs=-1)
    ba = score.mean()  # calculate the mean
    return ba


def define_new_model_estimator(clf_name, trial, seed):
    if clf_name == 'svm':
        c_svm = trial.suggest_float('c_svm', 0.001, 1, log=True)
        gamma_svm = trial.suggest_float('gamma_svm', 0.001, 1, log=True)

        estimator = OneVsRestClassifier(SVC(kernel='rbf', gamma=gamma_svm, C=c_svm, random_state=seed, max_iter=1000),
                                        n_jobs=-1)
    elif clf_name == 'lr':
        lr_solver = trial.suggest_categorical('lr_solver', ['newton-cg', 'lbfgs', 'newton-cholesky', 'sag', 'saga'])
        lr_c = trial.suggest_float('lr_c', 0.0001, 1, log=True)

        estimator = LogisticRegression(solver=lr_solver, C=lr_c, random_state=seed, penalty='l2',
                                       multi_class='ovr', max_iter=1000, n_jobs=-1)
    elif clf_name == 'gnb':
        gnb_var_smoothing = trial.suggest_float("gnb_var_smoothing", 1e-10, 1e-6, log=True)

        estimator = GaussianNB(var_smoothing=gnb_var_smoothing)
    elif clf_name == "rf":
        rf_n_estimators = trial.suggest_int('rf_n_estimators', 10, 200) #, step=50)
        rf_max_depth = trial.suggest_int('rf_max_depth', 1, 3)
        rf_max_features = trial.suggest_categorical('rf_max_features', ['sqrt', 'log2', None])
        #rf_min_impurity_decrease = trial.suggest_float('rf_min_impurity_decrease', 0.0, 0.2 )#, step=0.05)
        #rf_min_samples_leaf = trial.suggest_float('rf_min_samples_leaf', 2, 10 )#, step=2)

        estimator = RandomForestClassifier(n_estimators=rf_n_estimators, criterion='gini', 
                                           max_depth=rf_max_depth, max_features=rf_max_features,
                                           #min_impurity_decrease=rf_min_impurity_decrease,
                                           #min_samples_leaf=rf_min_samples_leaf, 
                                           n_jobs=-1, random_state=seed)
    elif clf_name == 'xgb':
        xgb_eta = trial.suggest_float('xgb_eta', 0.0, 0.5)
        xgb_gamma = trial.suggest_int('xgb_gamma', 0, 50)
        xgb_min_child_weight = trial.suggest_float('xgb_min_child_weight', 0, 5)
        #xgb_subsample = trial.suggest_float('xgb_subsample', 0.1, 1.0)
        #xgb_max_leaves = trial.suggest_int('xgb_max_leaves', 0, 100)

        estimator = XGBClassifier(booster='gbtree', reg_lambda=0.1, eta=xgb_eta, gamma=xgb_gamma, 
                                  min_child_weight=xgb_min_child_weight)#,
                                  #subsample=xgb_subsample, max_leaves=xgb_max_leaves)
    else:
        print('Unsupported classifier name. Please choose only between these options\n'
              '\t "lr": LinearRegression,\n'
              '\t "svm": Support Vector Machines,\n'
              '\t "gnb": Gaussian Naive Bayes,\n'
              '\t "rf": Random Forests,\n'
              '\t "xgb": XGBoost') 
        sys.exit(-1)

    if trial.should_prune():
        raise optuna.TrialPruned()

    return estimator


def retrain_best_model_objective(clf_name, trial, train_x, train_y, seed):
    estimator = define_new_model_estimator(clf_name=clf_name, trial=trial, seed=seed)
    # pipeline = make_pipeline(estimator)
    model = estimator.fit(train_x, train_y)
    return model


def optimize_one_model(data_x, data_y, clf_name, seed):
    my_objective = lambda trial: objective(trial, data_x, data_y, clf_name, seed)
    sampler = TPESampler(seed=seed)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(my_objective, n_trials=10, show_progress_bar=True, n_jobs=-1)

    best_estimator = retrain_best_model_objective(clf_name, study.best_trial, data_x, data_y, seed)
    best_params = study.best_params

    return best_estimator, best_params


# def run_optimization_all_classifiers(x, y, seed=148):
#     estimators_dict = {}
#     params_dict = {}
#     for key, name in classifiers.items():
#         print(f'Starting optimization for classifier {name}')
#         # res_df = nested_cv(clf, hyperparam_grids[key], x, y)
#         best_estimator, best_params = optimize_one_model(x, y, name, seed)
#         estimators_dict[name] = best_estimator
#         params_dict[name] = best_params
#         # break
#     return estimators_dict, params_dict