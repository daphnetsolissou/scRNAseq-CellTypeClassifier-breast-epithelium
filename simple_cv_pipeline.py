import pandas as pd
import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import TruncatedSVD


def objective(trial, X, y):
    # Define the search space for hyperparameters
    c_svm = trial.suggest_float('c_svm', 0.001, 1, log=True)
    gamma_svm = trial.suggest_float('gamma_svm', 0.001, 1, log=True)

    # Create the Random Forest Classifier with the suggested hyperparameters
    seed = 42
    clf = OneVsRestClassifier(SVC(kernel='rbf', gamma=gamma_svm, C=c_svm, random_state=seed, max_iter=1000),
                                        n_jobs=-1)

    mcc_scorer = make_scorer(matthews_corrcoef)

    if trial.should_prune():
        raise optuna.TrialPruned()

    svd = TruncatedSVD(n_components=2).fit(X)
    X = svd.transform(X)
    # Perform 5-fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring=mcc_scorer)

    # Return the mean accuracy score as the objective value to be maximized
    return scores.mean()



def optimize_hyperparameters(X, y):
    final_objective = lambda trial: objective(trial, X, y)
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(final_objective, n_trials=50, show_progress_bar=True, n_jobs=-1)

    best_params = study.best_params
    best_value = study.best_value

    print("Best Hyperparameters:", best_params)
    print("Best Value:", best_value)

    return best_params, best_value


