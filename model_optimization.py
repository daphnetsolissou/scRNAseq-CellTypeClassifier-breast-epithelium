import optuna
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC


def objective(trial, train_x, train_y, clf_name, seed):
    estimator = define_new_model_estimator(clf_name, trial, seed)
    # -- Make a pipeline
    # pipeline = make_pipeline(estimator)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    # -- Evaluate the score by cross-validation
    score = cross_val_score(estimator, train_x, train_y, scoring='f1_weighted', cv=cv, n_jobs=-1)
    f1 = score.mean()  # calculate the mean
    return f1


def define_new_model_estimator(clf_name, trial, seed):
    if clf_name == 'svm':
        c_svm = trial.suggest_float('c_svm', 0.1, 100, log=True)
        # c_tol = trial.suggest_float('c_tol', 1e-6, 1e-2, log=True)
        c_loss = trial.suggest_categorical('c_loss', ['hinge', 'squared_hinge'])

        estimator = OneVsRestClassifier(LinearSVC(loss=c_loss, C=c_svm, random_state=seed,
                                                  multi_class='ovr', max_iter=100000))
    elif clf_name == 'lr':
        lr_solver = trial.suggest_categorical('lr_solver', ['newton-cg', 'lbfgs', 'newton-cholesky', 'sag', 'saga'])
        lr_c = trial.suggest_float('lr_c', 0.01, 10, log=True)

        estimator = LogisticRegression(solver=lr_solver, C=lr_c, random_state=seed, penalty='l2', multi_class='ovr',
                                       n_jobs=-1)
    elif clf_name == 'gnb':
        gnb_var_smoothing = trial.suggest_float("gnb_var_smoothing", 1e-10, 1e-6, log=True)

        estimator = GaussianNB(var_smoothing=gnb_var_smoothing)
    else:
        print('Unsupported classifier name. Please choose only between these options\n'
              '\t "lr": LinearRegression,\n'
              '\t "svm": Support Vector Machines,\n'
              '\t "gnb": Gaussian Naive Bayes')
        sys.exit(-1)

    if trial.should_prune():
        raise optuna.TrialPruned()

    return estimator


def retrain_best_model_objective(clf_name, trial, train_x, train_y, seed):
    estimator = define_new_model_estimator(clf_name=clf_name, trial=trial, seed=seed)
    # pipeline = make_pipeline(estimator)
    model = estimator.fit(train_x, train_y)
    return model