import optuna
from optuna.visualization import plot_param_importances
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import numpy as np
import util
import time
import hyperparameter_util

TRIALS = 100
MODEL_NAME = ['Knn', 'Random Forest', 'XGBoost', "Ensemble Stacking"]

TRIALS_INDEX = 1
ACTION_DATA = []
ACTION_LABEL = []

DATA_STANDARIZATION = False
REMOVE_OUTLIERS = False
OUTLIERS_CLS = []
OUTLIERS_INDEX = [[]]

def knn_objective(trial):
    params = {
        'n_neighbors': trial.suggest_int('n_neighbors', 3, 9, 2),
        'weights': trial.suggest_categorical("weights", ["uniform", "distance"]),
        'algorithm': trial.suggest_categorical("algorithm", ["auto", "ball_tree", "kd_tree", "brute"]),
        'leaf_size': trial.suggest_int('leaf_size', 10, 50, 2),
        'p': trial.suggest_int('p', 1, 5),
    }

    clf = KNeighborsClassifier(**params)
    return fit_and_return_score(clf)

def random_forest_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 200, 10),
        'criterion': trial.suggest_categorical("criterion", ["gini", "entropy"]),
        'max_depth': trial.suggest_categorical("max_depth", [5, 10, 15, None]),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 5),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
        'max_features': trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        'max_leaf_nodes': trial.suggest_categorical("max_leaf_nodes", [5, 10, 15, None]),
        'class_weight': trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample", None]),
    }

    clf = RandomForestClassifier(**params)
    return fit_and_return_score(clf)

def xgboost_objective(trial):
    params = {
        'booster': trial.suggest_categorical('booster', ["gbtree", "dart"]),
        'eta': trial.suggest_float("eta", 0.0, 1.0, step=0.02),
        'gamma': trial.suggest_int('gamma', 0, 10),
        'max_depth': trial.suggest_int('max_depth', 5, 12),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
        'tree_method': trial.suggest_categorical("tree_method", ["auto", "exact", "approx", "hist"]),
        'grow_policy': trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
        'eval_metric': trial.suggest_categorical("eval_metric", ["logloss", "merror", "mlogloss"]),
        'use_label_encoder': False,
    }

    clf = XGBClassifier(**params)
    return fit_and_return_score(clf)

def ensemble_stacking_objective(trial):
    estimators = [
        ('RF', RandomForestClassifier(**hyperparameter_util.random_forest_params())),
        ('XGB', XGBClassifier(**hyperparameter_util.xgboost_params_origin())),
        ('KNN', KNeighborsClassifier(**hyperparameter_util.knn_params()))
    ]

    LR_params = {
        'C': trial.suggest_float("C", 0.1, 2.0, step=0.05),
        'class_weight': trial.suggest_categorical('class_weight', ["balanced", None]),
        'solver': trial.suggest_categorical('solver', ["sag", "saga", "lbfgs", "newton-cg"]),
        'multi_class': trial.suggest_categorical('multi_class', ["multinomial", "auto"]),
        'penalty': trial.suggest_categorical('penalty', ["l2", 'none']),
        'max_iter': 1000,
    }

    # MLP_params = {
    #     'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', [(30, 25, 20, 15), (30, 20, 15), (30, 15)]),
    #     'solver': trial.suggest_categorical('solver', ['lbfgs', 'sgd', 'adam']),
    #     'batch_size': trial.suggest_int('batch_size', 150, 220, 10),
    #     'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'invscaling', 'adaptive']),
    #     'random_state': trial.suggest_categorical("random_state", [1, 2, 3, None]),
    #     'max_iter': 1500,
    # }
    
    stacking_params = {
        'estimators': estimators,
        'final_estimator': LogisticRegression(**LR_params),
        # 'final_estimator': MLPClassifier(**MLP_params),
        'cv': trial.suggest_categorical('cv', [3, 5, 8, 10, None]),
    }

    clf = StackingClassifier(**stacking_params)
    return fit_and_return_score(clf)

def choose_objective():
    objective_list = []
    objective_list.append(knn_objective)
    objective_list.append(random_forest_objective)
    objective_list.append(xgboost_objective)
    objective_list.append(ensemble_stacking_objective)
    return objective_list

def fit_and_return_score(clf):
    global TRIALS_INDEX
    accuracy_list = []
    kf = KFold(n_splits = 10, shuffle=True, random_state=TRIALS_INDEX)
    TRIALS_INDEX += 1
    
    for train_index, test_index in kf.split(ACTION_DATA):
        split_data_train, split_label_train = ACTION_DATA[train_index], ACTION_LABEL[train_index]
        split_data_test, split_label_test = ACTION_DATA[test_index], ACTION_LABEL[test_index]

        if DATA_STANDARIZATION:
            scaler = StandardScaler()
            split_data_train = scaler.fit_transform(split_data_train)
            split_data_test = scaler.transform(split_data_test)
        if REMOVE_OUTLIERS:
            for i, cls in enumerate(OUTLIERS_CLS):
                for index in OUTLIERS_INDEX[i]:
                    split_data_train, split_label_train = util.remove_specific_class_and_feature_outliers(split_data_train, split_label_train, cls, index)
                    
        clf.fit(split_data_train, split_label_train)
        accuracy_list.append(np.round(clf.score(split_data_test, split_label_test), 4))
    return np.round(np.mean(accuracy_list), 4)

def draw_importancy_plot(study):
    plotly_config = {"staticPlot": True}
    fig = plot_param_importances(study)
    fig.show(config=plotly_config)

def set_global_variables(action_data, action_label, data_standarization, remove_outliers, outliers_cls, outliers_index):
    global ACTION_DATA
    global ACTION_LABEL
    global DATA_STANDARIZATION
    global REMOVE_OUTLIERS
    global OUTLIERS_CLS
    global OUTLIERS_INDEX

    DATA_STANDARIZATION = data_standarization
    REMOVE_OUTLIERS = remove_outliers
    OUTLIERS_CLS = outliers_cls
    OUTLIERS_INDEX = outliers_index
    ACTION_DATA = action_data
    ACTION_LABEL = action_label

def get_hyperparameter_optimization(model_name, action_data, action_label, data_standarization, remove_outliers, outliers_cls, outliers_index):
    set_global_variables(action_data, action_label, data_standarization, remove_outliers, outliers_cls, outliers_index)

    for i, name in enumerate(MODEL_NAME):
        if name == model_name:
            model_index = i

    start = time.time()
    # optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(multivariate=True, group=True)
    study = optuna.create_study(direction='maximize', sampler=sampler, pruner=optuna.pruners.HyperbandPruner())
    study.optimize(choose_objective()[model_index], n_trials = TRIALS, n_jobs=1)
    end = time.time()

    print("K-Fold cross validation result:")
    print("{} trials optimization time: {:.2f}".format(TRIALS, end - start))
    print('Best trial parameters:', study.best_trial.params)
    print("validate accuracy: ", study.best_trial.value)
    print()

    draw_importancy_plot(study)
    return study.best_trial.params, ["{:.2%}".format(study.best_trial.value), TRIALS, "{:.2f}".format(end - start)]

if __name__ == '__main__':
    get_hyperparameter_optimization('Knn')