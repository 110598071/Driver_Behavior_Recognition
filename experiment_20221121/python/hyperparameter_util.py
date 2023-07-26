from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

def knn_params():
    params = {
        'n_neighbors': 3,
        'weights': 'distance',
        'algorithm': 'kd_tree',
        'leaf_size': 20,
        'p': 1,
    }
    return params

def random_forest_params():
    params = {
        'n_estimators': 170, 
        'criterion': 'entropy', 
        'max_depth': None, 
        'min_samples_split': 2, 
        'min_samples_leaf': 1, 
        'max_features': 'sqrt', 
        'max_leaf_nodes': None, 
        'class_weight': 'balanced_subsample',
    }
    return params

def xgboost_params():
    params = {
        'booster': "dart",
        'eta': 0.44,
        'gamma': 0,
        'max_depth': 7,
        'min_child_weight': 3,
        'tree_method': "auto",
        'grow_policy': "depthwise",
        'eval_metric': "logloss",
        'use_label_encoder': False,
    }
    return params

def xgboost_params_origin():
    params = {
        'eta': 0.5,
        'eval_metric': "mlogloss",
        'use_label_encoder': False,
    }
    return params

def logistic_regression_params():
    params = {
        'C': 1.55,
        'class_weight': None,
        'solver': 'newton-cg',
        'multi_class': 'auto',
        'penalty': 'l2',
        'max_iter': 1000,
    }
    return params

def mlp_classifier_params():
    params = {
        'hidden_layer_sizes': (30, 20, 15),
        'solver': 'sgd',
        'batch_size': 210,
        'learning_rate': 'adaptive',
        'random_state': 2,
        'max_iter': 1000,
    }
    return params

def ensemble_stacking_with_LR_params():
    estimators = [
        ('RF', RandomForestClassifier(**random_forest_params())),
        ('XGB', XGBClassifier(**xgboost_params_origin())),
        ('KNN', KNeighborsClassifier(**knn_params()))
    ]
    
    stacking_params = {
        'estimators': estimators,
        'final_estimator': LogisticRegression(**logistic_regression_params()),
        'cv': 10,
    }
    return stacking_params

def ensemble_stacking_with_MLP_params():
    estimators = [
        ('RF', RandomForestClassifier(**random_forest_params())),
        ('XGB', XGBClassifier(**xgboost_params_origin())),
        ('KNN', KNeighborsClassifier(**knn_params()))
    ]
    
    stacking_params = {
        'estimators': estimators,
        'final_estimator': MLPClassifier(**mlp_classifier_params()),
        'cv':8,
    }
    return stacking_params