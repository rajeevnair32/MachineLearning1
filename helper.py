import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, validation_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import ShuffleSplit

"""
Helper functions
"""
def summarize_classification(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred, normalize=True)
    accuracy_count = accuracy_score(y_test, y_pred, normalize=False)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return {'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'accuracy_count': accuracy_count,
            'f1_score': f1
            }

"""
Helper functions
"""
def build_model(classifier_fn, name_of_y_col, name_of_x_cols, dataset, test_frac=0.2, options={}):
    X = dataset[name_of_x_cols]
    Y = dataset[name_of_y_col]

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_frac)
    t = time.process_time()
    model = classifier_fn(x_train, y_train, options=options)
    elapsed_training_time = time.process_time() - t

    y_pred = model.predict(x_test)
    y_pred_train = model.predict(x_train)

    train_summary = summarize_classification(y_train, y_pred_train)
    test_summary = summarize_classification(y_test, y_pred)

    pred_results = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
    model_crosstab = pd.crosstab(pred_results.y_pred, pred_results.y_test)

    return {'training': train_summary,
            'test': test_summary,
            'confusion_matrix': model_crosstab,
            'elapsed_time': elapsed_training_time
            }

"""
Helper function
"""
def compare_results(result_dict):
    COLUMNS = ['Classification Type', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Elapsed Time']
    train_df = pd.DataFrame([], columns=COLUMNS)
    test_df = pd.DataFrame([], columns=COLUMNS[:-1])
    for key in result_dict:
        tmp_arr = pd.DataFrame([[key,
                                 result_dict[key]['training']['accuracy'],
                                 result_dict[key]['training']['precision'],
                                 result_dict[key]['training']['recall'],
                                 result_dict[key]['training']['f1_score'],
                                 result_dict[key]['elapsed_time']]], columns=COLUMNS)
        train_df = insert_row(0, train_df, tmp_arr)

        tmp_arr = pd.DataFrame([[key,
                                 result_dict[key]['test']['accuracy'],
                                 result_dict[key]['test']['precision'],
                                 result_dict[key]['test']['recall'],
                                 result_dict[key]['test']['f1_score'],
                                 ]], columns=COLUMNS[:-1])
        test_df = insert_row(0, test_df, tmp_arr)

    return (train_df, test_df)

def insert_row(idx, df, df_insert):
    return df.iloc[:idx, ].append(df_insert).append(df.iloc[idx:, ]).reset_index(drop=True)

def print_results(result_dict):
    (train_df, test_df) = compare_results(result_dict)

    print()
    print("Results from Training Data")
    print(train_df.to_string(index=False))

    print()
    print("Results from Testing Data")
    print(test_df.to_string(index=False))

    plt.cla()
    plt.clf()
    ax = plt.gca()
    test_df.plot(kind='line', x='Classification Type', y='Accuracy', ax=ax)
    test_df.plot(kind='line', x='Classification Type', y='Precision', ax=ax)
    test_df.plot(kind='line', x='Classification Type', y='Recall', ax=ax)
    test_df.plot(kind='line', x='Classification Type', y='F1 Score', ax=ax)
    plt.legend()
    plt.title("Accuracy, Precision, Recall, F1")

    return plt

"""
Decision Tree
"""
def decision_tree_classifier(max_depth=None, criterion='gini', min_samples_split=2):
    return DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, min_samples_split=min_samples_split)

def decision_tree_fn(x_train, y_train, options={}):

    if 'max_depth' not in options:
        options['max_depth'] = 1.0
    if 'criterion' not in options:
        options['criterion'] = 'gini'
    if 'min_samples_split' not in options:
        options['min_samples_split'] = 2

    model = decision_tree_classifier(max_depth=options['max_depth'], criterion=options['criterion'], min_samples_split=options['min_samples_split'])
    model.fit(x_train, y_train)

    return model

"""
Neural Network
"""
def neural_network_classifier(hidden_layer_sizes=(100,), solver='lbfgs', activation='relu', learning_rate='constant'):
    return MLPClassifier(random_state=100,
                         hidden_layer_sizes=hidden_layer_sizes,
                         early_stopping=True,
                         max_iter=1000,
                         solver=solver,
                         activation=activation,
                         learning_rate=learning_rate)

def neural_network_fn(x_train, y_train, options={}):

    if 'solver' not in options:
        options['solver'] = 'lbfgs'
    if 'learning_rate' not in options:
        options['learning_rate'] = 'constant'
    if 'activation' not in options:
        options['activation'] = 'identity'
    if 'hidden_layer_sizes' not in options:
        options['hidden_layer_sizes'] = (100,)

    model = neural_network_classifier(hidden_layer_sizes=options['hidden_layer_sizes'], 
                                      solver=options['solver'], 
                                      learning_rate=options['learning_rate'],
                                      activation=options['activation'])
    model.fit(x_train, y_train)

    return model

"""
Boosting
"""
def ada_boosting_classifier(algorithm='SAMME.R', learning_rate=1.0, n_estimators=50):
    return AdaBoostClassifier(random_state=100, algorithm=algorithm, 
                              learning_rate=learning_rate, n_estimators=n_estimators)

def ada_boosting_fn(x_train, y_train, options={}):

    if 'algorithm' not in options:
        options['algorithm'] = 'SAMME.R'
    if 'learning_rate' not in options:
        options['learning_rate'] = 1
    if 'n_estimators' not in options:
        options['n_estimators'] = 100

    model = ada_boosting_classifier(learning_rate=options['learning_rate'], 
                                    n_estimators=options['n_estimators'],
                                    algorithm=options['algorithm'])
    model.fit(x_train, y_train)

    return model

def gradient_boosting_classifier(learning_rate=0.1, n_estimators=100, loss='exponential', criterion='friedman_mse'):
    return GradientBoostingClassifier(random_state=100, learning_rate=learning_rate, 
                                      n_estimators=n_estimators, loss=loss, criterion=criterion)

def gradient_boosting_fn(x_train, y_train, options={}):
    if 'criterion' not in options:
        options['criterion'] = 'friedman_mse'
    if 'learning_rate' not in options:
        options['learning_rate'] = 0.1
    if 'loss' not in options:
        options['loss'] = 'exponential'
    if 'n_estimators' not in options:
        options['n_estimators'] = 100

    model = gradient_boosting_classifier(learning_rate=options['learning_rate'], 
                                         n_estimators=options['n_estimators'],
                                         loss=options['loss'],
                                         criterion=options['criterion'])
    model.fit(x_train, y_train)

    return model

"""
SVM
"""
def get_best_svc_model(name_of_y_col, name_of_x_cols, dataset, test_frac=0.2):
    X = dataset[name_of_x_cols]
    Y = dataset[name_of_y_col]

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_frac)

    # defining parameter range
    param_grid = {'C': [0.1, 1, 10, 100],
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                  'kernel': ['linear', 'sigmoid', 'rbf']}

    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)

    # fitting the model for grid search
    grid.fit(x_train, y_train)

    # print best parameter after tuning
    print(grid.best_params_)

    # print how our model looks after hyper-parameter tuning
    print(grid.best_estimator_)

    grid_predictions = grid.predict(x_test)

    return summarize_classification(y_test, grid_predictions)

def linear_svm_classifier(C=1.0, max_iter=1000, tol=1e-3, loss='squared_hinge'):
    return LinearSVC(C=C, max_iter=max_iter, tol=tol, loss=loss)

def linear_svm_fn(x_train, y_train, options={}):
    if 'C' not in options:
        options['C'] = 1.0
    if 'max_iter' not in options:
        options['max_iter'] = 1000
    if 'loss' not in options:
        options['loss'] = 'squared_hinge'
    if 'tol' not in options:
        options['tol'] = 1e-3

    model = linear_svm_classifier(C = options['C'],
                                  max_iter= options['max_iter'],
                                  loss=options['loss'],
                                  tol=options['tol'])
    model.fit(x_train, y_train)
    return model

def svm_linear_classifier(C=1.0, max_iter=1000, tol=1e-3, gamma=1):
    return SVC(kernel='linear', C=C, max_iter=max_iter, tol=tol, gamma=gamma)

def svm_linear_fn(x_train, y_train, options={}):
    if 'C' not in options:
        options['C'] = 1.0
    if 'max_iter' not in options:
        options['max_iter'] = 1000
    if 'gamma' not in options:
        options['gamma'] = 1
    if 'tol' not in options:
        options['tol'] = 1e-3

    model = svm_linear_classifier(C = options['C'],
                                  max_iter= options['max_iter'],
                                  gamma=options['gamma'],
                                  tol=options['tol'])
    model.fit(x_train, y_train)

    return model

def svm_sigmoid_classifier(C=1.0, max_iter=1000, tol=1e-3, gamma=1):
    return SVC(kernel='sigmoid', C=C, max_iter=max_iter, tol=tol, gamma=gamma)

def svm_sigmoid_fn(x_train, y_train, options={}):
    if 'C' not in options:
        options['C'] = 1.0
    if 'max_iter' not in options:
        options['max_iter'] = 1000
    if 'gamma' not in options:
        options['gamma'] = 1
    if 'tol' not in options:
        options['tol'] = 1e-3

    model = svm_sigmoid_classifier(C = options['C'],
                                max_iter= options['max_iter'],
                                gamma=options['gamma'],
                                tol=options['tol'])
    model.fit(x_train, y_train)

    return model

def svm_rbf_classifier(C=1.0, max_iter=1000, tol=1e-3, gamma=1):
    return SVC(kernel='rbf', C=C, max_iter=max_iter, tol=tol, gamma=gamma)

def svm_rbf_fn(x_train, y_train, options={}):
    if 'C' not in options:
        options['C'] = 1.0
    if 'max_iter' not in options:
        options['max_iter'] = 1000
    if 'gamma' not in options:
        options['gamma'] = 1
    if 'tol' not in options:
        options['tol'] = 1e-3

    model = svm_rbf_classifier(C = options['C'],
                               max_iter= options['max_iter'],
                               gamma=options['gamma'],
                               tol=options['tol'])
    model.fit(x_train, y_train)

    return model

"""
kNN
"""
def knearest_neigbors_classifier( k=5, weights='uniform', algorithm='auto', leaf_size=30, p=2,
                          metric='minkowski'):
    return KNeighborsClassifier(n_neighbors=k, weights=weights, algorithm=algorithm, leaf_size=leaf_size, p=p,
                                 metric=metric)

def knearest_neighbors_fn(x_train, y_train, options={}):
    if 'k' not in options:
        options['k'] = 5
    if 'weights' not in options:
        options['weights'] = 'uniform'
    if 'algorithm' not in options:
        options['algorithm'] = 'auto'
    if 'leaf_size' not in options:
        options['leaf_size'] = 30
    if 'p' not in options:
        options['p'] = 2
    if 'metric' not in options:
        options['metric'] = 'minkowski'

    model = knearest_neigbors_classifier(k=options['k'],
                                         weights=options['weights'],
                                         algorithm=options['algorithm'],
                                         leaf_size=options['leaf_size'],
                                         p=options['p'],
                                         metric=options['metric'])
    model.fit(x_train, y_train)

    return model

def find_best_param(classifier_fn, name_of_y_col, name_of_x_cols, dataset, param_grid={}, test_frac=0.2):
    X = dataset[name_of_x_cols]
    Y = dataset[name_of_y_col]

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_frac)

    grid = GridSearchCV(classifier_fn(), param_grid, refit=True, verbose=3)

    # fitting the model for grid search
    grid.fit(x_train, y_train)

    # print best parameter after tuning
    print(grid.best_params_)

    # print how our model looks after hyper-parameter tuning
    print(grid.best_estimator_)

    grid_predictions = grid.predict(x_test)

    return summarize_classification(y_test, grid_predictions)

def plot_learning_curve(classifier_fn, name_of_y_col, name_of_x_cols, dataset, title, axes=None, ylim=None, n_jobs=4):
    X = dataset[name_of_x_cols]
    y = dataset[name_of_y_col]

    plt.cla()
    plt.clf()
    
    if axes is None:
        _, axes = plt.subplots(3, 1, figsize=(5, 15))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    train_sizes = np.linspace(.1, 1.0, 5)

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(classifier_fn, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt

def plot_loss_curve(name_of_y_col, name_of_x_cols, dataset, learning_rate):
    X = dataset[name_of_x_cols]
    y = dataset[name_of_y_col]

    plt.cla()
    plt.clf()
    
    clf = MLPClassifier(learning_rate_init=learning_rate)
    clf.fit(X, y)
    plt.title('MLP Classifier Loss Curve')
    plt.ylabel('Cost')
    plt.xlabel('Iterations')
    plt.title('Learning rate: ' + str(learning_rate))
    plt.plot(clf.loss_curve_)
    return plt

def plot_validation_curve(classifier_fn, name_of_y_col, name_of_x_cols, dataset, title, param_name, param_range):
    X = dataset[name_of_x_cols]
    y = dataset[name_of_y_col]

    train_scores, test_scores = validation_curve(
        classifier_fn, X, y, param_name=param_name, param_range=param_range,
        scoring="accuracy", n_jobs=1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.cla()
    plt.clf()
    
    plt.title(title)
    plt.xlabel(param_name)
    plt.ylabel("Score")
    #plt.ylim(0.0, 1.1)
    lw = 2
    plt.semilogx(param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")

    return plt
