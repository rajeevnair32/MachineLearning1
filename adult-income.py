import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC

from helper import build_model, print_results
from helper import decision_tree_fn
from helper import neural_network_fn
from helper import ada_boosting_fn, gradient_boosting_fn
from helper import svm_linear_fn, svm_sigmoid_fn, svm_rbf_fn, linear_svm_fn, get_best_svc_model
from helper import knearest_neighbors_fn
from helper import find_best_param
from helper import plot_validation_curve, plot_learning_curve, plot_loss_curve

import warnings
warnings.filterwarnings("ignore")

"""
Analyzing Adult Income Dataset
"""

adult_df = pd.read_csv('datasets/adult_processed.csv')
FEATURES = ['marital-status', 'educational-num', 'relationship', 'age']
fig_path = './figures/adult-income/'

def find_best():

    # defining parameter range
#    param_grid = {'C' : [0.01, 0.1, 0.5, 1], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['linear', 'rbf', 'sigmoid']}
#    find_best_param(SVC, 'income', FEATURES, adult_df, param_grid)

    # defining parameter range
#    param_grid = {'C' : [0.01, 0.1, 0.5, 1], 'loss': ['hinge', 'squared_hinge']}
#    find_best_param(LinearSVC, 'income', FEATURES, adult_df, param_grid)

    # defining parameter range
#    param_grid = {'activation' : ['identity', 'logistic', 'tanh', 'relu'], 'solver' : ['lbfgs','sgd','adam'], 'learning_rate': ['constant','invscaling','adaptive']}
#    find_best_param(MLPClassifier, 'income', FEATURES, adult_df, param_grid)

    # defining parameter range
#    param_grid = {'n_estimators': [20, 50, 100, 200, 500, 1000], 'learning_rate': [1, 0.1, 0.01, 0.001], 'algorithm' : ['SAMME', 'SAMME.R']}
#    find_best_param(AdaBoostClassifier, 'income', FEATURES, adult_df, param_grid)

    # defining parameter range
    param_grid = {'loss': ['deviance', 'exponential'], 'criterion': ['friedman_mse', 'mse', 'mae'], 'n_estimators': [10, 20, 50, 100, 200], 'learning_rate': [1, 0.1, 0.01, 0.001],}
    find_best_param(GradientBoostingClassifier, 'income', FEATURES, adult_df, param_grid)

    # defining parameter range
    param_grid = {'criterion': ['gini', 'entropy'], 'max_depth': np.arange(1, 10, 1), 'min_samples_split': np.arange(2, 21, 1)}
    find_best_param(DecisionTreeClassifier, 'income', FEATURES, adult_df, param_grid)

    # defining parameter range - using only odd numbers
    param_grid = {'n_neighbors': np.arange(1, 42, 2)}
    find_best_param(KNeighborsClassifier, 'income', FEATURES, adult_df, param_grid)

# Need to run only once to get best params for respective classifier
#find_best()

result_dict = {
               'income - kNearestNeighbors': build_model(knearest_neighbors_fn, 'income', FEATURES, adult_df, options={'k': 35}),
               'income - Linear SVM': build_model(linear_svm_fn, 'income', FEATURES, adult_df, options={'C': 0.1, 'loss': 'hinge'}),
               'income - SVM Linear': build_model(svm_linear_fn, 'income', FEATURES, adult_df, options={'C': 1, 'gamma': 0.1}),
               'income - SVM RBF': build_model(svm_rbf_fn, 'income', FEATURES, adult_df, options={'C': 1, 'gamma': 0.1}),
               'income - Ada Boosting': build_model(ada_boosting_fn, 'income', FEATURES, adult_df, options={'algorithm': 'SAMME.R', 'learning_rate': 1, 'n_estimators': 500}),
               'income - Gradient Boosting': build_model(gradient_boosting_fn, 'income', FEATURES, adult_df, options={'criterion': 'friedman_mse', 'learning_rate': 0.1, 'loss': 'exponential', 'n_estimators': 100}),
               'income - Neural networks': build_model(neural_network_fn, 'income', FEATURES, adult_df, options={'activation':'tanh', 'learning_rate':'invscaling', 'solver': 'adam'}),
               'income - Decision_tree': build_model(decision_tree_fn, 'income', FEATURES, adult_df, options={'criterion': 'gini', 'max_depth': 3, 'min_samples_split': 2})
               }

# Running code with default values
plt = print_results(result_dict)
#plt.show()
plt.savefig(fig_path + 'results.png')

title = "Learning Curves for Decision Tree"
plt = plot_learning_curve(DecisionTreeClassifier(criterion='gini', max_depth= 8, min_samples_split=12), 'income', FEATURES, adult_df, title, ylim=(0.4, 1.01))
#plt.show()
plt.savefig(fig_path + 'learning_curve_dt.png')

title = "Learning Curves for Neural Networks"
plt = plot_learning_curve(MLPClassifier(activation='tanh', learning_rate='invscaling', solver='adam'), 'income', FEATURES, adult_df, title, ylim=(0.4, 1.01))
#plt.show()
plt.savefig(fig_path + 'learning_curve_neural.png')

title = "Learning Curves for AdaBoost"
plt = plot_learning_curve(AdaBoostClassifier(algorithm='SAMME.R', learning_rate=1, n_estimators= 500), 'income', FEATURES, adult_df, title, ylim=(0.4, 1.01))
#plt.show()
plt.savefig(fig_path + 'learning_curve_adaboost.png')

title = "Learning Curves for GradientBoost"
plt = plot_learning_curve(GradientBoostingClassifier(criterion='friedman_mse', learning_rate=0.1, loss='deviance', n_estimators=100), 'income', FEATURES, adult_df, title, ylim=(0.4, 1.01))
#plt.show()
plt.savefig(fig_path + 'learning_curve_gradientboost.png')

title = "Learning Curves for KNeighbors"
plt = plot_learning_curve(KNeighborsClassifier(n_neighbors=33), 'income', FEATURES, adult_df, title, ylim=(0.4, 1.01))
#plt.show()
plt.savefig(fig_path + 'learning_curve_kneighbors.png')

title = "Learning Curves for SVC"
plt = plot_learning_curve(SVC(C=1, gamma=0.1, kernel='rbf'), 'income', FEATURES, adult_df, title, ylim=(0.4, 1.01))
#plt.show()
plt.savefig(fig_path + 'learning_curve_svc.png')

title = "Learning Curves for LinearSVC"
plt = plot_learning_curve(LinearSVC(C=0.1, loss='hinge'), 'income', FEATURES, adult_df, title, ylim=(0.4, 1.01))
#plt.show()
plt.savefig(fig_path + 'learning_curve_linearsvc.png')


param_name = "gamma"
param_range = np.logspace(-6, -1, 5)
title='Validation Curve with SVC'
plt = plot_validation_curve(SVC(C=1, gamma=0.1, kernel='rbf'), 'income', FEATURES, adult_df, title, param_name, param_range)
#plt.show()
plt.savefig(fig_path + 'validation_curve_svc.png')

param_name = "max_depth"
param_range = np.arange(1, 21)
title='Validation Curve with Decision Tree'
plt = plot_validation_curve(DecisionTreeClassifier(criterion='gini', max_depth= 8, min_samples_split=12), 'income', FEATURES, adult_df, title, param_name, param_range)
#plt.show()
plt.savefig(fig_path + 'validation_curve_dt.png')

param_name = "min_samples_split"
param_range = np.arange(2, 21)
title='Validation Curve with Decision Tree'
plt = plot_validation_curve(DecisionTreeClassifier(criterion='gini', max_depth= 8, min_samples_split=12), 'income', FEATURES, adult_df, title, param_name, param_range)
#plt.show()
plt.savefig(fig_path + 'validation_curve_dt2.png')

param_name = "n_estimators"
param_range = np.arange(10, 500, 10)
title='Validation Curve with AdaBoost Classifier'
plt = plot_validation_curve(AdaBoostClassifier(algorithm='SAMME.R', learning_rate=1, n_estimators=500), 'income', FEATURES, adult_df, title, param_name, param_range)
#plt.show()
plt.savefig(fig_path + 'validation_curve_adaboost.png')

param_name = "n_estimators"
param_range = np.arange(10, 500, 10)
title='Validation Curve with GradientBoost Classifier'
plt = plot_validation_curve(GradientBoostingClassifier(criterion='friedman_mse', learning_rate=0.1, loss='deviance', n_estimators=100), 'income', FEATURES, adult_df, title, param_name, param_range)
#plt.show()
plt.savefig(fig_path + 'validation_curve_gradientboost.png')

param_name = "n_neighbors"
param_range = np.arange(1, 42, 2)
title='Validation Curve with K Neighbor Classifier'
plt = plot_validation_curve(KNeighborsClassifier(n_neighbors=35), 'income', FEATURES, adult_df, title, param_name, param_range)
#plt.show()
plt.savefig(fig_path + 'validation_curve_kneighbors.png')

plt = plot_loss_curve('income', FEATURES, adult_df, 0.001)
#plt.show()
plt.savefig(fig_path + 'loss_curve_0.001.png')

plt = plot_loss_curve('income', FEATURES, adult_df, 0.1)
#plt.show()
plt.savefig(fig_path + 'loss_curve_0.1.png')

plt = plot_loss_curve('income', FEATURES, adult_df, 1)
#plt.show()
plt.savefig(fig_path + 'loss_curve_1.png')
