import numpy as np
import pandas as pd

from scipy.stats import pointbiserialr, spearmanr
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, roc_auc_score

# for SVM
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("datasets/adult.csv")
data.head()

data = data[data["workclass"] != "?"]
data = data[data["occupation"] != "?"]
data = data[data["native-country"] != "?"]

print(data.shape)

print(data.describe())

print(data["income"].value_counts()[0] / data.shape[0])
print(data["income"].value_counts()[1] / data.shape[0])

data.replace(['Divorced', 'Married-AF-spouse',
              'Married-civ-spouse', 'Married-spouse-absent',
              'Never-married','Separated','Widowed'],
             ['not married','married','married','married',
              'not married','not married','not married'], inplace = True)

category_col =['workclass', 'race', 'education','marital-status', 'occupation',
               'relationship', 'gender', 'native-country', 'income']
for col in category_col:
    b, c = np.unique(data[col], return_inverse=True)
    data[col] = c

print(data.head())

data.to_csv('datasets/adult_processed.csv', index=False)

col_names = data.columns

param=[]
correlation=[]
abs_corr=[]

for c in col_names:
    #Check if binary or continuous
    if c != "income":
        if len(data[c].unique()) <= 2:
            corr = spearmanr(data['income'],data[c])[0]
        else:
            corr = pointbiserialr(data['income'],data[c])[0]
        param.append(c)
        correlation.append(corr)
        abs_corr.append(abs(corr))

#Create dataframe for visualization
param_df=pd.DataFrame({'correlation':correlation,'parameter':param, 'abs_corr':abs_corr})

#Sort by absolute correlation
param_df=param_df.sort_values(by=['abs_corr'], ascending=False)

#Set parameter name as index
param_df=param_df.set_index('parameter')

print(param_df)

scoresCV = []
scores = []

## Decision Tree Classifier
for i in range(1, len(param_df)):
    new_df = data[param_df.index[0:i + 1].values]
    X = new_df.iloc[:, 1::]
    y = new_df.iloc[:, 0]
    clf = DecisionTreeClassifier()
    scoreCV = cross_val_score(clf, X, y, cv=10)
    scores.append(np.mean(scoreCV))

plt.figure(figsize=(15, 5))
plt.plot(range(1, len(scores) + 1), scores, '.-')
plt.axis("tight")
plt.title('Feature Selection', fontsize=14)
plt.xlabel('# Features', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.grid()
plt.show()

best_features=param_df.index[0:4].values
print('Best features:\t',best_features)

# SVM
predictors = ['age','workclass','education','educational-num',
              'marital-status', 'occupation','relationship','race','gender',
              'capital-gain','capital-loss','hours-per-week', 'native-country']

predictors = ['marital-status', 'educational-num', 'relationship', 'age']

pred_data = data[predictors] #X
target = data["income"] #y

algorithms = [
    #linear kernel
    [Pipeline([('scaler',StandardScaler()),
               ('svc',LinearSVC(random_state=1))]), predictors],
    #rbf kernel
    [Pipeline([('scaler',StandardScaler()),
               ('svc',SVC(kernel="rbf", random_state=1))]), predictors],
    #polynomial kernel
    [Pipeline([('scaler',StandardScaler()),
               ('svc',SVC(kernel='poly', random_state=1))]), predictors],
    #sigmoidf kernel
    [Pipeline([('scaler',StandardScaler()),
               ('svc',SVC(kernel='sigmoid', random_state=1))]), predictors]
]

alg_acc = {}
alg_auc = {}
for alg, predictors in algorithms:
    alg_acc[alg] = 0
    alg_auc[alg] = 0
i = 0

pred_data = data[predictors]  # X
target = data["income"]  # y

# stratified sampling
# random_state=1: we get the same splits every time we run this
# sss = StratifiedShuffleSplit(target, 10, test_size=0.1, random_state=1)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=1)
for train_index, test_index in sss.split(X=pred_data, y=target):
    i += 1
    train_data = data.iloc[train_index]
    test_data = data.iloc[test_index]
    train_data = pd.concat([train_data,
                            train_data[train_data["income"] == 1],
                            train_data[train_data["income"] == 1]])
    X_train, X_test = train_data[predictors], test_data[predictors]
    y_train, y_test = train_data["income"], test_data["income"]

    # Make predictions for each algorithm on each fold for alg, predictors in algorithms:
    for alg, predictors in algorithms:
        alg.fit(X_train, y_train)
        y_pred = alg.predict(X_test)
        acc_score = accuracy_score(y_test, y_pred)
        print(acc_score)
        alg_acc[alg] += acc_score
        auc_score = roc_auc_score(y_test, y_pred)
        print(auc_score)
        alg_auc[alg] += auc_score

for alg, predictors in algorithms:
    alg_acc[alg] /= 1
    alg_auc[alg] /= 1
    print("## %s ACC=%f" % (alg, alg_acc[alg]))
    print("## %s AUC=%f" % (alg, alg_auc[alg]))

