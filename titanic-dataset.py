import sklearn
from sklearn import preprocessing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

titanic_df = pd.read_csv('datasets/titanic_train.csv')

print("Before:" + str(titanic_df.count()))

# Removing unwanted fields
titanic_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], 'columns', inplace=True)

# Drop null fields
titanic_df = titanic_df.dropna()

print("After:" + str(titanic_df.count()))

# Visualizing relationships

fig, ax = plt.subplots(figsize=(12,8))
plt.scatter(titanic_df['Age'], titanic_df['Survived'])
plt.xlabel('Age')
plt.ylabel('Survived')
plt.show()

fig, ax = plt.subplots(figsize=(12,8))
plt.scatter(titanic_df['Fare'], titanic_df['Survived'])
plt.xlabel('Fare')
plt.ylabel('Survived')
plt.show()

pd.crosstab(titanic_df['Sex'], titanic_df['Survived'])
pd.crosstab(titanic_df['Pclass'], titanic_df['Survived'])

titanic_data_corr = titanic_df.corr()
print(titanic_data_corr)

# Show heat map for correlation
fig,ax = plt.subplots(figsize=(12, 10))
sns.heatmap(titanic_data_corr, annot=True)

# Convert Sex into digits M=1,F=0
label_encoding = preprocessing.LabelEncoder()
titanic_df['Sex'] = label_encoding.fit_transform(titanic_df['Sex'].astype(str))

# Multi value columns
titanic_df = pd.get_dummies(titanic_df, columns=['Embarked'])

titanic_df = titanic_df.sample(frac=1).reset_index(drop=True)

titanic_df.to_csv('datasets/titanic_processed.csv', index=False)