import pandas as pd
import numpy as np
import tensorflow as tf
#from tensorflow.contrib import skflow
import sklearn.ensemble as ske
from sklearn import preprocessing, datasets, model_selection, tree, metrics, svm, linear_model

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)


train_df = pd.read_csv('~/Desktop/csv/train.csv')

#Encodes string values to integers so we can work with them nicely
def encoder(df):
	processed_df = df.copy()
	le = preprocessing.LabelEncoder()
	
	#Turns 'female' and 'male' values to 0 and 1 respectively
	processed_df.Sex = le.fit_transform(processed_df.Sex)
	
	#Turns Embarked locations to 0, 1, 2, 3
	#0 == Missing Value
	#1 == Cherbourg (C)
	#2 == Queens (Q)
	#3 == Southampton (S)
	processed_df.Embarked = le.fit_transform(processed_df.Embarked.fillna('0'))
	processed_df.Age = processed_df.Age.fillna(processed_df.Age.mean())
	processed_df = processed_df.drop(['Name', 'Ticket', 'Cabin'], axis=1)
	return processed_df

processed_df = encoder(train_df)
#print(processed_df.Age)

#Let's use a Decision Tree!
def DecisionTree(df):
	tree_df = df.copy()
	X = tree_df.drop(['Survived'], axis = 1).values
	y = tree_df['Survived'].values
	X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.2)

	clf_dt = tree.DecisionTreeClassifier(max_depth=10)
	clf_dt.fit (X_train, y_train)
	return clf_dt.score (X_test, y_test)
tree_df = DecisionTree(processed_df)
print("Decision Tree:", tree_df * 100)


def RandomForest(df):
	ran_df = df.copy()
	X = ran_df.drop(['Survived'], axis = 1).values
	y = ran_df['Survived'].values
	X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.2)
	clf_rf = ske.RandomForestClassifier(criterion='gini', min_samples_split=10, min_samples_leaf=5, n_estimators=100, max_depth=10, random_state=0)
	clf_rf.fit(X_train, y_train)
	return clf_rf.score(X_test, y_test)
ran_df= RandomForest(processed_df)
print("Random Forest:", ran_df * 100)

def LogReg(df):
	log_df = df.copy()
	X = log_df.drop(['Survived'], axis = 1).values
	y = log_df['Survived'].values
	X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.2)
	clf_lr = linear_model.LogisticRegression()
	clf_lr.fit(X_train, y_train)
	return clf_lr.score(X_test, y_test)
log_df=LogReg(processed_df)
print("Logistic Regression:", log_df*100)

def Voting(df):
	vot_df = df.copy()
	X = vot_df.drop(['Survived'], axis = 1).values
	y = vot_df['Survived'].values
	X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.2)
	eclf = ske.VotingClassifier([('dt', tree_df), ('rf', ran_df), ('lr', log_df)], voting='soft')
	eclf.fit(X_train, y_train)
	return eclf.score(X_test, y_test)
vot_df = Voting(processed_df)
print("The Best Is:", vot_df)

X = processed_df.drop(['Survived'], axis=1).values
y = processed_df['Survived'].values
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.2)

clf_dt=tree.DecisionTreeClassifier(max_depth=10)
clf_dt.fit(X_train,y_train)
clf_dt.score(X_test,y_test)


shuffle_validator= model_selection.StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=None)
def test_classifier(clf):
	scores = model_selection.cross_val_score(clf, X, y,cv=shuffle_validator)
	print("Accuracy: %0.4f (+/- %0.2f" % (scores.mean(), scores.std()))

clf_rf=ske.RandomForestClassifier(n_estimators=50)
test_classifier(clf_rf)

clf_gb = ske.GradientBoostingClassifier(n_estimators=50)
test_classifier(clf_gb)

eclf = ske.VotingClassifier([('dt', clf_dt), ('rf', clf_rf), ('gb', clf_gb)])
test_classifier(eclf)











