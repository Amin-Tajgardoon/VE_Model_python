'''
Created on Sep 27, 2016

@author: mot16


'''
from imblearn.over_sampling import SMOTE
from sklearn import svm
from sklearn import tree
from sklearn.cross_validation import train_test_split  
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from default.evaluation import crossValidation, test
from default.plots import plotAllCurves, plotFreq
from default.readData import readData, saveData
import numpy as np
from sklearn import preprocessing
import pandas as pd


seed = np.random.RandomState(100)

X, y = readData("C:\\Users\\mot16\\projects\\master\\data\\gsk_108134_cleaned_final_binary_class_missing_dates.csv");


X.sex[X.sex == 1] = 2
X.sex[X.sex == 0] = 1
X.cty_cod[X.cty_cod == 1] = 2
X.cty_cod[X.cty_cod == 0] = 1

### plotFreq(X, y)

X = X.drop(labels=['schedule', 'days_to_ili_start', 'ili_nb','days_to_infect'], axis=1)


#X = X[X.group_nb == 1 && ]

X_dum = pd.get_dummies(X, columns = ['center', 'ethnic', 'race', 'sex', 'group_nb', 'cty_cod'])
X_dum.age = preprocessing.MinMaxScaler().fit_transform(X.age)

X = X_dum

#saveData(X,y,"C:\\Users\\mot16\\projects\\master\\data\\saved_from_python_code2.csv")
X_train, X_test, y_train, y_test = train_test_split(\
    X, y, test_size=0.2, random_state=seed, stratify=y)

svm = svm.SVC(kernel='linear', probability=True,
                     random_state=seed, verbose=False)
NB = GaussianNB();
tree = tree.DecisionTreeClassifier()
lrL2 = LogisticRegression(penalty='l2', verbose=0)
lrL1 = LogisticRegression(penalty='l1', verbose=0)

gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
                                 max_depth=1, random_state=seed)

smote = SMOTE(kind='svm')


bestModel, all_probas , all_preds, all_y_test = crossValidation(
    svm, X_train, y_train, smote)

test(bestModel, X_test, y_test)

plotAllCurves(all_probas, all_y_test)
