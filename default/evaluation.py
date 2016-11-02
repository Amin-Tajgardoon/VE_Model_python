'''
Created on Sep 28, 2016

@author: mot16
'''

import numpy as np

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.cross_validation import StratifiedKFold


def test(classifier, X_test, y_test):
    probas_ = classifier.predict_proba(X_test)
    y_pred = classifier.predict(X_test)
        
    # Compute performance metrics
    prec = precision_score(y_test, y_pred , pos_label=1, average='binary')
    recall = recall_score(y_test, y_pred , pos_label=1, average='binary')
    f1 = f1_score(y_test, y_pred , pos_label=1, average='binary')
    auRoc = roc_auc_score(y_test, probas_[:,1], average = 'weighted')
    auPrecRecall = average_precision_score(y_test, probas_[:,1], average='weighted')
    
    print("\nTest Results:\n")
    print("Precision: %.3f" % prec)
    print("Recall: %.3f" % recall)
    print("F1: %.3f" % f1)
    print("Weighted AUROC: %.3f" % auRoc)
    print("Weighted AU_PREC_RECALL: %.3f" % auPrecRecall)


def crossValidation(classifier, X, y, smote='None'):
    cv = StratifiedKFold(y, n_folds=10)
    
    allPrec = [];
    allRecall = [];
    allF1 = [];
    allAuRoc = [];
    allWeightedAuRoc = [];
    allAuPrecRecall = [];
    allWeightedAuPrecRecall = [];
    
    probas_ = [];
    y_pred = [];
    
    y_test = [];


    for i, (train, test) in enumerate(cv):
        # re-sample train data
        if smote == 'None':
            X_train = X.iloc[train]
            y_train = y.iloc[train]
        else:
            X_train, y_train = smote.fit_sample(X.iloc[train], y.iloc[train])
        
        # if smote, fit classifier on resampled-train and test on original test  
        model = classifier.fit(X_train, y_train)
        probas_.append(model.predict_proba(X.iloc[test]))
        y_pred.append(model.predict(X.iloc[test]))
        
        # Compute performance metrics
        allPrec.append(precision_score(y.iloc[test], y_pred[i] , pos_label=1, average='binary'))
        allRecall.append(recall_score(y.iloc[test], y_pred[i] , pos_label=1, average='binary'))
        allF1.append(f1_score(y.iloc[test], y_pred[i] , pos_label=1, average='binary'))
        allAuRoc.append(roc_auc_score(y.iloc[test], probas_[i][:,1], average = 'macro'))
        allWeightedAuRoc.append(roc_auc_score(y.iloc[test], probas_[i][:,1], average = 'weighted'))
        auPrecRecall = average_precision_score(y.iloc[test], probas_[i][:,1], average='macro')
        allAuPrecRecall.append(auPrecRecall)
        
        y_test.append(y.iloc[test])

        auPrecRecall = average_precision_score(y.iloc[test], probas_[i][:,1], average='weighted')
        allWeightedAuPrecRecall.append(auPrecRecall)
        if auPrecRecall >= np.max(allWeightedAuPrecRecall):
            bestModel = model
    
    print("\nCross-Validation results:\n")        
    print("Average Precision: %.3f" % np.mean(allPrec))
    print("Average Recall: %.3f" % np.mean(allRecall))
    print("Average F1: %.3f" % np.mean(allF1))
    print("Average AUROC: %.3f" % np.mean(allAuRoc))
    print("Average Weighted AUROC: %.3f" % np.mean(allWeightedAuRoc))
    print("Average AU_PREC_RECALL: %.3f" % np.mean(allAuPrecRecall))
    print("Average Weighted AU_PREC_RECALL: %.3f" % np.mean(allWeightedAuPrecRecall))
    
    return bestModel, probas_, y_pred, y_test
