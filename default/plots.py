'''
Created on Sep 27, 2016

@author: mot16
'''

import numpy as np
from scipy import interp
import matplotlib.pyplot as plt

from sklearn.metrics import auc, precision_recall_curve, roc_curve
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve
import seaborn as sns

def plotFreq(X, y):
    data = X
    data['class'] = y
    i = 1;
    for colname in data.columns:
        plt.figure(i)
        i += 1
        sns.countplot(x=colname, hue='class', data=data)
    
    plt.show()


def plotAllCurves(all_probas, all_y_test):
    plotRoc(all_probas, all_y_test)
    plotPRC(all_probas, all_y_test)
    plotCalibration(all_probas, all_y_test)


def plotRoc(probas, y_test):
# Classification and ROC analysis
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    
    plt.figure()
    
    for i in range(0,len(probas)):

        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y_test[i], probas[i][:, 1], pos_label=1)
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, "--o", label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

    mean_tpr /= len(probas)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--',
             label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

def plotPRC(probas, y_test):
# Classification and ROC analysis
    mean_pr = 0.0
    mean_rc = np.linspace(0, 1, 100)

    plt.figure()

    for i in range(0,len(probas)):

        # Compute ROC curve and area the curve
        pr, rc, thresholds = precision_recall_curve(y_test[i], probas[i][:, 1], pos_label=1)
        #print('fold:%i thresholds ' % i)
        #print(thresholds)
        #print('max threshold = %0.6f in fold %d' % (np.mean(thresholds), i))
        mean_pr += interp(mean_rc, rc, pr)
        mean_pr[0] = 0.0
        prc_auc = auc(rc, pr)
        plt.plot(rc, pr, "--o", label='PRC fold %d (area = %0.2f)' % (i, prc_auc))

    plt.plot([1, 0], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

    mean_pr /= len(probas)
    mean_pr[-1] = 1.0
    mean_auc = auc(mean_rc, mean_pr)
    plt.plot(mean_rc, mean_pr, 'k--',
             label='Mean PRC (area = %0.2f)' % mean_auc, lw=2)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall')
    plt.legend(loc="lower left")
    plt.show()

def plotCalibration(probas, y_test):
    
    plt.figure()

    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        
    mean_frpos = 0.0
    mean_pv = np.linspace(0, 1, 100)
    
    for i in range(0,len(probas)):
        prob_pos = probas[i][:, 1]
        clf_score = brier_score_loss(y_test[i], prob_pos, pos_label=1)
        #print("\tBrier: %1.3f" % (clf_score))
        fraction_of_positives, mean_predicted_value = \
                calibration_curve(y_test[i], prob_pos, n_bins=10)
            
        mean_frpos += interp(mean_pv, mean_predicted_value, fraction_of_positives)
        mean_frpos[0] = 0.0
        
        #print( (fraction_of_positives,mean_predicted_value ))
    
        ax1.plot(mean_predicted_value, fraction_of_positives, "--o", label="fold%i (%1.3f)" % (i,clf_score))
    
        ax2.hist(prob_pos, range=(0, 1), bins=10, label='fold%i' % i,
                     histtype="step", lw=2)
    
    mean_frpos /= len(probas)
    mean_frpos[-1] = 1.0
    ax1.plot(mean_pv, mean_frpos, 'k--',
             label='Mean Calibration curve', lw=2)
    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')
    
    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)
    
    plt.tight_layout()
    plt.show()
       
        
def rocCurveWithSmote(classifier, X, y, smote):
# Classification and ROC analysis
    cv = StratifiedKFold(y, n_folds=10)
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    for i, (train, test) in enumerate(cv):
        # re-sample train data
        X_resampled, y_resampled = smote.fit_sample(X.iloc[train], y[train])
        
        # fit classifier on resampled-train, test on original test  
        probas_ = classifier.fit(X_resampled, y_resampled).predict_proba(X.iloc[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1], pos_label=1)
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, "--o", label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

    mean_tpr /= len(cv)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--',
             label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


def prcCurveWithSmote(classifier,X, y, smote):
    cv = StratifiedKFold(y, n_folds=10)
    mean_pr = 0.0
    mean_rc = np.linspace(0, 1, 100)
 
    for i, (train, test) in enumerate(cv):
        # re-sample train data
        X_resampled, y_resampled = smote.fit_sample(X.iloc[train], y[train])
        # fit classifier on resampled-train, test on original test  
        probas_ = classifier.fit(X_resampled, y_resampled).predict_proba(X.iloc[test])
        # Compute PR curve and area the curve
        pr, rc, thresholds = precision_recall_curve(y[test], probas_[:, 1], pos_label=1)
        #print('fold:%i thresholds ' % i)
        #print(thresholds)
        #print('max threshold = %0.6f in fold %d' % (np.mean(thresholds), i))
        mean_pr += interp(mean_rc, rc, pr)
        mean_pr[0] = 0.0
        prc_auc = auc(rc, pr)
        plt.plot(rc, pr, "--o", label='PRC fold %d (area = %0.2f)' % (i, prc_auc))

    plt.plot([1, 0], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

    mean_pr /= len(cv)
    mean_pr[-1] = 1.0
    mean_auc = auc(mean_rc, mean_pr)
    plt.plot(mean_rc, mean_pr, 'k--',
             label='Mean PRC (area = %0.2f)' % mean_auc, lw=2)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall')
    plt.legend(loc="lower left")
    plt.show()
    
def calibCurveWithSmote(classifier,X, y, smote):
    plt.figure(2, figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    
    cv = StratifiedKFold(y, n_folds=10)
    
    mean_frpos = 0.0
    mean_pv = np.linspace(0, 1, 100)
    
    for i, (train, test) in enumerate(cv):
        probas_ = classifier.fit(X.iloc[train], y[train]).predict_proba(X.iloc[test])
        prob_pos = probas_[:, 1]
        clf_score = brier_score_loss(y[test], prob_pos, pos_label=y.max())
        #print("\tBrier: %1.3f" % (clf_score))
        fraction_of_positives, mean_predicted_value = \
                calibration_curve(y[test], prob_pos, n_bins=10)
            
        mean_frpos += interp(mean_pv, mean_predicted_value, fraction_of_positives)
        mean_frpos[0] = 0.0
        
        #print( (fraction_of_positives,mean_predicted_value ))
    
        ax1.plot(mean_predicted_value, fraction_of_positives, "--o", label="fold%i (%1.3f)" % (i,clf_score))
    
        ax2.hist(prob_pos, range=(0, 1), bins=10, label='fold%i' % i,
                     histtype="step", lw=2)
    
    mean_frpos /= len(cv)
    mean_frpos[-1] = 1.0
    ax1.plot(mean_pv, mean_frpos, 'k--',
             label='Mean Calibration curve', lw=2)
    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')
    
    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)
    
    plt.tight_layout()
    plt.show()
