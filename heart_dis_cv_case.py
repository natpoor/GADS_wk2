#!/usr/bin/env python

# Directly adopted from Jason's code, for GADS. (ie, starting with that code and modifying it)
# That is, I didn't write most of it, but I rewrote some of it and var names especially.

# IMPORT LIBRARIES
import numpy   as np
import pandas  as pd
import pylab   as pl      # note this is part of matplotlib

from sklearn import cross_validation as cv
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import roc_curve, auc


# VARIABLES
# TRAIN_PCT = 0.7
number_of_folds = 10
train_file_name = 'logit_train.csv'
#test_file_name = 'logit_test.csv'



# FUNCTIONS

def roc_it():
    train_file = pd.read_csv(train_file_name, delimiter=',').dropna()
    train_file.rename(columns = {'heartdisease::category|0|1' : 'has_it'}, inplace = True)

    # Keep only wanted columns I think.
    features = train_file.iloc[:, 0:12]  # drops "has_it" from features.
    has_it = train_file['has_it']        # maybe not the best object name since is a field name too.

    # create cv iterator (note: train pct is set implicitly by number of folds)
    num_recs = len(train_file)
    the_folds = cv.KFold(n = num_recs, n_folds = number_of_folds, shuffle = True)

    # initialize results sets -- less pythonic right now, ok.
    fp_rates = np.zeros(number_of_folds) # numpy lists, len = # folds.
    tp_rates = np.zeros(number_of_folds)
    all_aucs = np.zeros(number_of_folds)
    
    
    for i, (train_index, test_index) in enumerate(the_folds):  
    # The_folds is thus a sequence, so has indexes. Note the leading i in the loop for the indexes!
    # train_index and _test_index are values not indices???
        # initialize & train model
        model = LR() # model is a logistic regression object from sklearn

        # Debug! In case more NAs snuck in there.
        train_features = features.loc[train_index].dropna()
        train_labels = has_it.loc[train_index].dropna()
        test_features = features.loc[test_index].dropna()
        test_labels = has_it.loc[test_index].dropna()

        model.fit(train_features, train_labels) # TRAIN the MODEL!!!! with those two related things.

        # predict labels for test features
        pred_labels = model.predict(test_features) # MODEL AGAIN!

        # calculate ROC/AUC
        fpr, tpr, thresholds = roc_curve(test_labels, pred_labels, pos_label=1) # from sklearn.metrics
            # That is running actual test labels v. predicted labels for the same (test part) data!
            # What is thresholds? No the online docs don't help!
        roc_auc = auc(fpr, tpr) # from sklearn.metrics

        print '\n'
        print 'Iteration number: ', i + 1
        print 'fpr = {0}'.format(fpr)
        print 'tpr = {0}'.format(tpr)
        print 'auc = {0}'.format(roc_auc)
        
        print "This model's percentage mislabeld/misclassified: ", 'no idea', '%'
                
        fp_rates[i] = fpr[1]
        tp_rates[i] = tpr[1]
        all_aucs[i] = roc_auc

    print '\n'
    print 'Summaries:'
    print 'fp_rates = {0}'.format(fp_rates)
    print 'tp_rates = {0}'.format(tp_rates)
    print 'all_aucs = {0}'.format(all_aucs)
    
    print '\n'
    print 'Mean AUC is: ', sum(all_aucs) / float(len(all_aucs))

    # plot ROC curve -- crazy Jason code for an interactive graphic window!!!
#    pl.clf()
#    pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
#    pl.plot([0, 1], [0, 1], 'k--')
#    pl.xlim([0.0, 1.0])
#    pl.ylim([0.0, 1.0])
#    pl.xlabel('False Positive Rate')
#    pl.ylabel('True Positive Rate')
#    pl.title('Receiver operating characteristic example')
#    pl.legend(loc="lower right")
#    pl.show() 

# End of roc-it



# MAIN
if __name__ == '__main__':

    roc_it()

# End of Main






