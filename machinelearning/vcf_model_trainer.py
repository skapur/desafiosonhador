import pickle

from sklearn.ensemble.bagging import BaggingClassifier
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.ensemble.voting_classifier import VotingClassifier
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection._validation import cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn.svm.classes import SVC
from sklearn.tree.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline

from mlxtend.classifier import StackingClassifier
from mlxtend.feature_selection import ColumnSelector
import numpy as np


class VCFModelTrainer(object):
    
    def __init__(self, muctectIndex=None, strelkaIndex=None):
        if muctectIndex is not None:
            muctect_pipe = make_pipeline(ColumnSelector(
                cols=muctectIndex), 
                VotingClassifier(estimators=[('svm', SVC(probability = True)), 
                                             ('nnet', MLPClassifier()), 
                                             ('logisticRegression', LogisticRegression(C = 0.001))],
                                  voting='soft')
                )
        if strelkaIndex is not None:
            strelka_pipe = make_pipeline(ColumnSelector(
                cols=strelkaIndex), 
                VotingClassifier(estimators=[('svm', SVC(probability = True)), 
                                             ('nnet', MLPClassifier()), 
                                             ('logisticRegression', LogisticRegression(C = 0.001))],
                                  voting='soft')
                )
        self.__methods = { 
            "knn" : KNeighborsClassifier(7),
            "nbayes" : GaussianNB(),
            "decisionTree" : DecisionTreeClassifier(max_depth = 4, criterion = "gini", splitter = "random"),
            "logisticRegression" : LogisticRegression(C = 0.001),
            "svm" : SVC(probability = True),
            'nnet' : MLPClassifier(),
            'rand_forest' : RandomForestClassifier(max_depth = 5, criterion = "entropy", n_estimators = 100),
            'bagging': BaggingClassifier(max_samples = 1, bootstrap = True),
            'voting' : VotingClassifier(estimators=[('svm', SVC(probability = True)), ('nnet', MLPClassifier()), ('logisticRegression', LogisticRegression(C = 0.001))], voting='soft')
            }
        if muctectIndex is not None and strelkaIndex is not None:
            self.__methods['stacking'] = StackingClassifier(
                classifiers= [muctect_pipe, strelka_pipe],
                          use_probas=True,
                          average_probas=False,
                          meta_classifier=
                          VotingClassifier(estimators=[('svm', SVC(probability = True)), 
                                                       ('nnet', MLPClassifier()), 
                                                       ('logisticRegression', LogisticRegression())], 
                                           voting='soft'))
            
    #solver = 'newton-cg', C = 0.1, penalty = "l2", tol = 0.001, multi_class = 'multinomial'
    def df_reduce(self, X, y, inputer = None, variance = None, scaler = None, fts = None, filename = None):
        z = None
        if inputer is not None:
            inputer.fit(X,y)
            X = inputer.transform(X)
        if variance is not None:
            variance.fit(X, y)
            X = variance.transform(X)
        if scaler is not None:
            scaler.fit(X, y)
            X = scaler.transform(X)
        if fts is not None:
            fts.fit(X, y)
            X = fts.transform(X)
            z = fts.get_support(True)
        if filename is not None: # save the objects to disk
            f = open(filename, 'wb')
            pickle.dump({'inputer': inputer, 'variance':variance, 'scaler': scaler, 'fts': fts}, f)
            f.close()
        return X, y, z
    
    def testAllMethodsCrossValidation(self, X, y, folds=10):
        for method in self.__methods.keys():
            self.doCrossValidation(method, X, y, folds)
    
    def doCrossValidation(self, method, X, y, folds = 10):
        if method in self.__methods.keys():
            clf = self.__methods[method]
            name = clf.__class__.__name__
            mscore = ["accuracy", "precision", "recall", "f1", "roc_auc", "neg_log_loss"]
            scores = cross_validate(clf, X, y, cv = folds, scoring = mscore)
            self.__report_CV(name, scores)
        else:
            print("Model method not correct!")
    
    def __report_CV(self, name, scores):
        print("="*40)
        print(name)
        print('****Results****')
        for name, array in scores.items():
            print(str(name) + ": %0.4f (+/- %0.4f)" % (np.mean(array), np.std(array)))
        print("="*40)
        
    def trainModel(self, method, X, y):
        if method in self.__methods.keys():
            clf = self.__methods[method]
            clf.fit(X,y)
            return clf
        else:
            print("Model method not correct!")
            return None