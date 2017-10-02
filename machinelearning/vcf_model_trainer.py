from sklearn.ensemble.bagging import BaggingClassifier
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection._validation import cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn.svm.classes import SVC
from sklearn.tree.tree import DecisionTreeClassifier

import numpy as np
import pickle

class VCFModelTrainer(object):
    
    def __init__(self):
        self.__methods = { 
            "knn" : KNeighborsClassifier(7),
            "nbayes" : GaussianNB(),
            "decisionTree" : DecisionTreeClassifier(max_depth = 4, criterion = "gini", splitter = "random"),
            "logisticRegression" : LogisticRegression(solver = 'newton-cg', C = 1, penalty = "l2", tol = 0.001, multi_class = 'multinomial'),
            "svm" : SVC(kernel = "linear", C = 1, probability = True, gamma = 0.0001),
            'nnet' : MLPClassifier(solver = 'lbfgs', activation = "logistic", hidden_layer_sizes = (250, 125, 75, 25), alpha = 0.001),
            'rand_forest' : RandomForestClassifier(max_depth = 5, criterion = "entropy", n_estimators = 100),
            'bagging': BaggingClassifier(max_samples = 1, bootstrap = True)
            }
    
    def df_reduce(self, X, y, scaler = None, fts = None, filename = None):
        scaler.fit(X, y); X = scaler.transform(X) 
        fts.fit(X, y); X = fts.transform(X)
        if filename is not None: # save the objects to disk
            f = open(filename, 'wb')
            pickle.dump({'scaler': scaler, 'fts': fts}, f)
            f.close()
        return X, y, fts.get_support(True)
    
    def testAllMethodsCrossValidation(self, X, y, folds=10):
        for method in self.__methods.keys():
            self.doCrossValidation(method, X, y, folds)
    
    def doCrossValidation(self, method, X, y, folds = 10):
        if method in self.__methods.keys():
            clf = self.__methods[method]
            name = clf.__class__.__name__
            mscore = ['accuracy', 'f1', 'neg_log_loss', 'recall']
            scores = cross_validate(clf, X, y, cv = folds, scoring = mscore)
            print("="*30)
            print(name)
            print('****Results****')
            print('Fit time: %0.4f (+/- %0.4f)' % (np.mean(scores['fit_time']), np.std(scores['fit_time'])))
            print('Score time: %0.4f (+/- %0.4f)' % (np.mean(scores['score_time']), np.std(scores['score_time'])))
            print('Test accuracy: %0.4f (+/- %0.4f)' % (np.mean(scores['test_accuracy']), np.std(scores['test_accuracy'])))
            print('Train accuracy: %0.4f (+/- %0.4f)' % (np.mean(scores['train_accuracy']), np.std(scores['train_accuracy'])))
            print('Test F-score: %0.4f (+/- %0.4f)' % (np.mean(scores['test_f1']), np.std(scores['test_f1'])))
            print('Train F-score: %0.4f (+/- %0.4f)' % (np.mean(scores['train_f1']), np.std(scores['train_f1'])))
            print('Test log-loss: %0.4f (+/- %0.4f)' % (-np.mean(scores['test_neg_log_loss']), np.std(scores['test_neg_log_loss'])))
            print('Train log-loss: %0.4f (+/- %0.4f)' % (-np.mean(scores['train_neg_log_loss']), np.std(scores['train_neg_log_loss'])))
            print('Test recall: %0.4f (+/- %0.4f)' % (np.mean(scores['test_recall']), np.std(scores['test_recall'])))
            print('Train recall: %0.4f (+/- %0.4f)' % (np.mean(scores['train_recall']), np.std(scores['train_recall'])))
            print("="*30)
        else:
            print("Model method not correct!")
        
    def trainModel(self, method, X, y):
        if method in self.__methods.keys():
            clf = self.__methods[method]
            clf.fit(X,y)
            return clf
        else:
            print("Model method not correct!")
            return None