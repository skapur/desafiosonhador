
import numpy as np
import pandas as pd
import pickle
import sys

from sklearn.feature_selection import SelectPercentile
from sklearn.preprocessing import MaxAbsScaler, QuantileTransformer, MinMaxScaler, StandardScaler, Normalizer, Imputer
from sklearn.pipeline import Pipeline

#Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, Lasso, SGDClassifier
from sklearn.svm import SVC,LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, VotingClassifier
from data_preprocessing import MMChallengeData, MMChallengePredictor
from genomic_data_test import df_reduce, cross_val_function, rnaseq_prepare_data

def report (cv_res):
    print("="*40)
    for name, array in cv_res.items():
        print(str(name) + ": %0.4f (+/- %0.4f)" % (np.mean(array), np.std(array)))
    print("="*40)

#Load data
data_folder = '/home/skapur/synapse/syn7222203/CH2'
clin_file = '/home/skapur/synapse/syn7222203/Clinical Data/sc2_Training_ClinAnnotations.csv'
mmcd = MMChallengeData(clin_file)
mmcd.generateDataDict(clinicalVariables=["D_Age", "D_ISS"], outputVariable="HR_FLAG", directoryFolder=data_folder, columnNames=None, NARemove=[True, False])

#Transformers
# scl = MaxAbsScaler()

from sklearn.base import clone as clone_sk

clin_norm = Pipeline(steps=[('impute', Imputer(strategy='median')),('normalizer', MaxAbsScaler())])
# =========================
#          RNA-SEQ
# =========================

RNA_gene, RNA_gene_cd, RNA_gene_output = mmcd.dataDict[("RNASeq", "gene")]

X_rseq, y_rseq = rnaseq_prepare_data(RNA_gene, None, RNA_gene_output)
X_clin_rseq = clin_norm.fit_transform(RNA_gene_cd.loc[X_rseq.index])
scl_rseq = Pipeline(steps=[('impute', Imputer(strategy='median')),('normalizer', Normalizer())])
fts_rseq = SelectPercentile(percentile = 100)
X_rseq_t, y_rseq_t, fts_vector = df_reduce(X_rseq, y_rseq, scl_rseq, fts_rseq, True, filename = 'transformers_rna_seq.sav')

# =============================
#          MICROARRAYS
# =============================

MA_gene, MA_gene_cd, MA_gene_output = mmcd.dataDict[("MA", "gene")]
scl_ma = Pipeline(steps=[('impute', Imputer(strategy='median')),('normalizer', Normalizer())])
fts_ma = SelectPercentile(percentile = 10)
X_marrays, y_marrays = rnaseq_prepare_data(MA_gene, None, MA_gene_output, axis = 1)
X_clin_ma = clin_norm.fit_transform(MA_gene_cd.loc[X_marrays.index])
X_marrays_t, y_marrays_t, fts_vector = df_reduce(X_marrays, y_marrays, scl_ma, fts_ma, True, filename = 'transformers_microarrays.sav')


# =============================
#          ENSEMBLE
# =============================

clf1 = LogisticRegression(random_state = 1, solver = 'newton-cg', C = 1, penalty = "l2", tol = 0.001, multi_class = 'multinomial')
clf2 = RandomForestClassifier(random_state = 1, max_depth = 5, criterion = "entropy", n_estimators = 100)
clf3 = GaussianNB()
clf4 = SVC(kernel = "linear", C = 0.5, probability = True, gamma = 0.0001)
clf5 = MLPClassifier(solver = 'adam', activation = "relu", hidden_layer_sizes = (50,25), alpha = 0.001)

clf_list = [('logreg',clf1),('nb',clf3),('svm',clf4),('mlp',clf5)]
eclf_ma = VotingClassifier(voting='soft', estimators=clf_list)
eclf_ma.fit(X_marrays_t, y_marrays_t)

with open('ma_voting_clf.sav','wb') as f:
    pickle.dump(eclf_ma, f)

# =============================
# RNA-Seq meta-classifier
# =============================
for clf_name, clf in clf_list:
    clf.fit(X_rseq_t, y_rseq_t)

with open('rna_fitted_classifier_list.sav','wb') as f:
    pickle.dump(clf_list, f)

scores_rna = np.stack([clf.predict_proba(X_rseq_t)[:,1] for clf_name, clf in clf_list]).T

meta_classifier_rna_seq = LogisticRegression(
    random_state = 1,
    solver = 'newton-cg',
    C = 0.5,
    penalty = "l2",
    tol = 0.001,
    multi_class = 'multinomial'
)

meta_classifier_rna_seq.fit(scores_rna, y_rseq_t)
cv_meta_rna = cross_val_function(scores_rna, y_rseq_t, meta_classifier_rna_seq)
print(report(cv_meta_rna))

with open('rna_fitted_stack_classifier.sav','wb') as f:
    pickle.dump(meta_classifier_rna_seq, f)

# =============================
# Microarray meta-classifier
# =============================
for clf_name, clf in clf_list:
    clf.fit(X_marrays_t, y_marrays_t)

with open('ma_fitted_classifier_list.sav', 'wb') as f:
    pickle.dump(clf_list, f)

scores_ma = np.stack([clf.predict_proba(X_marrays_t)[:, 1] for clf_name, clf in clf_list], axis=1)

meta_classifier_ma = LogisticRegression(
    random_state=1,
    solver='newton-cg',
    C=0.5,
    penalty="l2",
    tol=0.001,
    multi_class='multinomial'
)

meta_classifier_ma.fit(scores_ma, y_marrays_t)
cv_meta_ma = cross_val_function(scores_ma, y_marrays_t, meta_classifier_ma)
print(report(cv_meta_ma))

with open('ma_fitted_stack_classifier.sav', 'wb') as f:
    pickle.dump(meta_classifier_ma, f)

# eclf1 = VotingClassifier(estimators=[('lr', clf1), ('svm', clf4), ('nnet', clf5)],
#                          voting = 'soft', n_jobs = -1)
#eclf1 = eclf1.fit(X_rseq_t, y_rseq_t)

#RNA-Seq
cv = cross_val_function(X_rseq_t, y_rseq_t, clf = meta_classifier_ma)
report(cv)

#Fit Models
clf_rseq = eclf1
clf_rseq.fit(X_rseq_t, y_rseq_t)
with open("fittedModel_rna_seq.sav", 'wb') as f:
    pickle.dump(clf_rseq, f)

with open("fittedModel_logreg_ensemble_rna_seq.sav", 'rb') as f:
    m = pickle.load(f)
#Microarrays
cv = cross_val_function(X_marrays_t, y_marrays_t, clf = eclf1)
report(cv)

#Fit Models
clf_marrays = eclf1
clf_marrays.fit(X_marrays_t, y_marrays_t)
with open("fittedModel_microarrays.sav", 'wb') as f:
    pickle.dump(clf_marrays, f)

# clf = MLPClassifier(solver = 'adam', activation = "relu", hidden_layer_sizes = (50,25), alpha = 0.001)
clf = eclf1
cv = cross_val_function(X_rseq_t, y_rseq_t, clf = clf)
report(cv)


# =========================
#          PLOTS
# =========================

# import matplotlib as plt
# plt.use('Qt5Agg')
# import matplotlib.pyplot as plt
# #import PyQt5
# #import matplotlib.pyplot as plt
#
# clf1 = LogisticRegression(random_state = 1, solver = 'newton-cg', C = 1, penalty = "l2", tol = 0.001, multi_class = 'multinomial')
# clf2 = RandomForestClassifier(random_state = 1, max_depth = 5, criterion = "entropy", n_estimators = 100)
# clf3 = GaussianNB()
# eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting = 'soft', n_jobs = -1, weights = [1, 1, 5])
#
# # predict class probabilities for all classifiers
# probas = [c.fit(X_rseq_t, y_rseq_t).predict_proba(X_rseq_t) for c in (clf1, clf2, clf3, eclf)]
#
# # get class probabilities for the first sample in the dataset
# class1_1 = [pr[0, 0] for pr in probas]
# class2_1 = [pr[0, 1] for pr in probas]
#
# # plotting
# N = 4  # number of groups
# ind = np.arange(N)  # group positions
# width = 0.35  # bar width
#
# fig, ax = plt.subplots()
#
# # bars for classifier 1-3
# p1 = ax.bar(ind, np.hstack(([class1_1[:-1], [0]])), width, color='green', edgecolor='k')
# p2 = ax.bar(ind + width, np.hstack(([class2_1[:-1], [0]])), width, color='lightgreen', edgecolor='k')
#
# # bars for VotingClassifier
# p3 = ax.bar(ind, [0, 0, 0, class1_1[-1]], width, color='blue', edgecolor='k')
# p4 = ax.bar(ind + width, [0, 0, 0, class2_1[-1]], width, color='steelblue', edgecolor='k')
#
# # plot annotations
# plt.axvline(2.8, color='k', linestyle='dashed')
# ax.set_xticks(ind + width)
# ax.set_xticklabels(['LogisticRegression\nweight 1',
#                     'GaussianNB\nweight 1',
#                     'RandomForestClassifier\nweight 5',
#                     'VotingClassifier\n(average probabilities)'],
#                    rotation=40,
#                    ha='right')
# plt.ylim([0, 1])
# plt.title('Class probabilities for sample 1 by different classifiers')
# plt.legend([p1[0], p2[0]], ['class 1', 'class 2'], loc='upper left')
# plt.show()














