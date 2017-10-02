import numpy as np
import pandas as pd
import pickle
import sys

from sklearn import decomposition
from sklearn.feature_selection import SelectPercentile
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

#Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier

from data_preprocessing import MMChallengeData, MMChallengePredictor

def cross_val_function(X, y, clf):
    print("Cross validating "+type(clf).__name__)
    return cross_validate(pipeline_factory(clf), X, y, scoring=["accuracy", "recall", "f1", "neg_log_loss", "precision", "roc_auc"], cv=10, verbose=1)


from numpy import where
report = lambda cvr : '\n'.join([str(name + " : " + str(np.mean(array)) + " +/- " + str(np.std(array))) for name, array in cvr.items()])


def rnaseq_prepare_data(X_gene, X_trans, y_orig, axis = 0):
    if X_trans is not None:
        X = pd.concat([X_gene, X_trans], axis=1).dropna(axis = axis)
    else:
        X = X_gene.dropna(axis=axis)
    y = y_orig[X.index]
    valid_samples = y != "CENSORED"
    X, y = X[valid_samples], y[valid_samples] == "TRUE"
    y = y.astype(int)
    return X, y



def df_reduce(X, y, scaler = None, fts = None, fit = True, filename = None):
    import pickle
    if fit:
        scaler.fit(X, y); X = scaler.transform(X)
        fts.fit(X, y); X = fts.transform(X)
        if filename is not None: # save the objects to disk
            f = open(filename, 'wb')
            pickle.dump({'scaler': scaler, 'fts': fts}, f)
            f.close()
    elif not fit and filename is None:
        X = scaler.transform(X);
        X = fts.transform(X)
    else:
        try: # load the objects from disk
            f = open(filename, 'rb')
            dic = pickle.load(f)
            scaler = dic['scaler']; fts = dic['fts']
            f.close()
            X = scaler.transform(X); X = fts.transform(X)
        except Exception as e:
            print ("Unexpected error:", sys.exc_info())
            #raise e
    return X, y, fts.get_support(True)


def cross_validate_models(X, y, model_dict, param_dict):
    return {name:cross_val_function(X, y, clf) for name, clf in model_dict.items()}

def train_models(X, y, filename, model_list = None):
    fittedModels = {}
    if model_list is None: #Fit all models
        for model in models.keys():
            estimator = models[str(model)]
            estimator.fit(X, y)
            fittedModels[str(model)] = estimator
    else: #Fit selected
        for model in model_list:
            estimator = models[str(model)]
            estimator.fit(X, y)
            fittedModels[str(model)] = estimator

    # save fitted models to disk
    f = open(filename, 'wb')
    pickle.dump(fittedModels, f)
    f.close()

def crossValEnseble (X, y, cv = 5, clf = DecisionTreeClassifier(), n = 50):
    estimator = AdaBoostClassifier(base_estimator = clf, n_estimators = n)
    name = estimator.__class__.__name__
    mscore = ['accuracy', 'f1', 'neg_log_loss', 'recall']
    scores = cross_validate(estimator, X, y, cv = cv, scoring = mscore)
    print("="*30)
    print(name, 'with', clf.__class__.__name__, '(n = ' + str(n) + ')')
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



if __name__ == '__main__':

    #Correr o script:
    # python -W ignore .\ch2_script.py "D:\Dropbox\workspace_intelij\DREAM_Challenge" "D:\Dropbox\workspace_intelij\DREAM_Challenge\predictions.tsv"


    #DIR = "/home/skapur/synapse/syn7222203/"
    DIR = 'D:\Dropbox\workspace_intelij\DREAM_Challenge'
    mmcd = MMChallengeData(DIR)
    mmcd.generateDataDict()

    MA_gene, MA_gene_cd, MA_gene_output = mmcd.dataDict[("MA", "gene")]
    MA_probe, MA_probe_cd, MA_probe_output = mmcd.dataDict[("MA", "probe")]

    RNA_gene, RNA_gene_cd, RNA_gene_output = mmcd.dataDict[("RNASeq", "gene")]
    RNA_trans, RNA_trans_cd, RNA_trans_output = mmcd.dataDict[("RNASeq", "trans")]


    params = {'knn': {'n_neighbors': 4}, 'decisionTree': {'criterion': 'gini', 'max_depth': 4, 'splitter': 'random'},
              'logReg': {'solver': 'newton-cg', 'C': 1, 'penalty': "l2", 'tol': 0.001, 'multi_class': 'multinomial'},
              'svm': {'kernel': 'linear', 'C': 1, 'probability': True, 'gamma': 0.0001},
              'bagging': {'max_samples': 1, 'bootstrap': True},
              'nnet': {'solver': 'lbfgs', 'activation': "logistic", 'hidden_layer_sizes': (90,), 'alpha': 0.9},
              'randForest': {'max_depth': 5, 'criterion': 'entropy', 'n_estimators': 100}}

    models = {'knn': KNeighborsClassifier(**params['knn']), 'nbayes': GaussianNB(), 'decisionTree': DecisionTreeClassifier(**params['decisionTree']),
              'logReg': LogisticRegression(**params['logReg']), 'svm': SVC(**params['svm']), 'bagging': BaggingClassifier(**params['bagging']),
              'nnet': MLPClassifier(**params['nnet']), 'randForest': RandomForestClassifier(**params['randForest'])}


    pipeline_factory = lambda clf: Pipeline(steps=[("classify", clf)])

    scl = MaxAbsScaler()
    fts = SelectPercentile(percentile=30)

    # =====================
    #   RNA Seq Data Test
    # =====================

    #Create files to use in ch2_script.py
    X_rseq, y_rseq = rnaseq_prepare_data(RNA_gene, RNA_trans, RNA_gene_output)
    X_rseq_t, y_rseq_t, fts_vector = df_reduce(X_rseq, y_rseq, scl, fts, True, filename = 'transformers_rna_seq.sav')
    # cv_rseq = cross_validate_models(X_rseq_t, y_rseq_t, models, params)

    #Fit NBayes
    clf_rseq = GaussianNB()
    clf_rseq.fit(X_rseq_t, y_rseq_t)
    with open("fittedModel_rna_seq.sav", 'wb') as f:
        pickle.dump(clf_rseq, f)

    #train_models(X_rseq_t, y_rseq_t, filename = 'fittedModels_rna_seq.sav') # Train Models


    # =========================
    #   Microarrays Data Test      = Participants may also output the number/percent of genes in common across expression data sets (challenge questions 2 and 3) and the number/percent of mutations (or genes) in common across WES data sets (challenge questions 1 and 3).  This may again be done on a per data set basis and/or across all data sets.  However, in no case should this information be linked to particular samples.
    # =========================

    X_marrays, y_marrays = rnaseq_prepare_data(MA_gene, None, MA_gene_output, axis = 1)
    X_marrays_t, y_marrays_t, fts_vector = df_reduce(X_marrays, y_marrays, scl, fts, True, filename = 'transformers_microarrays.sav')
    #cv_marrays = cross_validate_models(X_marrays_t, y_marrays_t, models, params)

    clf_marrays = GaussianNB()
    clf_marrays.fit(X_marrays_t, y_marrays_t)
    with open("fittedModel_microarrays.sav", 'wb') as f:
        pickle.dump(clf_marrays, f)





    # =======================================================
    #                      ENSEMBLE Test
    # =======================================================

    # http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier
    from sklearn.ensemble import AdaBoostClassifier

    data_folder = '.\Expression Data\All Data'
    clin_file = '.\Clinical Data\sc2_Training_ClinAnnotations.csv'
    mmcd = MMChallengeData(clin_file)

    mmcd.generateDataDict(clinicalVariables=["D_Age", "D_ISS"], outputVariable="HR_FLAG", directoryFolder=data_folder, columnNames=None, NARemove=[True, False])

    #Transformers
    scl = MaxAbsScaler()
    fts = SelectPercentile(percentile=30)

    # =========================
    #          RNA-SEQ
    # =========================

    #Prepare data
    Xrseq, cdrseq, yrseq = mmcd.dataDict[("RNASeq","gene")]
    Xnrseq = Xrseq.dropna(axis = 1)
    yrseq = yrseq == "TRUE"
    yrseq = yrseq.astype(int)

    X_rseq_t, y_rseq_t, fts_vector = df_reduce(Xnrseq, yrseq, scl, fts, True, filename = 'transformers_rna_seq.sav')

    #Cross-validation test
    crossValEnseble(X_rseq_t, y_rseq_t, clf = SVC(probability = True, kernel = 'linear'), n = 1)
    crossValEnseble(X_rseq_t, y_rseq_t, clf = GaussianNB(), n = 50)
    crossValEnseble(X_rseq_t, y_rseq_t, clf = LogisticRegression(), n = 50)


    clf_rseq = AdaBoostClassifier(base_estimator = LogisticRegression(), n_estimators = 50)
    clf_rseq.fit(X_rseq_t, y_rseq_t)
    with open("fittedModel_logreg_ensemble_rna_seq.sav", 'wb') as f:
        pickle.dump(clf_rseq, f)

    # with open('transformers_rna_seq.sav', 'rb') as f:
    #     trf_rseq = pickle.load(f)
    #
    # with open('fittedModel_rna_seq.sav', 'rb') as f:
    #     clf_rseq = pickle.load(f)

    mv_fun_rseq = lambda x: df_reduce(x.values.reshape(1,-1), [], fit = False, scaler = trf_rseq['scaler'], fts = trf_rseq['fts'])[0]

    # Make predictions
    mod_rseq = MMChallengePredictor(
        mmcdata = mmcd,
        predict_fun = lambda x: clf_rseq.predict(x)[0],
        # confidence_fun = lambda x: 1 - min(clf_rseq.predict_proba(x)[0]),
        confidence_fun = lambda x: clf_rseq.predict_proba(x)[0][1],
        data_types = [("RNASeq", "gene"), ("RNASeq", "trans")],
        single_vector_apply_fun = lambda x: x,
        multiple_vector_apply_fun = mv_fun_rseq
    )
    res_rseq = mod_rseq.predict_dataset()

    # =============================
    #          MICROARRAYS
    # =============================

    Xma, cdma, yma = mmcd.dataDict[("MA","gene")]
    Xnma = Xma.dropna(axis = 1)
    yma = yma == "TRUE"
    yma = yma.astype(int)

    X_ma_t, y_ma_t, fts_vector = df_reduce(Xnma, yma, scl, fts, True, filename = 'transformers_microarrays.sav')

    #Cross-validation test
    crossValEnseble(X_ma_t, y_ma_t, clf = LogisticRegression(), n = 50)

    clf_marrays = AdaBoostClassifier(base_estimator = LogisticRegression(), n_estimators = 50)
    clf_marrays.fit(X_ma_t, y_ma_t)
    with open("fittedModel_logreg_ensemble_microarrays.sav", 'wb') as f:
        pickle.dump(clf_marrays, f)


    # with open('transformers_microarrays.sav', 'rb') as f:
    #     trf_marrays = pickle.load(f)
    #
    # with open('fittedModel_microarrays.sav', 'rb') as f:
    #     clf_marrays = pickle.load(f)

    mv_fun = lambda x: df_reduce(x.values.reshape(1,-1), [], scaler = trf_marrays['scaler'], fts = trf_marrays['fts'], fit = False)[0]

    # Make predictions
    mod_marryas = MMChallengePredictor(
        mmcdata = mmcd,
        predict_fun = lambda x: clf_marrays.predict(x)[0],
        confidence_fun = lambda x: clf_marrays.predict_proba(x)[0][1],
        data_types = [("MA", "gene")],
        single_vector_apply_fun = mv_fun,
        multiple_vector_apply_fun = lambda x: x
    )
    res_marrays = mod_marryas.predict_dataset()

    # =========================
    #       Final Dataset
    # =========================

    final_res = res_rseq.combine_first(res_marrays)
    final_res.columns = ['study', 'patient', 'predictionscore', 'highriskflag']
    final_res['highriskflag'] = final_res['highriskflag'] == 1
    final_res['highriskflag'] = final_res['highriskflag'].apply(lambda x: str(x).upper())

    final_res["predictionscore"] = final_res["predictionscore"].fillna(value=0.5)
    #final_res["highriskflag"] = final_res["highriskflag"].fillna(value=True)

    final_res.to_csv("predictions.tsv", index = False, sep = '\t')

    print(str(final_res.isnull().any()))
    print(str(final_res.isnull().all()))










# ### EXPERIMENTAL
    #
    # scl.transform(vec)
    # for scaler in ppscalers:
    #     print("*" * 80)
    #     print(scaler.__name__)
    #     cv = cross_val_function(*microarray_prepare_data(MA_probe, MA_probe, MA_gene_output, scaler))
    #     report(cv)
    #
    # report(cross_val_function(*microarray_prepare_data(MA_gene, MA_probe, MA_gene_output)))
    #
    # cv_m = cross_validate(pipeline_factory(GaussianNB()), Xr, yr,
    #                       scoring=["accuracy", "recall", "f1", "neg_log_loss", "precision", "roc_auc"], cv=10)
    #
    # import matplotlib.pyplot as plt
    # from numpy import where
    #
    # pca = decomposition.PCA(n_components=100)
    # X_pca = pca.fit_transform(X, y)
    #
    # sp = SelectPercentile(percentile=30)
    # sp.fit(X, y)
    # sig = where(sp.pvalues_ < 0.01)
    #
    # len(sig[0])
    #
    # X_sig = X[:, sig[0]]
    # X_sig.shape
    #
    #
    # def plot_pca(pca, colors, y_ids, y_labels, plot_title, plotkwargs, legendkwargs={}):
    #     fig = plt.figure()
    #     for color, i, target_name in zip(cokm.
    #                                              lors, y_ids, y_labels):
    #         plot_args = plotkwargs(color=color, target_name=target_name)
    #         plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], **plot_args)
    #     plt.legend(**legendkwargs)
    #     plt.title(plot_title)
    #     return fig
    #
    #
    # pca = decomposition.PCA(n_components=50)
    # pca.explained_variance_ratio_
    # X_pca = pca.fit_transform(X_sig, y)
    #
    # plot_args = lambda color, target_name: dict(color=color, alpha=.8, lw=2, label=target_name)
    # legend_args = dict(loc='best', shadow=False, scatterpoints=1)
    #
    # plot_pca(pca, ['red', 'green'], [0, 1], ['False', 'True'], "PCA", legendkwargs=legend_args, plotkwargs=plot_args)
    #
    # from sklearn.cluster.k_means_ import KMeans
    # from scipy.stats import rankdata
    # from mpl_toolkits.mplot3d import Axes3D
    # from numpy import concatenate, array
    #
    # km = KMeans(n_clusters=2)
    # km.fit(X_sig, y)
    # y_pred = km.predict(X_sig)
    #
    # rank = rankdata(sp.pvalues_, "ordinal")
    # colors = ["red", "green", "blue", "yellow", "pink", "black"]
    # fig = plt.figure()
    # ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    # ax.scatter(X[:, rank[0]], X[:, rank[1]], X[:, rank[2]], c=y_pred)
    # plt.title("Incorrect Number of Blobs")
    #
    # features = concatenate((X[:, rank[:4]], array(y_pred).reshape((428, 1)), array(y).reshape((428, 1))), axis=1)
    #
    # df = pd.DataFrame(features[:, [4, 5]]).groupby(0).hist()
    #







