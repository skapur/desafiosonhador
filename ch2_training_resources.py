from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.feature_selection import SelectPercentile, VarianceThreshold
from sklearn.preprocessing import Normalizer, MaxAbsScaler, RobustScaler, MinMaxScaler

import numpy as np
import pandas as pd
import sys
import pickle

classifier_dict = {
    #DecisionTreeClassifier: {'criterion': ['gini'], 'max_depth': [4], 'splitter': ['random']},
    #LogisticRegression: {'solver': ['newton-cg'], 'C': [1], 'penalty': ["l2"], 'tol': [0.001], 'multi_class': ['multinomial']},
    SVC: {'kernel': ['linear'], 'C': [0.5], 'probability': [True], 'gamma': [0.0001]},
    #BaggingClassifier: {'max_samples': [1], 'bootstrap': [True]},
    RandomForestClassifier: {'max_depth': [5], 'criterion': ['entropy'], 'n_estimators': [20,500]}
}

feature_selection_base_steps = [
    #('normalize', Normalizer()),
    #('scaler', MaxAbsScaler()),
    #('variance', VarianceThreshold()),
    ('percentile', SelectPercentile(percentile=10))
]

pca_step = ('pca', PCA(n_components=300))

base_param_grid = {
'percentile__percentile':[1]
#'variance__threshold':[0.0]
}

def cross_val_function(X, y, clf, cv=StratifiedKFold(n_splits=10),
                       metrics=["accuracy", "recall", "f1", "neg_log_loss", "precision", "roc_auc"], **kwargs):
    print("Cross validating " + type(clf).__name__)
    #pipeline_factory = lambda clf: Pipeline(steps=[("classify", clf)])
    return cross_validate(clf, X, y, scoring=metrics, cv=cv, **kwargs)


report = lambda cvr: ' \n '.join(
    [str(name + " : " + str(np.mean(array)) + " +/- " + str(np.std(array))) for name, array in cvr.items()])


# def rnaseq_prepare_data(X_gene, X_trans, y_orig, axis = 0):
#     if X_trans is not None:
#         X = pd.concat([X_gene, X_trans], axis=1).dropna(axis = axis)
#     else:
#         X = X_gene.dropna(axis=axis)
#     y = y_orig[X.index]
#     valid_samples = y != "CENSORED"
#     X, y = X[valid_samples], y[valid_samples] == "TRUE"
#     y = y.astype(int)
#     return X, y

# def df_reduce(X, y, scaler = None, fts = None, fit = True, filename = None):
#
#     if fit:
#         scaler.fit(X, y); X = scaler.transform(X)
#         fts.fit(X, y); X = fts.transform(X)
#         if filename is not None: # save the objects to disk
#             f = open(filename, 'wb')
#             pickle.dump({'scaler': scaler, 'fts': fts}, f)
#             f.close()
#     elif not fit and filename is None:
#         X = scaler.transform(X);
#         X = fts.transform(X)
#     else:
#         try: # load the objects from disk
#             f = open(filename, 'rb')
#             dic = pickle.load(f)
#             scaler = dic['scaler']; fts = dic['fts']
#             f.close()
#             X = scaler.transform(X); X = fts.transform(X)
#         except Exception as e:
#             print ("Unexpected error:", sys.exc_info())
#             #raise e
#     return X, y, fts.get_support(True)

def merge_data_types(frames, na_axis_drop=0):
    X = pd.concat(frames, axis=1).dropna(axis=na_axis_drop)


def preprocess_data(X_orig, y_orig, axis=0):
    if isinstance(X_orig, list):
        X = pd.concat(X_orig, axis=1).dropna(axis=axis)
    else:
        X = X_orig.dropna(axis=axis)
    y = y_orig[X_orig.index]
    valid_samples = y != "CENSORED"
    X, y = X_orig[valid_samples], y[valid_samples] == "TRUE"
    y = y.astype(int)
    return X, y

def save_as_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def read_pickle(path):
    with open(path, 'rb') as f:
        pickle.load(f)

def cross_validate_models(X, y, model_dict, param_dict):
    return {name: cross_val_function(X, y, clf) for name, clf in model_dict.items()}


def apply_fx_by_study(df, transform_fx, keep_study_col=False, study_col_id = 'Study'):
    dd = df.copy()
    # dd[study_col_id] = [ind[:1] for ind in dd.index.tolist()]
    # dd[study_col_id] = dd[study_col_id].apply(ord).astype(int)
    dd_by_study = dd.groupby(study_col_id)
    transformed = dd_by_study.apply(lambda x: np.append(transform_fx(x.drop(study_col_id, axis=1)),x[study_col_id].values.reshape(x.shape[0],1), axis=1))
    #print(transformed.index.unique())
    for idx in [i for i in transformed.index.unique()]:
        dd.loc[dd[study_col_id] == idx, :] = transformed[idx]

    if not keep_study_col:
        del dd[study_col_id]
    return dd

def df_reduce(X, y, transformer, fit=True):
    if fit:
        transformer.fit(X, y)
    return transformer.transform(X)

def read_from_data_dict(combination, data_dict):
    clin_cols = []
    data_cols = []

    if combination in data_dict.keys():
        d_file, d_clin, y = data_dict[combination]
        data_cols = d_file.columns
        clin_cols = d_clin.columns
        data = preprocess_data(pd.concat(objs=[d_file, d_clin], axis = 1), y)
    elif isinstance(combination, list):
        frames_to_merge = []
        for indcomb in combination:
            d_file, d_clin, y = data_dict[indcomb]
            frames_to_merge.append(d_file)
            frames_to_merge.append(d_clin)
            data_cols = data_cols + [col for col in d_file.columns if col not in data_cols]
            clin_cols = clin_cols + [col for col in d_clin.columns if col not in clin_cols]
        data = preprocess_data(pd.concat(frames_to_merge, axis=1).T.drop_duplicates().T, y)
    else:
        raise Exception("Invalid data type(s)!")

    return data, data_cols, clin_cols

def get_mm_challenge_data(clin_data_path = "/home/skapur/synapse/syn7222203/Clinical Data/sc2_Training_ClinAnnotations.csv", directory_folder='/home/skapur/synapse/syn7222203/CH2/'):
    from data_preprocessing import MMChallengeData
    mmcd = MMChallengeData(clin_data_path)

    mmcd.generateDataDict(
        clinicalVariables=["D_Age", "D_ISS",] + ["CYTO_predicted_feature_" + '{0:02d}'.format(i) for i in range(1, 19)],
        outputVariable="HR_FLAG", directoryFolder=directory_folder,
        columnNames=None, NARemove=[True, True], colParseFunDict=None)

    return mmcd