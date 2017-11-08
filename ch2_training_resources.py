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
from sklearn.preprocessing import Normalizer, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_validate, StratifiedKFold, GridSearchCV
from sklearn.feature_selection import SelectPercentile, VarianceThreshold
from sklearn.preprocessing import Normalizer, MaxAbsScaler, RobustScaler, MinMaxScaler
from sklearn.base import clone as clone_sklearn
from itertools import combinations, chain

import numpy as np
import pandas as pd
import sys
import pickle

# classifier_dict = {
#     DecisionTreeClassifier: {'criterion': ['gini'], 'max_depth': [4], 'splitter': ['random']},
#     LogisticRegression: {'solver': ['newton-cg'], 'C': [1], 'penalty': ["l2"], 'tol': [0.001], 'multi_class': ['multinomial']},
#     SVC: {'kernel': ['linear'], 'C': [0.5], 'probability': [True], 'gamma': [0.0001]},
#     BaggingClassifier: {'max_samples': [1], 'bootstrap': [True]},
#     RandomForestClassifier: {'max_depth': [5], 'criterion': ['entropy'], 'n_estimators': [20,500]}
# }
#
# feature_selection_base_steps = [
#     #('normalize', Normalizer()),
#     #('scaler', MaxAbsScaler()),
#     #('variance', VarianceThreshold()),
#     ('percentile', SelectPercentile(percentile=10))
# ]
#
# pca_step = ('pca', PCA(n_components=300))
#
# base_param_grid = {
# 'percentile__percentile':[1,5,10,20,30]
# #'variance__threshold':[0.0]
# }

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
        X = X_orig.dropna(axis=axis, how='any')
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

        data = preprocess_data(pd.concat(objs=[d_file, d_clin], axis = 1), y)
        data = (data[0].dropna(axis=1, how='all'), data[1])
        data_cols = [col for col in d_file.columns if col in data[0].columns]
        clin_cols = [col for col in d_clin.columns if col in data[0].columns]
    elif isinstance(combination, list):
        frames_to_merge = []
        for indcomb in combination:
            d_file, d_clin, y = data_dict[indcomb]
            frames_to_merge.append(d_file)
            frames_to_merge.append(d_clin)
            data_cols = data_cols + [col for col in d_file.columns if col not in data_cols]
            clin_cols = clin_cols + [col for col in d_clin.columns if col not in clin_cols]
        data = preprocess_data(pd.concat(frames_to_merge, axis=1).T.drop_duplicates().T, y).dropna(axis=1)
    else:
        raise Exception("Invalid data type(s)!")

    return data, data_cols, clin_cols



def join_clin_data_dataframe(data, X_data_columns, X_clin_columns):
    X_orig, y_orig = data
    X_orig_nar = X_orig.dropna(axis=1, how='all')
    #X_orig_nar_fill = X_orig_nar.fillna(X_orig_nar.median())
    X_orig_data, X_orig_clin = X_orig_nar.loc[:, X_data_columns], X_orig_nar.loc[:,X_clin_columns]

    X_total = pd.concat([X_orig_data, X_orig_clin], axis=1)
    X_data_columns, X_clin_columns = X_orig_data.columns, X_orig_clin.columns
    return X_total, y_orig, X_data_columns, X_clin_columns




def parameter_search_cross_validation(X, y, cv, estimator_tup, selector_tup, df_note_string, report_best='mean_test_f1',**kwargs):
    estimator, estimator_params = estimator_tup
    selector, selector_params = selector_tup
    pipe = Pipeline([('selector', selector), ('estimator', estimator)])
    ppl_params = {**estimator_params, **selector_params}
    grid_search = GridSearchCV(pipe, cv=cv, param_grid=ppl_params, **kwargs)
    grid_search.fit(X, y)
    cvr_frame = pd.DataFrame(grid_search.cv_results_)
    cvr_frame['notes'] = df_note_string
    if report_best is not None and report_best in cvr_frame.columns:
        print('Best', report_best, ":", cvr_frame[report_best].max())
    return grid_search, cvr_frame


def cross_validate_combination(file_name, mmcd, name, combination_params, param_grid, selector_fx, base_params, gs_params, cv = StratifiedKFold(n_splits=10)):
    combination = combination_params['data']
    print("Combination", name, ":", str(combination))

    X_total, y_orig, X_data_columns, X_clin_columns = join_clin_data_dataframe(
        *read_from_data_dict(combination, mmcd.dataDict))

    data_col_ids = np.where(X_total.columns.isin(X_data_columns))[0]
    clin_col_ids = np.where(X_total.columns.isin(X_clin_columns))[0]

    selector = selector_fx(data_col_ids, clin_col_ids)
    selector_tup = (selector, base_params)
    X_study_index = mmcd.clinicalData.loc[X_total.index, 'Study']

    cvr_frames, report_strs, permutation_best = [], [], []

    if combination_params['split_by_study']:
        opts = X_study_index.unique()
        study_permutations = list(chain(*[combinations(opts, i) for i in range(1, len(opts) + 1)]))
        for permutation in study_permutations:
            X, y, X_not, y_not = select_by_study(X_total, X_study_index, y_orig, permutation)
            for clf, params in param_grid.items():
                df_note_string = ';'.join([type(clf).__name__, '-'.join(permutation)])
                gs, cvr_frame = parameter_search_cross_validation(X, y, cv, (clf(), params),
                                                                  selector_tup, df_note_string,
                                                                  report_best='mean_test_f1', **gs_params)
                cvr_frames.append(cvr_frame)
    else:
        for clf, params in param_grid.items():
            df_note_string = ';'.join([type(clf).__name__,'_Full'])
            gs, cvr_frame = parameter_search_cross_validation(X_total, y_orig, cv, (clf(), params),
                                                              selector_tup, df_note_string,
                                                              report_best='mean_test_f1', **gs_params)
            cvr_frames.append(cvr_frame)

    df_final = pd.concat(cvr_frames, axis=0)
    df_final.to_csv(str(name) + "_" + file_name +"_crossval_results" + ".csv")
    return df_final

def select_by_study(X_total, X_study_index, y_orig, studyNames):
    sample_index = X_study_index.isin(studyNames)
    comp_index = np.logical_not(sample_index)
    return X_total.loc[sample_index, :], y_orig[sample_index], X_total.loc[comp_index, :], y_orig[comp_index]

def get_df_cols(x, ind):
        return x[:, ind]

def selective_transformer(toSelect, toKeep, previous_steps, transform_steps, **kwargs):
    select_pipeline = Pipeline(
        steps=previous_steps +[('add', FunctionTransformer(func=get_df_cols, kw_args={'ind': toSelect}))] + transform_steps)
    ftsel = FeatureUnion(transformer_list=[('transform', select_pipeline),
                                           ('add', FunctionTransformer(func=get_df_cols, kw_args={'ind': toKeep}))])
    ftsel.set_params(**kwargs)
    return ftsel


def get_mm_challenge_data(clin_data_path = "/home/skapur/synapse/syn7222203/Clinical Data/sc2_Training_ClinAnnotations.csv", directory_folder='/home/skapur/synapse/syn7222203/CH2/'):
    from data_preprocessing import MMChallengeData
    mmcd = MMChallengeData(clin_data_path)

    mmcd.generateDataDict(
        clinicalVariables=["D_Age", "D_ISS"] + ["CYTO_predicted_feature_" + '{0:02d}'.format(i) for i in range(1, 19)],
        outputVariable="HR_FLAG", directoryFolder=directory_folder,
        columnNames=None, NARemove=[False, True], colParseFunDict=None)

    return mmcd

def gen_clf_list_from_rankings(best, clf_dict, selector_instance):
    list = []
    selector = clone_sklearn(selector_instance)
    for row in best.iterrows():
        clf_name, studies = row[1]['notes'].split(';')
        clf_instance = clf_dict[clf_name]
        clf = clf_instance()
        pipe = Pipeline(steps=[('selector', selector), ('estimator', clf)])
        print(row[1]['params'])
        pipe.set_params(**eval(row[1]['params']))
        list.append((clf_name+'_'+studies, pipe))
    return list


def get_data_from_combination(combination, data_dict, previous_steps, transform_steps):
    X_total, y_orig, X_data_columns, X_clin_columns = join_clin_data_dataframe(
        *read_from_data_dict(combination, data_dict))

    data_col_ids = np.where(X_total.columns.isin(X_data_columns))[0]
    clin_col_ids = np.where(X_total.columns.isin(X_clin_columns))[0]

    selector = selective_transformer(data_col_ids, clin_col_ids, previous_steps, transform_steps)
    return X_total, y_orig, selector
