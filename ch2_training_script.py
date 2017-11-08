from ch2_training_resources import *
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import Normalizer, StandardScaler, FunctionTransformer, Imputer, QuantileTransformer, robust_scale, quantile_transform
from mlxtend.feature_selection import ColumnSelector
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_validate, StratifiedKFold, GridSearchCV
from sklearn.feature_selection import SelectPercentile, VarianceThreshold, f_classif
from sklearn.preprocessing import Normalizer, MaxAbsScaler, RobustScaler, MinMaxScaler
from sklearn.base import ClassifierMixin
from mlxtend.classifier import StackingClassifier
from os import listdir

# MAIN_FOLDER = "C:\\Users\\vitor\\synapse\\syn7222203\\"
# mmcd = get_mm_challenge_data(directory_folder=MAIN_FOLDER + "CH2",
#                              clin_data_path=MAIN_FOLDER + "Clinical Data\\sc2_Training_ClinAnnotations.csv")
mmcd = get_mm_challenge_data()
data_dict = mmcd.dataDict
metrics = ["accuracy", "recall", "f1", "neg_log_loss", "precision", "roc_auc"]


def make_default_selective_transformer(toSelect, toKeep):
    transform_steps = [('norm', Normalizer()), ('scale', StandardScaler()), ('select', SelectPercentile())]
    return selective_transformer(toSelect, toKeep, transform_steps)


data_combinations = {
    'microarrays':
        {
            'data': ('MA', 'gene'),
            'transform_by_study': True,
            'split_by_study': False
        },
    'rnaseq':
        {
            'data': ('RNASeq', 'gene'),
            'transform_by_study': False,
            'split_by_study': False,
        }
}

classifier_dict = {
    DecisionTreeClassifier: {'criterion': ['gini'], 'max_depth': [4], 'splitter': ['random']},
    LogisticRegression: {'solver': ['newton-cg'], 'C': [0.1, 1, 10], 'penalty': ["l2"], 'tol': [0.001],
                         'multi_class': ['multinomial']},
    SVC: {'kernel': ['linear', 'rbf'], 'C': [0.1, 1, 10], 'probability': [True], 'gamma': [0.0001, 0.1]},
    BaggingClassifier: {'max_samples': [1], 'bootstrap': [True]},
    RandomForestClassifier: {'max_depth': [5], 'criterion': ['entropy'], 'n_estimators': [20, 500]},
    GaussianNB: {}
}

feature_selection_base_steps = [
    # ('normalize', Normalizer()),
    # ('scaler', MaxAbsScaler()),
    # ('variance', VarianceThreshold()),
    ('percentile', SelectPercentile(percentile=10))
]

param_grid = {clf: {**{"estimator__" + k: v for k, v in params.items()}} for clf, params in classifier_dict.items()}
base_params = {'selector__transform__select__percentile': [1, 5, 10, 20]}
gs_params = dict(error_score=0, verbose=3, scoring=metrics, refit='f1', n_jobs=-1)

transform_steps_norm = [('norm', Normalizer()), ('select', SelectPercentile())]
transform_steps_maxabs = [('norm', MaxAbsScaler()), ('select', SelectPercentile())]
transform_steps_std_scale = [('norm', StandardScaler()), ('select', SelectPercentile())]
transform_steps_norm_maxabs = [('norm', Normalizer()), ('scale', (MaxAbsScaler())), ('select', SelectPercentile())]
impute_steps = [('imputer', Imputer(strategy='median', axis=0))]

selectors = {
    'imputer_norm': lambda data, clin: selective_transformer(data, clin, transform_steps_norm),
    'imputer_max_abs': lambda data, clin: selective_transformer(data, clin, transform_steps_maxabs),
    'imputer_norm_max_abs': lambda data, clin: selective_transformer(data, clin, transform_steps_norm_maxabs),
    'imputer_std_scale': lambda data, clin: selective_transformer(data, clin, transform_steps_std_scale)
}

RNA_d, RNA_c, RNA_fts = data_dict[('RNASeq', 'gene')]
RNA_dn = RNA_d.dropna(how='all', axis=0)
RNA_cn = RNA_c.loc[RNA_dn.index, :]


def binarize_rows_by_quantile(df, q):
    return (df.T > df.quantile(q, axis=1)).T


RNA_xo, RNA_yo = preprocess_data(RNA_d, RNA_fts, axis=0)
RNA_x, RNA_y = RNA_xo.dropna(axis=0), RNA_yo[RNA_xo.dropna(axis=0).index]

quantile_steps = np.linspace(0.1, 0.9, 5)

def quantile_dataframe(X, y, quantile_steps):
    fc_dict = {i: f_classif(binarize_rows_by_quantile(X, q).astype(int), y)[1] for i, q in
               enumerate(quantile_steps)}
    fcdf = pd.DataFrame(fc_dict, index=X.columns)
    return fcdf

def f_binarization_by_row(X, y, quantile_steps):
    fcdf = quantile_dataframe(X, y, quantile_steps)
    fcl_min = fcdf.idxmin(axis=1).dropna().astype(int)
    quantile_combinations = {q_out: [g for g, q in fcl_min.iteritems() if q == q_out] for q_out in fcl_min.unique()}
    return binarize_multiple_quantiles(X, quantile_combinations)

def binarize_multiple_quantiles(X, quantile_combinations):
    return pd.concat([binarize_rows_by_quantile(X, quantile_steps[q]).loc[:, gs] for q, gs in
                      quantile_combinations.items()], axis=1).astype(int), quantile_combinations


RNA_fcdf = quantile_dataframe(RNA_x, RNA_y, quantile_steps)


RNA_bin_total = pd.concat([binarize_rows_by_quantile(RNA_x, q) for q in quantile_steps], axis=1).astype(int)

pipe_steps = [
    ('imputer',Imputer(strategy='median',axis=1,missing_values='NaN')),
    ('variance',VarianceThreshold()),
    ('percentile', SelectPercentile(percentile=10))
]
pipe = Pipeline(steps=pipe_steps)

minmax = lambda x: (x-min(x))/(max(x)-min(x))

RNA_x_final = pipe.fit_transform(RNA_bin_total, RNA_y)
RNA_ci = Imputer(strategy='median',axis=0,missing_values='NaN').fit_transform(RNA_c.loc[RNA_bin_total.index,['D_Age','D_ISS']].values, RNA_y)
RNA_ci[:,0] = minmax(RNA_ci[:,0])*10
RNA_ci[:,1] = minmax(RNA_ci[:,1])*10
RNA_ci = np.concatenate([RNA_ci, np.mean(RNA_ci, axis=1).reshape(-1,1)], axis=1)

RNA_x_total = np.concatenate([RNA_x_final, RNA_ci], axis=1)
data_fts = RNA_x_final.shape[1]


clf1 = LogisticRegression(C=1, class_weight='balanced', penalty='l2')
clf2 = MLPClassifier()
clf3 = SVC(probability=True)


eclf = StackingClassifier(classifiers=[clf1, clf2, clf3],
                          meta_classifier=LogisticRegression(),
                          use_probas=True,
                          average_probas=False)

pipe.fit(RNA_bin_total, RNA_y)
eclf.fit(RNA_x_total, RNA_y)
save_as_pickle(pipe, 'rnaseq_stack_pipeline_08112017')
save_as_pickle(eclf, 'rnaseq_stack_classifier_08112017')
print(report(cross_val_function(RNA_x_total, RNA_y, eclf)))



RNA_x_scores = clf.predict_proba(RNA_x_final)[:,1].reshape(-1,1)
RNA_c_old = (RNA_ci[:,0] > 50).astype(int).reshape(-1,1)
RNA_c_adv = (RNA_ci[:,1] > 2).astype(int).reshape(-1,1)

RNA_stack_total = np.concatenate([RNA_x_scores], axis=1)

clf_stack = DecisionTreeClassifier()
print(report(cross_val_function(RNA_stack_total, RNA_y, clf_stack)))


pipe.fit(RNA_bin_total, RNA_y)
save_as_pickle(pipe, 'rnaseq_logreg_pipeline_07112017')
save_as_pickle(clf, 'rnaseq_logreg_classifier_07112017')

# RNA_ci = Imputer(strategy='median',axis=0,missing_values='NaN').fit_transform(RNA_c.loc[RNA_bin_total.index,['D_Age','D_ISS']].values, RNA_y)
# RNA_c_old = (RNA_ci[:,0] > 50).astype(int).reshape(-1,1)
# RNA_c_adv = (RNA_ci[:,1] > 2).astype(int).reshape(-1,1)
# RNA_cif = np.concatenate([RNA_ci,RNA_c_old,RNA_c_adv], axis=1)
# RNA_xc = np.concatenate([RNA_ci,RNA_x_final], axis=1)
# RNA_xci = Imputer(strategy='median',axis=0,missing_values='NaN').fit_transform(RNA_xc, RNA_y)


pipe_steps = [
    ('imputer',Imputer(strategy='median',axis=1,missing_values='NaN')),
    ('scaler', FunctionTransformer(lambda x: quantile_transform(x, axis=1, n_quantiles=5, output_distribution='uniform'))),
    ('variance', VarianceThreshold()),
    ('percentile', SelectPercentile(percentile=20)),
]

RNA_xt = Pipeline(pipe_steps).fit_transform(RNA_x, RNA_y)
clf = LogisticRegression()
print(RNA_xt.shape)
print(report(cross_val_function(RNA_xt, RNA_y, clf)))
# pipe_steps = [
#     ('variance', Normalizer(axis=0)),
#     ('percentile', SelectPercentile(percentile=1)),
#     ('estimator', GaussianNB())
# ]



# for s_name, selector in selectors.items():
#     for name, combination_params in data_combinations.items():
#         cross_validate_combination(s_name, mmcd, name, combination_params, param_grid, selector, base_params, gs_params)

# filtrar genes por sample, pela mediana das rows!


# gene_names = pd.read_csv('gene_names/non_alt_loci_set.tsv', sep='\t')
# feature_dict = {fp: pickle.load(open('serialized_features/' + fp, 'rb')) for fp in listdir("serialized_features")}
#
# gfa = [k.split('_')[0] for k in feature_dict['MuTectsnvs_filtered_genesFunctionAssociated_featColumns_CH1.pkl']]
# scoring = feature_dict['MuTectsnvs_filtered_genesScoring_featColumns_CH1.pkl']
# tlod = [k.split('_')[1] for k in feature_dict['MuTectsnvs_filtered_genesTlod_featColumns_CH1.pkl']]
# feature_list = list(chain(*[k.split('.')[0].split('-') for k in list(set(scoring) | set(tlod) | set(gfa))]))
#
# indices = gene_names['symbol'].isin(feature_list)
# vcf_features = gene_names['entrez_id'][indices].dropna()

clf_dict = {type(clf()).__name__: clf for clf in param_grid}
max_by_f1 = lambda x: x.sort_values('mean_test_f1', ascending=False).drop_duplicates(['notes'])

rna = max_by_f1(pd.read_csv("rnaseq_crossval_results.csv"))
ma = max_by_f1(pd.read_csv("microarrays_crossval_results.csv"))

rna_best = rna[rna['mean_test_f1'] > np.median(rna['mean_test_f1'])]
ma_best = ma[ma['mean_test_f1'] > np.median(ma['mean_test_f1'])]

X_total_ma, y_orig_ma, ma_selector = get_data_from_combination(('MA', 'gene'), data_dict, impute_steps,
                                                               transform_steps_maxabs)
ma_clf = gen_clf_list_from_rankings(ma_best, clf_dict, ma_selector)

ma_selector.fit_transform(X_total_ma, y_orig_ma)

for name, clf in ma_clf:
    print(name)
    clf.fit(X_total_ma, y_orig_ma)

ma_cvr = cross_val_function(X_total_ma, y_orig_ma, ma_voting, StratifiedKFold(n_splits=10))
print(report(ma_cvr))

X_total_rna, y_orig_rna, rna_selector = get_data_from_combination(('RNASeq', 'gene'), data_dict, transform_steps_maxabs)
rna_clf = gen_clf_list_from_rankings(rna_best, clf_dict, rna_selector)
rna_voting = VotingClassifier(estimators=rna_clf, voting='soft', n_jobs=-1)
rna_pipe = Pipeline(steps=[('selector', rna_selector), ('estimator', rna_voting)])

rna_cvr = cross_val_function(X_total_rna, y_orig_rna, rna_pipe, StratifiedKFold(n_splits=10))
print(report(rna_cvr))
# row['notes'].split(';')[0]



eclf1 = VotingClassifier(estimators=ma_clf,
                         voting='soft', n_jobs=-1)

# import pickle
# with open('a','wb') as f: pickle.dump(get_df_cols, f);
#
# hov_steps = [('sp', SelectPercentile(percentile=1)),
#              ('clf', RandomForestClassifier(max_depth=5, criterion='entropy', n_estimators=20))]
# emtab_steps = [('sp', SelectPercentile(percentile=1)), ('clf', BaggingClassifier(max_samples=1))]
# gsm_steps = [('sp', SelectPercentile(percentile=1)), ('clf', BaggingClassifier(max_samples=1))]
#
# hovon = X_orig_st.loc[X_orig_st['Study'] == 'HOVON65', :], y_orig[X_orig_st['Study'] == 'HOVON65'], hov_steps
# emtab = X_orig_st.loc[X_orig_st['Study'] == 'EMTAB4032', :], y_orig[X_orig_st['Study'] == 'EMTAB4032'], emtab_steps
# gsm = X_orig_st.loc[X_orig_st['Study'] == 'GSE24080UAMS', :], y_orig[X_orig_st['Study'] == 'GSE24080UAMS'], gsm_steps
#
# ft_vec = []
# clfs = []
# for df, y, steps in [hovon, emtab, gsm]:
#     ppl = Pipeline(steps=steps)
#     dft = df.drop('Study', axis=1)
#     ppl.fit(dft, y)
#     clfs.append(ppl)
#
#
# def compare_two_sets(s1, s2, name1='Set1', name2='Set2'):
#     intersect = s1 & s2
#     exclusive_s1 = s1 - s2
#     exclusive_s2 = s2 - s1
#     print('*' * 80)
#     print(name1, 'has', len(s1), 'elements.')
#     print(name2, 'has', len(s2), 'elements.')
#     print('Intersection has', len(intersect), 'elements.')
#     print(name1, 'has', len(exclusive_s1), ' exclusive elements.')
#     print(name2, 'has', len(exclusive_s2), ' exclusive elements.')
#     print('*' * 80)
#
#
# compare_two_sets(ft_vec[0], ft_vec[1], 'HOVON', 'EMTAB')
# compare_two_sets(ft_vec[0], ft_vec[2], 'HOVON', 'GSM')
# compare_two_sets(ft_vec[1], ft_vec[2], 'EMTAB', 'GSM')
#
# best_features = list(ft_vec[0] | ft_vec[1] | ft_vec[2] | set('D_Age'))
#
# X_orig_st_bf = X_orig_st.loc[:, best_features]
#
# clf = RandomForestClassifier(max_depth=5, criterion='entropy', n_estimators=20)
#
# print(report(cross_val_function(X_orig_st_bf.dropna(axis=1), y_orig, clf, StratifiedKFold(n_splits=10), metrics)))
#
# for ppl in clfs:
