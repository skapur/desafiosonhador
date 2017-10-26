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
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import Normalizer, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_validate, StratifiedKFold, GridSearchCV
from sklearn.feature_selection import SelectPercentile, VarianceThreshold
from sklearn.preprocessing import Normalizer, MaxAbsScaler, RobustScaler, MinMaxScaler

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
            'data':('RNASeq','gene'),
            'transform_by_study':False,
            'split_by_study':False,
        }
}
classifier_dict = {
    DecisionTreeClassifier: {'criterion': ['gini'], 'max_depth': [4], 'splitter': ['random']},
    LogisticRegression: {'solver': ['newton-cg'], 'C': [0.1,1,10], 'penalty': ["l2"], 'tol': [0.001], 'multi_class': ['multinomial']},
    SVC: {'kernel': ['linear','rbf'], 'C': [0.1,1,10], 'probability': [True], 'gamma': [0.0001,0.1]},
    BaggingClassifier: {'max_samples': [1], 'bootstrap': [True]},
    RandomForestClassifier: {'max_depth': [5], 'criterion': ['entropy'], 'n_estimators': [20,500]},
    GaussianNB:{}
}

feature_selection_base_steps = [
    #('normalize', Normalizer()),
    #('scaler', MaxAbsScaler()),
    #('variance', VarianceThreshold()),
    ('percentile', SelectPercentile(percentile=10))
]

param_grid = {clf: {**{"estimator__" + k: v for k, v in params.items()}} for clf, params in classifier_dict.items()}
base_params = {'selector__transform__select__percentile': [1, 5, 10, 20]}
gs_params = dict(error_score=0, verbose=3, scoring=metrics, refit='f1', n_jobs=-1)

transform_steps_norm = [('norm', Normalizer()), ('select', SelectPercentile())]
transform_steps_maxabs = [('norm', MaxAbsScaler()), ('select', SelectPercentile())]
transform_steps_std_scale = [('norm', StandardScaler()), ('select', SelectPercentile())]
transform_steps_norm_maxabs = [('norm', Normalizer()), ('scale',(MaxAbsScaler())),('select', SelectPercentile())]
impute_steps = [('imputer',Imputer(strategy='median',axis=0))]

selectors = {
    'imputer_norm': lambda data, clin: selective_transformer(data, clin, transform_steps_norm),
    'imputer_max_abs': lambda data, clin: selective_transformer(data, clin, transform_steps_maxabs),
    'imputer_norm_max_abs':lambda data, clin: selective_transformer(data, clin, transform_steps_norm_maxabs),
    'imputer_std_scale':lambda data, clin: selective_transformer(data, clin, transform_steps_std_scale)
}

for s_name, selector in selectors.items():
    for name, combination_params in data_combinations.items():
        cross_validate_combination(s_name, mmcd, name, combination_params, param_grid, selector, base_params, gs_params)

clf_dict = {type(clf()).__name__: clf for clf in param_grid}
max_by_f1 = lambda x: x.sort_values('mean_test_f1', ascending=False).drop_duplicates(['notes'])

rna = max_by_f1(pd.read_csv("rnaseq_crossval_results.csv"))
ma = max_by_f1(pd.read_csv("microarrays_crossval_results.csv"))

rna_best = rna[rna['mean_test_f1'] > np.median(rna['mean_test_f1'])]
ma_best = ma[ma['mean_test_f1'] > np.median(ma['mean_test_f1'])]


X_total_ma, y_orig_ma, ma_selector = get_data_from_combination(('MA','gene'), data_dict, impute_steps, transform_steps_maxabs)
ma_clf = gen_clf_list_from_rankings(ma_best, clf_dict, ma_selector)


ma_selector.fit_transform(X_total_ma, y_orig_ma)

for name, clf in ma_clf:
    print(name)
    clf.fit(X_total_ma, y_orig_ma)


ma_cvr = cross_val_function(X_total_ma, y_orig_ma, ma_voting, StratifiedKFold(n_splits=10))
print(report(ma_cvr))

X_total_rna, y_orig_rna, rna_selector = get_data_from_combination(('RNASeq','gene'), data_dict, transform_steps_maxabs)
rna_clf = gen_clf_list_from_rankings(rna_best, clf_dict, rna_selector)
rna_voting = VotingClassifier(estimators=rna_clf, voting='soft',n_jobs=-1)
rna_pipe = Pipeline(steps=[('selector',rna_selector), ('estimator',rna_voting)])

rna_cvr = cross_val_function(X_total_rna, y_orig_rna, rna_pipe, StratifiedKFold(n_splits=10))
print(report(rna_cvr))
    #row['notes'].split(';')[0]



eclf1 = VotingClassifier(estimators=ma_clf,
                         voting = 'soft', n_jobs = -1)

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
