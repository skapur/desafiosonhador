from ch2_training_resources import *
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from itertools import combinations, chain

mmcd = get_mm_challenge_data()


data_dict = mmcd.dataDict
metrics = ["accuracy", "recall", "f1", "neg_log_loss", "precision", "roc_auc"]
param_grid = {clf() : {**base_param_grid, **{'classifier__'+k:v for k,v in params.items()}} for clf, params in classifier_dict.items()}
#data_combinations = {'microarrays':('MA','gene'),'rnaseq':('RNASeq','gene')}
data_combinations = {
    'microarrays':
        {
            'data':('MA','gene'),
            'transform_by_study':True,
            'split_by_study':True
        }
}

norm = Normalizer()
scaler = MaxAbsScaler()


## Normalize and scale micro-array dataframe before creating models
# dd = data_dict[("MA","gene")][0].assign(Study=mmcd.clinicalData.loc[data_dict[("MA","gene")][0].index,'Study'])
# ddx = apply_fx_by_study(dd, lambda x: scaler.fit_transform(norm.transform(x)), keep_study_col=True)
#
# #data_dict[("MA","gene")] = (ddx, *data_dict[("MA","gene")][1:])

for name, combination_params in data_combinations.items():
    combination = combination_params['data']

    print("Combination",name,":",str(combination))
    data, X_data_columns, X_clin_columns = read_from_data_dict(combination, data_dict)
    X_orig, y_orig = data
    X_orig = X_orig.dropna(axis=1)

    if combination_params['transform_by_study']:
        X_orig_st = X_orig.assign(Study=mmcd.clinicalData.loc[X_orig.index,'Study'])
        X_orig_st = apply_fx_by_study(X_orig_st, lambda x: scaler.fit_transform(norm.transform(x)), keep_study_col=True)

    cvr_frames = []
    report_strs = []
    if combination_params['split_by_study']:
        opts = X_orig_st['Study'].unique()
        study_permutations = list(chain(*[combinations(opts, i) for i in range(2, len(opts)+1)]))
        for permutation in study_permutations:
            print("Testing "+str(permutation)+" studies.")
            perm_index = X_orig_st['Study'].isin(permutation)
            X, y = X_orig_st.drop('Study', axis=1).loc[perm_index,:], y_orig[perm_index]
            #del X['Study']
            for clf in param_grid:
                print("Testing with "+str(type(clf).__name__))
                pipe = Pipeline(steps=feature_selection_base_steps + [('classifier', clf)], memory='memory')
                grid_search = GridSearchCV(pipe, cv=StratifiedKFold(n_splits=10), n_jobs=-1, param_grid=param_grid[clf],
                                           error_score=0, verbose=0, scoring=metrics, refit='f1')
                grid_search.fit(X, y)
                cvr_frame = pd.DataFrame(grid_search.cv_results_)
                cvr_frame['combination'] = str(permutation)
                cvr_frame['classifier'] = str(type(clf))
                cvr_frames.append(cvr_frame)
                print("Cross-validating with the original dataframe...")
                report_str = report(cross_val_function(X_orig_st.drop('Study', axis=1), y_orig, grid_search.best_estimator_, StratifiedKFold(n_splits=10), metrics))
                write_str = str(combination)+" "+str(permutation)+" "+type(clf).__name__+"\n"+report_str
                report_strs.append(write_str)

    else:
        for clf in param_grid:
            pipe = Pipeline(steps=feature_selection_base_steps+[('classifier',clf)],memory='memory')
            grid_search = GridSearchCV(pipe, cv=StratifiedKFold(n_splits=10), n_jobs=-1, param_grid=param_grid[clf], error_score=0, verbose=3, scoring=metrics, refit='f1')
            grid_search.fit(X, y)
            cvr_frame = pd.DataFrame(grid_search.cv_results_)
            cvr_frame['classifier'] = str(type(clf).__name__)
            cvr_frames.append(cvr_frame)

    df_final = pd.concat(cvr_frames, axis=0)
    df_final.to_csv(str(name)+"_crossval_results"+".csv")
    with open(name+'_full_ds_cross_validation.txt','w') as f:
        f.write('\n\n'.join(report_strs))

