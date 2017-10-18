from ch2_training_resources import *
import pprint
from data_preprocessing import MMChallengeData
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import  Pipeline

mmcd = get_mm_challenge_data()

data_dict = mmcd.dataDict
metrics = ["accuracy", "recall", "f1", "neg_log_loss", "precision", "roc_auc"]
param_grid = {clf() : {**base_param_grid, **{'classifier__'+k:v for k,v in params.items()}} for clf, params in classifier_dict.items()}

pprint.pprint(param_grid)
data_combinations = {'microarrays':('MA','gene'),'rnaseq':('RNASeq','gene')}

for name, combination in data_combinations.items():
    print("Combination",name,":",str(combination))
    data, X_data_columns, X_clin_columns = read_from_data_dict(combination, data_dict)
    X, y = data
    X = X.dropna(axis=1)
    cvr_frames = []
    for clf in param_grid:
        pipe = Pipeline(steps=feature_selection_base_steps+[('classifier',clf)],memory='memory')
        grid_search = GridSearchCV(pipe, cv=StratifiedKFold(n_splits=10), n_jobs=-1, param_grid=param_grid[clf], error_score=0, verbose=3, scoring=metrics, refit='f1')
        grid_search.fit(X, y)
        cvr_frames.append(pd.DataFrame(grid_search.cv_results_))

    pd.concat(cvr_frames, axis = 0).to_csv(name+"_cv_results_.csv")
    save_as_pickle(grid_search.estimator, "gscv_best_estimator_"+name+".csv")

