# Make predictions
for model in fit_models.keys():
    clf = fit_models[str(model)]
    mod = MMChallengePredictor(
        mmcdata = mmcd,
        predict_fun = lambda x: clf.predict(x)[0],
        confidence_fun = lambda x: 1 - min(clf.predict_proba(x)[0]),
        data_types = [("RNASeq", "gene"), ("RNASeq", "trans")],
        single_vector_apply_fun = lambda x: x,
        multiple_vector_apply_fun = lambda x: df_reduce(x.values.reshape(1,-1), [], fit = False, filename = 'transformers_rna_seq.sav')[0]
    )
    res[str(model)] = mod.predict_dataset()


mv_fun = lambda x: df_reduce(x.values.reshape(1,-1),[],scaler=trf_marrays['scaler'],fts=trf_marrays['fts'],fit = False)[0]

for model in fit_models.keys():
    print(model)
    clf = fit_models[str(model)]
    mod = MMChallengePredictor(
        mmcdata = mmcd,
        predict_fun = lambda x: clf.predict(x)[0],
        confidence_fun = lambda x: 1 - min(clf.predict_proba(x)[0]),
        data_types = [("MA", "gene")],
        single_vector_apply_fun = mv_fun,
        multiple_vector_apply_fun = lambda x: x)
    res[str(model)] = mod.predict_dataset()

