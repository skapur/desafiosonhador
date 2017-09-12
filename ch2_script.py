#!/usr/bin/python

import pickle
import sys, getopt
from data_preprocessing import MMChallengeData, MMChallengePredictor
from genomic_data_test import df_reduce

def main(argv):
    #DIR = "/home/skapur/synapse/syn7222203/"
    #DIR = 'D:\Dropbox\workspace_intelij\DREAM_Challenge'
    #mmcd = MMChallengeData(DIR)
    mmcd = MMChallengeData(sys.argv[1])
    mmcd.generateDataDict()

    # ======== RNA-SEQ ========

    #Load fitted transformers and model
    with open('transformers_rna_seq.sav', 'rb') as f:
        trf_rseq = pickle.load(f)

    with open('fittedModel_rna_seq.sav', 'rb') as f:
        clf_rseq = pickle.load(f)

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


    # ======== MICROARRAYS ========

    #Load fitted transformers and model
    with open('transformers_microarrays.sav', 'rb') as f:
        trf_marrays = pickle.load(f)

    with open('fittedModel_microarrays.sav', 'rb') as f:
        clf_marrays = pickle.load(f)

    mv_fun = lambda x: df_reduce(x.values.reshape(1,-1), [], scaler = trf_marrays['scaler'], fts = trf_marrays['fts'], fit = False)[0]

    # Make predictions
    mod_marryas = MMChallengePredictor(
                mmcdata = mmcd,
                predict_fun = lambda x: clf_marrays.predict(x)[0],
                confidence_fun = lambda x: clf_rseq.predict_proba(x)[0][1],
                data_types = [("MA", "gene")],
                single_vector_apply_fun = mv_fun,
                multiple_vector_apply_fun = lambda x: x
    )
    res_marrays = mod_marryas.predict_dataset()
    
    #Final dataset
    final_res = res_rseq.combine_first(res_marrays)
    final_res.columns = ['study', 'patient', 'predictionscore', 'highriskflag']
    final_res['highriskflag'] = final_res['highriskflag'] == 1
    final_res['highriskflag'] = final_res['highriskflag'].apply(lambda x: str(x).upper())

    final_res.to_csv(sys.argv[2], index = False)

    print('Done!')

if __name__ == "__main__":
    main(sys.argv[1:])