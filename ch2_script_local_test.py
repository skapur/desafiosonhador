import pickle
from os import chdir
chdir("C:/Users/vitor/MEOCloud/Projectos/DREAMChallenge_MultipleMyeloma/desafiosonhador")
import sys, getopt
from data_preprocessing import MMChallengeData, MMChallengePredictor
from genomic_data_test import df_reduce

#DIR = "/home/skapur/synapse/syn7222203/"
#DIR = 'D:\Dropbox\workspace_intelij\DREAM_Challenge'
#mmcd = MMChallengeData(DIR)
print("Starting the script!")
print("Reading clinical data")
argv = "C:/Users/vitor/synapse/syn7222203/Clinical Data/sc2_Training_ClinAnnotations.csv"
mmcd = MMChallengeData(argv)

with open('colnames.sav','rb') as f:
    colname_dict = pickle.load(f)

print("Reading files using information from clinical data")
mmcd.generateDataDict(clinicalVariables=["D_Age", "D_ISS"], outputVariable="HR_FLAG", directoryFolder='/home/skapur/synapse/syn7222203/CH2', columnNames=None, NARemove=[True, False])


X, cd, y = mmcd.dataDict[("MA","gene")]
Xn = X.dropna(axis = 1)
y = y == "TRUE"
y = y.astype(int)

Xm, 

coln = Xn.shape[1]
from minepy import MINE
mine = MINE(alpha=0.6, c=15, est="mic_approx")
scores = {}
for i in range(coln):
    mine.compute_score(Xn.values[:,i], y)
    scores[i] = mine.mic()

coldict = {datype: df[0].columns.tolist() for datype,df in mmcd.dataDict.items()}
with open('colnames.sav','wb') as f:
    pickle.dump(coldict, f)
# ======== RNA-SEQ ========

print("Loading RS transformer")
#Load fitted transformers and model
with open('transformers_rna_seq.sav', 'rb') as f:
    trf_rseq = pickle.load(f)

print("Loading RS classifier")
with open('fittedModel_rna_seq.sav', 'rb') as f:
    clf_rseq = pickle.load(f)

mv_fun_rseq = lambda x: df_reduce(x.values.reshape(1,-1), [], fit = False, scaler = trf_rseq['scaler'], fts = trf_rseq['fts'])[0]

print("Predicting with RS...")
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
print("Loading MA transformer")
#Load fitted transformers and model
with open('transformers_microarrays.sav', 'rb') as f:
    trf_marrays = pickle.load(f)

print("Loading MA classifier")
with open('fittedModel_microarrays.sav', 'rb') as f:
    clf_marrays = pickle.load(f)

mv_fun = lambda x: df_reduce(x.values.reshape(1,-1), [], scaler = trf_marrays['scaler'], fts = trf_marrays['fts'], fit = False)[0]

# Make predictions
print("Predicting with MA...")
mod_marryas = MMChallengePredictor(
            mmcdata = mmcd,
            predict_fun = lambda x: clf_marrays.predict(x)[0],
            confidence_fun = lambda x: clf_marrays.predict_proba(x)[0][1],
            data_types = [("MA", "gene")],
            single_vector_apply_fun = mv_fun,
            multiple_vector_apply_fun = lambda x: x
)
res_marrays = mod_marryas.predict_dataset()

#Final dataset
print("Generating prediction matrix")
final_res = res_rseq.combine_first(res_marrays)
final_res.columns = ['study', 'patient', 'predictionscore', 'highriskflag']
final_res['highriskflag'] = final_res['highriskflag'] == 1
final_res['highriskflag'] = final_res['highriskflag'].apply(lambda x: str(x).upper())

final_res["predictionscore"] = final_res["predictionscore"].fillna(value=0.5)
#final_res["highriskflag"] = final_res["highriskflag"].fillna(value=True)

print("Writing prediction matrix")
final_res.to_csv("predictions.tsv", index = False, sep = '\t')

print("Any failed prediction column in the prediction matrix?")
print(str(final_res.isnull().any()))
print("All failed prediction column in the prediction matrix?")
print(str(final_res.isnull().all()))
