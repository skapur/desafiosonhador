import pickle
import sys
import numpy as np
import pandas as pd

from data_preprocessing import MMChallengeData, MMChallengePredictor
from sklearn.preprocessing import Imputer

def read_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def binarize_rows_by_quantile(df, q):
    return (df.T > df.quantile(q, axis=1)).T

def generate_binary_features(x, qs):
    return pd.concat([binarize_rows_by_quantile(x, q).rename(columns={st:str(st)+"_"+str(q) for st in x.columns.tolist()}) for q in qs], axis=1).astype(int)

def minmax(x):
    return (x-min(x))/(max(x)-min(x))

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

def get_ch2_data(clin_file_path, data_file_path, forTraining=True):

	#clin_file_path = '/home/skapur/synapse/syn7222203/Clinical Data/sc2_Training_ClinAnnotations.csv'
	#data_file_path = '/home/skapur/synapse/syn7222203/CH2'
	mmcd = MMChallengeData(clin_file_path)

	colname_dict = {
		('RNASeq', 'gene'): read_pickle('RNASeq_genes_08112017'),
		('MA', 'gene'): read_pickle('MA_genes_08112017'),
		('MA', 'probe'): [],
		('RNASeq', 'trans'): []
	}
	col_parse_dict = {
			('RNASeq', 'trans'): lambda x: x.split('.')[0]
		}
	clinical_variables = ["D_Age", "D_ISS"]
    if forTraining:
	       output_variable = "HR_FLAG"
    else:
        output_variable = "D_Age"
	mmcd.generateDataDict(clinicalVariables=clinical_variables,
						  outputVariable=output_variable,
						  directoryFolder=data_file_path,
						  columnNames=colname_dict,
						  NARemove=[False, True],
						  colParseFunDict=col_parse_dict
						  )

	### RNASeq data
	RNA_quantile_steps = np.linspace(0.1, 0.9, 5)

	RNA_X, RNA_C, RNA_y = mmcd.dataDict[('RNASeq', 'gene')]
	RNA_Xb = generate_binary_features(RNA_X, RNA_quantile_steps)

	RNA_transformer = read_pickle('rnaseq_stack_pipeline_08112017')
	RNA_imputer = Imputer(strategy='median', axis=0)

	RNA_Xbd_imp = RNA_imputer.fit_transform(RNA_Xb)
	RNA_Xbd_sel = RNA_transformer.transform(RNA_Xbd_imp)

	RNA_x_final = pd.DataFrame(RNA_Xbd_sel,index=RNA_X.index)

	RNA_step_list = RNA_transformer.steps
	sp_index = np.where(RNA_step_list[-1][1].get_support())[0]
	vtr_index = np.where(RNA_step_list[-2][1].get_support())[0]

	RNA_x_final.columns = RNA_Xb.columns[vtr_index[sp_index]].tolist()

	### Microarray data

	MA_X, MA_C, MA_y = mmcd.dataDict[('MA', 'gene')]
	MA_imputer = Imputer(strategy='median', axis=0)
	MA_X_imp = pd.DataFrame(MA_imputer.fit_transform(MA_X, MA_y))

	MA_transformer = read_pickle('transformers_microarrays.sav')
	MA_Xt, _, MA_support = df_reduce(MA_X_imp, [], scaler = MA_transformer['scaler'], fts = MA_transformer['fts'], fit = False)
	MA_X_final = pd.DataFrame(MA_Xt, index=MA_X.index, columns=MA_X.columns[MA_support])

	return RNA_x_final, RNA_C, RNA_y, MA_X_final, MA_C, MA_y

if __name__ == '__main__':
	clin_file_path = '/home/skapur/synapse/syn7222203/Clinical Data/sc2_Training_ClinAnnotations.csv'
	data_file_path = '/home/skapur/synapse/syn7222203/CH2'

	RNA_x_final, MA_X_final = get_ch2_data(clin_file_path, data_file_path)



