import numpy as np
import pandas as pd
from sklearn import decomposition
from sklearn.feature_selection import SelectPercentile
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from data_preprocessing import MMChallengeData, MMChallengePredictor

DIR = "C:/Users/vitor/synapse/syn7222203"
mmcd = MMChallengeData(DIR)
mmcd.generateDataDict()

MA_gene, MA_gene_cd, MA_gene_output = mmcd.dataDict[("MA", "gene")]
MA_probe, MA_probe_cd, MA_probe_output = mmcd.dataDict[("MA", "probe")]
RNA_gene, RNA_gene_cd, RNA_gene_output = mmcd.dataDict[("RNASeq", "gene")]
RNA_trans, RNA_trans_cd, RNA_trans_output = mmcd.dataDict[("RNASeq", "trans")]

pipeline_factory = lambda clf: Pipeline(steps=[("classify", clf)])
cross_val_function = lambda X, y, clf: cross_validate(pipeline_factory(clf), X, y,
                                                      scoring=["accuracy", "recall", "f1", "neg_log_loss", "precision",
                                                               "roc_auc"], cv=10)


def report(cvr):
    for name, array in cvr.items():
        print(name + " : " + str(np.mean(array)) + " +/- " + str(np.std(array)))


def rnaseq_prepare_data(X_gene, X_trans, y_orig):
    X = pd.concat([X_gene, X_trans], axis=1).dropna(axis=0)
    y = y_orig[X.index]
    valid_samples = y != "CENSORED"
    X, y = X[valid_samples], y[valid_samples] == "TRUE"
    y = y.astype(int)
    return X, y


def df_reduce(X, y, scaler, fts, fit=True):
    if fit:
        scaler.fit(X, y)
    X = scaler.transform(X)
    if fit:
        fts.fit(X, y)
    X = fts.transform(X)
    return X, y, fts.get_support(True)


from sklearn.naive_bayes import GaussianNB

scl = MaxAbsScaler()
fts = SelectPercentile(percentile=30)
clf = GaussianNB()

Xv, yv = rnaseq_prepare_data(RNA_gene, RNA_trans, RNA_gene_output)
Xvt, yvt, fts_vector = df_reduce(Xv, yv, scl, fts, True)
cvr = cross_val_function(Xvt, yvt, clf)
report(cvr)

Xt, yt, support_ft = df_reduce(Xv, yv, scl, fts, False)
clf_final = GaussianNB()
clf_final.fit(Xt, yt)

mod = MMChallengePredictor(
    mmcdata=mmcd,
    predict_fun=lambda x: clf_final.predict(x)[0],
    confidence_fun=lambda x: 1 - min(clf_final.predict_proba(x)[0]),
    data_types=[("RNASeq", "gene"), ("RNASeq", "trans")],
    single_vector_apply_fun=lambda x: x,
    multiple_vector_apply_fun=lambda x: df_reduce(x.values.reshape(1,-1), [], scl, fts, False)[0]
)

rdf = mod.predict_dataset()

### EXPERIMENTAL


scl.transform(vec)
for scaler in ppscalers:
    print("*" * 80)
    print(scaler.__name__)
    cv = cross_val_function(*microarray_prepare_data(MA_probe, MA_probe, MA_gene_output, scaler))
    report(cv)

report(cross_val_function(*microarray_prepare_data(MA_gene, MA_probe, MA_gene_output)))

cv_m = cross_validate(pipeline_factory(GaussianNB()), Xr, yr,
                      scoring=["accuracy", "recall", "f1", "neg_log_loss", "precision", "roc_auc"], cv=10)

import matplotlib.pyplot as plt
from numpy import where

pca = decomposition.PCA(n_components=100)
X_pca = pca.fit_transform(X, y)

sp = SelectPercentile(percentile=30)
sp.fit(X, y)
sig = where(sp.pvalues_ < 0.01)

len(sig[0])

X_sig = X[:, sig[0]]
X_sig.shape


def plot_pca(pca, colors, y_ids, y_labels, plot_title, plotkwargs, legendkwargs={}):
    fig = plt.figure()
    for color, i, target_name in zip(cokm.
                                             lors, y_ids, y_labels):
        plot_args = plotkwargs(color=color, target_name=target_name)
        plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], **plot_args)
    plt.legend(**legendkwargs)
    plt.title(plot_title)
    return fig


pca = decomposition.PCA(n_components=50)
pca.explained_variance_ratio_
X_pca = pca.fit_transform(X_sig, y)

plot_args = lambda color, target_name: dict(color=color, alpha=.8, lw=2, label=target_name)
legend_args = dict(loc='best', shadow=False, scatterpoints=1)

plot_pca(pca, ['red', 'green'], [0, 1], ['False', 'True'], "PCA", legendkwargs=legend_args, plotkwargs=plot_args)

from sklearn.cluster.k_means_ import KMeans
from scipy.stats import rankdata
from mpl_toolkits.mplot3d import Axes3D
from numpy import concatenate, array

km = KMeans(n_clusters=2)
km.fit(X_sig, y)
y_pred = km.predict(X_sig)

rank = rankdata(sp.pvalues_, "ordinal")
colors = ["red", "green", "blue", "yellow", "pink", "black"]
fig = plt.figure()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
ax.scatter(X[:, rank[0]], X[:, rank[1]], X[:, rank[2]], c=y_pred)
plt.title("Incorrect Number of Blobs")

features = concatenate((X[:, rank[:4]], array(y_pred).reshape((428, 1)), array(y).reshape((428, 1))), axis=1)

df = pd.DataFrame(features[:, [4, 5]]).groupby(0).hist()
