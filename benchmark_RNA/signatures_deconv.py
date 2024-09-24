import pandas as pd
import json
import numpy as np
import gseapy as gp
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVR
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import lsq_linear
import os
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')
dirname = os.path.dirname(__file__)

pd.set_option("future.no_silent_downcasting", True)

_DATA_LM22 = pd.read_csv(os.path.join(dirname, "data/deconv_signatures/LM22.txt"), sep="\t", index_col=0)

with open(os.path.join(dirname, "data/deconv_signatures/Xcell_signatures.json")) as f:
    _DIC_Xcell_SIG = json.load(f)

_DATA_Xcell_coef = pd.read_excel(os.path.join(dirname, "data/deconv_signatures/Xcell_coef.xlsx"), engine="openpyxl", sheet_name=0,
                                 index_col=0)
_DATA_Xcell_P = _DATA_Xcell_coef["Power coefficient"]
_DATA_Xcell_V = _DATA_Xcell_coef["Calibration parameter"]
_DATA_Xcell_K = _DATA_Xcell_coef.iloc[:, 3:]


def get_CD8T_MCPcounter_score(data):

    gene = "CD8B"
    if gene in data.columns:
        def fun(row):
            return np.log(row[gene] + 1)
    else:
        def fun(row):
            return np.nan
    return fun


def get_CD8T_Xcell_score(data):
    pathways_dic = {}
    for pa, genes in _DIC_Xcell_SIG.items():
        temp = set(data.columns) & set(genes)
        if len(temp) > 0:
            pathways_dic[pa] = list(temp)

    if len(pathways_dic) > 0:
        min_size = min(15, np.min([len(g) for _, g in pathways_dic.items()]))
        ss_gsea = gp.ssgsea(data=data.T, gene_sets=pathways_dic, min_size=min_size)
        results_es = ss_gsea.res2d.pivot(index="Name", columns="Term", values="ES")
        results_es.index.name = None
        results_es.columns.name = None
        x_cell_raw = _xcell_aggregate(results_es.T)
        min_values = x_cell_raw.min(axis=1).astype("float64")

        def fun(row):
            gsea = gp.ssgsea(data=row.squeeze(), gene_sets=pathways_dic, min_size=min_size)
            res = gsea.res2d.pivot(index="Name", columns="Term", values="ES")
            res.index.name = None
            res.columns.name = None
            raw_scores = _xcell_aggregate(res.T).astype("float64")
            transformed_scores = _xcell_transform_scores(raw_scores.squeeze(), min_values)
            adjusted_scores = _xcell_spillover(transformed_scores, _DATA_Xcell_K.copy())
            return adjusted_scores.loc["CD8+ T-cells"].values[0]
    else:

        def fun(row):
            return np.nan

    return fun


def get_CD8T_CIBERSORT_score(data):
    Signature = _DATA_LM22.copy()
    common_genes = list(set(Signature.index) & set(data.columns))
    if len(common_genes) > 0:
        Signature_new = StandardScaler().fit_transform(Signature.loc[common_genes].values)

        def fun(row):
            row_scaled = StandardScaler().fit_transform(row[common_genes].values.reshape(-1, 1))
            freq, _ = _cibersort(row_scaled.squeeze(), Signature_new)
            freq = pd.DataFrame(freq.reshape(1, -1), columns=_DATA_LM22.columns)
            return freq["T cells CD8"].values[0]

    else:
        def fun(row):
            return np.nan

    return fun


def get_Immuno_CIBERSORT_score(data):
    Signature = _DATA_LM22.copy()
    common_genes = list(set(Signature.index) & set(data.columns))
    if len(common_genes) > 0:
        Signature_new = StandardScaler().fit_transform(Signature.loc[common_genes].values)

        def fun(row):
            row_scaled = StandardScaler().fit_transform(row[common_genes].values.reshape(-1, 1))
            freq, _ = _cibersort(row_scaled.squeeze(), Signature_new)
            freq = pd.DataFrame(freq.reshape(1, -1), columns=_DATA_LM22.columns)
            return (1.13 * freq["B cells naive"] + 1.36 * freq["B cells memory"] + 5.92 * freq["Eosinophils"]
                    + 9.70 * freq["T cells follicular helper"] + 15.34 * freq["T cells regulatory (Tregs)"]
                    - 1.14 * freq["Macrophages M0"] - 2.31 * freq["Plasma cells"]
                    - 4.52 * freq["T cells gamma delta"]).values[0]
    else:
        def fun(row):
            return np.nan
    return fun


def get_EcoTyper_score(data):
    raise NotImplementedError


def _cibersort(y, S):
    nus = [0.25, 0.5, 0.75]
    results = np.zeros(2)
    coefs = np.zeros((S.shape[1], 3))
    rmses = np.zeros(3)
    r2 = np.zeros(3)
    corr = np.zeros(3)
    for j in range(3):
        regr = NuSVR(kernel='linear', nu=nus[j])
        regr.fit(S, y)
        pred = regr.predict(S)
        rmses[j] = mean_squared_error(y, pred)
        r2[j] = r2_score(y, pred)
        corr[j] = np.corrcoef(y, pred)[0, 1]
        coefs[:, j] = regr.coef_
    temp = np.argmin(rmses)
    results[0] = r2[temp]
    results[1] = corr[temp] ** 2
    freq = coefs[:, temp]
    freq = np.where(freq >= 0, freq, 0)
    freq = freq / np.sum(freq)
    return freq, results


def _xcell_aggregate(raw_scores):
    L = []
    for ind in raw_scores.index:
        L.append(ind.split('_')[0])
    raw_scores["cell types"] = L
    return raw_scores.groupby("cell types").mean()


def _xcell_transform_scores(raw_scores, min_values):
    tscores = ((raw_scores - min_values) / 5000).clip(lower=0)
    tscores = (tscores ** _DATA_Xcell_P.copy()) / (_DATA_Xcell_V.copy()*2)
    return tscores


def _xcell_spillover(tscores, K, alpha=0.5):
    K = K.loc[tscores.index, tscores.index].values
    K = K * alpha
    np.fill_diagonal(K, 1)

    def adjust_scores(x):
        res = lsq_linear(K, x, bounds=(0, np.inf))
        return res.x

    adjusted_scores = adjust_scores(tscores.values)  # .apply(adjust_scores, axis=0)
    adjusted_scores[adjusted_scores < 0] = 0
    adjusted_scores = pd.DataFrame(adjusted_scores, index=tscores.index)

    return adjusted_scores
