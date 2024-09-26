# Code adapted from Kang H, et al. A Comprehensive Benchmark of Transcriptomic Biomarkers for
# Immune Checkpoint Blockades. Cancers. 2023

# R source code (Kang H, et al.): https://ngdc.cncb.ac.cn/icb/resources


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

_DATA_Xcell_coef = pd.read_excel(os.path.join(dirname, "data/deconv_signatures/Xcell_coef.xlsx"), engine="openpyxl",
                                 sheet_name=0, index_col=0)
_DATA_Xcell_P = _DATA_Xcell_coef["Power coefficient"]
_DATA_Xcell_V = _DATA_Xcell_coef["Calibration parameter"]
_DATA_Xcell_K = _DATA_Xcell_coef.iloc[:, 3:]


def get_CD8T_MCPcounter_score(data):
    """ Get the CD8T_MCPcounter score

    Parameters
    ----------
    data: pandas DataFrame
        A DataFrame of shape (n_samples, n_genes) containing transcriptomic data used to define and fit the signature.
            - Samples are in row and genes in columns.
            - Column names should be gene symbols.

    Returns
    -------
    fun: Callable
        Function to apply to each row (i.e., sample) of a pandas DataFrame (shape samples x genes) to compute
        CD8T_MCPcounter score for each sample.

    Notes
    -----
    Description: Estimation of the proportion of CD8+ T cell with MCP counter deconvolution method.
    Cancer type: Multiple
    Antibodies: anti PD-1
    References (DOIs): 10.1038/nature13954
    """
    gene = "CD8B"
    if gene in data.columns:
        def fun(row):
            return np.log(row[gene] + 1)
    else:
        def fun(row):
            return np.nan
    return fun


def get_CD8T_Xcell_score(data):
    """ Get the CD8T_Xcell score

    Parameters
    ----------
    data: pandas DataFrame
        A DataFrame of shape (n_samples, n_genes) containing transcriptomic data used to define and fit the signature.
            - Samples are in row and genes in columns.
            - Column names should be gene symbols.

    Returns
    -------
    fun: Callable
        Function to apply to each row (i.e., sample) of a pandas DataFrame (shape samples x genes) to compute
        CD8T_Xcell score for each sample.

    Notes
    -----
    Here calibration parameters, power coefficients, and spillover compensation matrix were set for sequencing-based
    data. For microarray-based data you'll need to modify the preamble of this file and load a new _Data_Xcell_coef
    file with the following code:

    _DATA_Xcell_coef = pd.read_excel(os.path.join(dirname, "data/deconv_signatures/Xcell_coef.xlsx"), engine="openpyxl",
                                 sheet_name=1, index_col=0)

    Description: Estimation of the proportion of CD8+ T cell with Xcell deconvolution method.
    Cancer type: Multiple
    Antibodies: anti PD-1
    References (DOIs): 10.1038/nature13954
    """
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
    """ Get the CD8T_CIBERSORT score

    Parameters
    ----------
    data: pandas DataFrame
        A DataFrame of shape (n_samples, n_genes) containing transcriptomic data used to define and fit the signature.
            - Samples are in row and genes in columns.
            - Column names should be gene symbols.

    Returns
    -------
    fun: Callable
        Function to apply to each row (i.e., sample) of a pandas DataFrame (shape samples x genes) to compute
        CD8T_CIBERSORT score for each sample.

    Notes
    -----
    Description: Estimation of the proportion of CD8+ T cell with CIBERSORT deconvolution method.
    Cancer type: Multiple
    Antibodies: anti PD-1
    References (DOIs): 10.1038/nature13954
    """
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
    """ Get the Immuno_CIBERSORT score

    Parameters
    ----------
    data: pandas DataFrame
        A DataFrame of shape (n_samples, n_genes) containing transcriptomic data used to define and fit the signature.
            - Samples are in row and genes in columns.
            - Column names should be gene symbols.

    Returns
    -------
    fun: Callable
        Function to apply to each row (i.e., sample) of a pandas DataFrame (shape samples x genes) to compute
        Immuno_CIBERSORT score for each sample.

    Notes
    -----
    Description: Weighted sum of the proportion of 8 immune subsets estimated with CIBERSORT deconvolution method.
    Cancer type: Melanoma
    Antibodies: anti PD-1
    References (DOIs): 10.1038/nature13954
    """
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
    """ Get the EcoTyper score

    Parameters
    ----------
    data: pandas DataFrame
        A DataFrame of shape (n_samples, n_genes) containing transcriptomic data used to define and fit the signature.
            - Samples are in row and genes in columns.
            - Column names should be gene symbols.

    Returns
    -------
    fun: Callable
        Function to apply to each row (i.e., sample) of a pandas DataFrame (shape samples x genes) to compute
        EcoTyper score for each sample.

    Notes
    -----
    Description:
    Cancer type:
    Antibodies:
    References (DOIs):
    """
    raise NotImplementedError


def _cibersort(y, S):
    """ Fit nu-support vector regressions (ν-SVR) (with different values of nu coefficient) to estimate the abundance
    of different cell types within a sample (CIBERSORT method, see references).

    Parameters
    ----------
    y: 1D array-like
        Expression of CIBERSORT marker genes for one sample (shape (m,) with m the number of marker genes)

    S: 2D array-like
        Signature/basis matrix containing the expression of marker genes for different cell types (shape (m, c) with
        m the number of marker genes and c the number of cell types)

    Returns
    -------
    freq: 1D numpy array
        Estimated frequencies for CIBERSORT cell types for the sample of interest (shape (c,)

    results: 1D numpy array
        Coefficients of determination for the best fit for nu support vector regression (ν-SVR).

    Notes
    -----
    References (DOIs): 10.1038/nmeth.3337, 10.1038/s41587-019-0114-2
    """
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
    """ Average the ssGSEA score of gene signatures associated to the same cell type

    Parameters
    ----------
    raw_scores: pandas DataFrame
        Dataframe containing the ssGSEA scores of the 489 Xcell gene signatures for each sample. Gene signatures are
        in rows and samples in columns (shape 489 x N, with N number of samples). Row names should be the Xcell gene
        signatures (e.g., )

    Returns
    -------
    pandas DataFrame
        DataFrame containing the average score for each of the 64 cell types and each sample (shape 64 x N, with N
        number of samples)

    Notes
    -----
    Reference (DOI): 10.1186/s13059-017-1349-1
    """
    L = []
    for ind in raw_scores.index:
        L.append(ind.split('_')[0])
    raw_scores["cell types"] = L
    return raw_scores.groupby("cell types").mean()


def _xcell_transform_scores(raw_scores, min_values):
    """ Transorm cell types score

    Parameters
    ----------
    raw_scores: pandas DataFrame
        Scores for each cell type and each sample (shape 64 x N, with N the number of samples)

    min_values: pandas DataFrame
        Min values for the scores of each cell type across the different samples (shape 64 x 1)

    Returns
    -------
    tscores: pandas DataFrame
        Transformed scores (shape 64 x N, with N the number of samples)

    Notes
    -----
    Here calibration parameters and power coefficients were set for sequencing-based data. For microarray-based data
    you'll need to modify the preamble of this file and load a new _Data_Xcell_coef file with the following code:

    _DATA_Xcell_coef = pd.read_excel(os.path.join(dirname, "data/deconv_signatures/Xcell_coef.xlsx"), engine="openpyxl",
                                 sheet_name=1, index_col=0)

    Reference (DOI): 10.1186/s13059-017-1349-1
    """
    tscores = ((raw_scores - min_values) / 5000).clip(lower=0)
    tscores = (tscores ** _DATA_Xcell_P.copy()) / (_DATA_Xcell_V.copy()*2)
    return tscores


def _xcell_spillover(tscores, K, alpha=0.5):
    """ Adjust transformed Xcell scores with spillover compensation for each sample (i.e., adjust for spillover between
     dependent cell types)

    Parameters
    ----------
    tscores: pandas DataFrame
        Transformed Xcell scores of one sample (size 64 x 1)

    K: pandas DataFrame
        Compensation coefficients (shape 64 x 64). Column and row names should correspond to Xfer cell types.

    alpha: float
        Scaling parameter to multiply all off-diagonal coefficient of K (avoid over compensation). The default is 0.5
        (see reference below).

    Returns
    -------
    adjusted_scores: pandas DataFrame
        Ajusted Xcell scores of one sample to compensate spillover effect (shape 64 x 1).

    Notes
    -----
    Reference (DOI): 10.1186/s13059-017-1349-1
    """
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
