# Code adapted from Kang H, et al. A Comprehensive Benchmark of Transcriptomic Biomarkers for
# Immune Checkpoint Blockades. Cancers. 2023

# R source code (Kang H, et al.): https://ngdc.cncb.ac.cn/icb/resources


import pandas as pd
import numpy as np
import gseapy as gp
import json
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

import os
dirname = os.path.dirname(__file__)


_DATA_TIS = pd.read_csv(os.path.join(dirname, "data/gsea_signatures/IIS_TIS_signature.txt"), sep="\t")
_DIC_IPRES = gp.base.GSEAbase().load_gmt_only(os.path.join(dirname, "data/gsea_signatures/IPRES_signatures.gmt"))

# with open(os.path.join(dirname, "data/C_ECM.txt")) as f:
#     _DATA_ECM = [line.rstrip() for line in f]

with open(os.path.join(dirname, "data/gsea_signatures/MIAS.txt")) as f:
    _DATA_MIAS = [line.rstrip() for line in f][1:]

_DATA_IMS = pd.read_csv(os.path.join(dirname, "data/gsea_signatures/IMS_signature.txt"), sep="\t")
_DATA_IMS_META = pd.read_csv(os.path.join(dirname, "data/gsea_signatures/IMS_signature_meta.txt"), sep="\t")

with open(os.path.join(dirname, "data/gsea_signatures/PASS_PRE_Sig.JSON")) as f:
    _DIC_PASS_PRE_SIG = json.load(f)

_DATA_PASS_PRE_COEF = pd.read_csv(os.path.join(dirname, "data/gsea_signatures/PASS_PRE_Coeff.csv"), sep=";")

_DIC_MFP = gp.base.GSEAbase().load_gmt_only(os.path.join(dirname, "data/gsea_signatures/MFP.gmt"))
_DATA_MFP = pd.read_csv(os.path.join(dirname, "data/gsea_signatures/TCGA_LUAD_MFP.csv"), index_col=0)


def get_TIS_score(data):
    """ Get the TIS score

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
        TIS score for each sample.

    Notes
    -----
    Description: T cell infiltration score from nine T cell populations.
    Cancer type: clear cell Renal Cell Carcinoma
    Antibodies: anti PD-1
    References (DOIs): 10.1186/s13059-016-1092-z
    """
    temp = _DATA_TIS.copy()
    pathways = ["CD8 T cells", "T helper cells", "Tcm cells", "Tem cells", "Th1 cells", "Th2 cells", "Th17 cells",
                "Treg cells"]
    temp = temp[temp["Cell type"].isin(pathways)]
    pathways_dic = {}
    for p, d in temp.groupby("Cell type"):
        genes = set(d["Symbol"].values) & set(data.columns)
        if len(genes) > 0:
            pathways_dic[p] = list(genes)
    if len(pathways_dic) > 0:
        min_size = min(15, np.min([len(genes) for _, genes in pathways_dic.items()]))
        ss_gsea = gp.ssgsea(data=data.T, gene_sets=pathways_dic, min_size=min_size)
        results_es = ss_gsea.res2d.pivot(index="Name", columns="Term", values="NES")
        results_es.index.name = None
        results_es.columns.name = None
        scaling = StandardScaler().fit(results_es)

        def fun(row):
            gsea = gp.ssgsea(data=row.squeeze(), gene_sets=pathways_dic, min_size=min_size)
            res = gsea.res2d.pivot(index="Name", columns="Term", values="NES")
            res.index.name = None
            res.columns.name = None
            return scaling.transform(res).mean()
    else:

        def fun(row):
            return np.nan

    return fun


def get_IIS_score(data):
    """ Get the IIS score

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
        IIS score for each sample.

    Notes
    -----
    Description: Overall immune cell infiltration score from both adaptive and innate immune cell populations
    Cancer type: clear cell Renal Cell Carcinoma
    Antibodies: anti PD-1
    References (DOIs): 10.1186/s13059-016-1092-z
    """
    temp = _DATA_TIS.copy()
    pathways_dic = {}
    for p, d in temp.groupby("Cell type"):
        genes = set(d["Symbol"].values) & set(data.columns)
        if len(genes) > 0:
            pathways_dic[p] = list(genes)
    if len(pathways_dic) > 0:
        min_size = min(15, np.min([len(genes) for _, genes in pathways_dic.items()]))
        ss_gsea = gp.ssgsea(data=data.T, gene_sets=pathways_dic, min_size=min_size)
        results_es = ss_gsea.res2d.pivot(index="Name", columns="Term", values="NES")
        results_es.index.name = None
        results_es.columns.name = None
        scaling = StandardScaler().fit(results_es)

        def fun(row):
            gsea = gp.ssgsea(data=row.squeeze(), gene_sets=pathways_dic, min_size=min_size)
            res = gsea.res2d.pivot(index="Name", columns="Term", values="NES")
            res.index.name = None
            res.columns.name = None
            return scaling.transform(res).mean()
    else:

        def fun(row):
            return np.nan

    return fun


def get_APM_score(data):
    """ Get the APM score

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
        APM score for each sample.

    Notes
    -----
    Description: Seven-gene antigen presenting machinery (APM) signature that consisted of MHC class I genes
                 and genes involved in processing and loading antigens.
    Cancer type: clear cell Renal Cell Carcinoma
    Antibodies: anti PD-1
    References (DOIs): 10.1186/s13059-016-1092-z
    """
    pathways_dic = {"APM_pathway":
                    list(set(["HLA-A", "HLA-B", "HLA-C", "B2M", "TAP1", "TAP2", "TAPBP"]) & set(data.columns))}
    if len(pathways_dic["APM_pathway"]) > 0:
        min_size = len(pathways_dic["APM_pathway"])

        def fun(row):
            ss_gsea = gp.ssgsea(data=row.squeeze(), gene_sets=pathways_dic, min_size=min_size)
            res = ss_gsea.res2d.pivot(index="Name", columns="Term", values="ES")
            res.index.name = None
            res.columns.name = None
            return res.values[0, 0]
    else:

        def fun(row):
            return np.nan

    return fun


def get_IPRES_score(data):
    """ Get the IPRES score

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
        IPRES score for each sample.

    Notes
    -----
    Description: Score indicating concurrent up-expression of genes involved in the regulation of
                 mesenchymal transition, cell adhesion, extracellular matrix remodeling, angiogenesis,
                 and wound healing
    Cancer type: Multiple
    Antibodies: anti PD-1
    References (DOIs): 10.1016/j.cell.2016.02.065
    """
    pathways_dic = {}
    for pa, gl in _DIC_IPRES.items():
        temp = set(gl) & set(data.columns)
        if len(temp) > 0:
            pathways_dic[pa] = list(temp)

    if len(pathways_dic) > 0:
        min_size = min(15, np.min([len(genes) for _, genes in pathways_dic.items()]))
        ss_gsea = gp.ssgsea(data=np.log(data + 1).T, gene_sets=pathways_dic, min_size=min_size)
        results_es = ss_gsea.res2d.pivot(index="Name", columns="Term", values="NES")
        results_es.index.name = None
        results_es.columns.name = None
        scaling = StandardScaler().fit(results_es)

        def fun(row):
            gsea = gp.ssgsea(data=np.log(row.squeeze() + 1), gene_sets=pathways_dic, min_size=min_size)
            res = gsea.res2d.pivot(index="Name", columns="Term", values="NES")
            res.index.name = None
            res.columns.name = None
            return scaling.transform(res).mean()
    else:

        def fun(row):
            return np.nan

    return fun


def get_CECMup_score(data):
    """ Get the CECMup score

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
        CECMup score for each sample.

    Notes
    -----
    Description: Signature of extracellular matrix (ECM) genes upregulated in cancer tissue.
    Cancer type: Multiple
    Antibodies: anti PD-1
    References (DOIs): 10.1038/s41467-018-06654-8
    """
    ecm_up = ['MMP11', 'ADAMT', 'SERPINH1', 'COL10A1', 'COL11A1', 'MMP1', 'COL1A1', 'LOXL2', 'MMP9', 'ADAM12', 'ACAN',
              'SPP1', 'FAP', 'ADAM8', 'COL5A2', 'TIMP1', 'MMP12', 'ITGAX', 'COL5A1', 'COL7A1', 'COL5A3', 'TGFBI',
              'COMP', 'MFAP2', 'VCAN', 'COL1A2', 'COL3A1', 'SULF1', 'POSTN', 'FN1']
    pathways_dic = {"ECMup_pathway": list(set(ecm_up) & set(data.columns))}
    if len(pathways_dic["ECMup_pathway"]) > 0:
        min_size = len(pathways_dic["ECMup_pathway"])

        def fun(row):
            ss_gsea = gp.ssgsea(data=row.squeeze(), gene_sets=pathways_dic, min_size=min_size)
            res = ss_gsea.res2d.pivot(index="Name", columns="Term", values="ES")
            res.index.name = None
            res.columns.name = None
            return res.values[0, 0]
    else:

        def fun(row):
            return np.nan

    return fun


def get_CECMdown_score(data):
    """ Get the CECMdown score

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
        CECMdown score for each sample.

    Notes
    -----
    Description: Signature of extracellular matrix (ECM) genes downregulated in cancer tissue.
    Cancer type: Multiple
    Antibodies: anti PD-1
    References (DOIs): 10.1038/s41467-018-06654-8
    """
    ecm_down = ['TNXB', 'DPT', 'GPM6B', 'MFAP4', 'DCN', 'FBLN5', 'JAM2', 'MYH11', 'ABI3BP', 'RECK', 'PDGFRA', 'LAMA2',
                'CYR61', 'FGF2', 'COL4A6', 'DDR2', 'ECM2', 'COL14A1', 'A2M', 'COL4A3', 'FBLN1', 'ITGA9', 'COL4A4',
                'CTGF', 'FLRT2', 'TLL1', 'LAMC3', 'MFAP5']
    pathways_dic = {"ECMdown_pathway": list(set(ecm_down) & set(data.columns))}
    if len(pathways_dic["ECMdown_pathway"]) > 0:
        min_size = len(pathways_dic["ECMdown_pathway"])

        def fun(row):
            ss_gsea = gp.ssgsea(data=row.squeeze(), gene_sets=pathways_dic, min_size=min_size)
            res = ss_gsea.res2d.pivot(index="Name", columns="Term", values="ES")
            res.index.name = None
            res.columns.name = None
            return res.values[0, 0]
    else:

        def fun(row):
            return np.nan

    return fun


def get_IMS_score(data):
    """ Get the IMS score

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
        IMS score for each sample.

    Notes
    -----
    Description: Immune cell infiltration score for 27 immune cell populations associated with gastric cancer patients
                 prognosis
    Cancer type: Gastric cancer
    Antibodies: anti PD-L1, anti PD-1
    References (DOIs): 10.1038/s41525-021-00249-x
    """
    pathways_dic = {}
    temp_keys = []

    for pa, df in _DATA_IMS.groupby("Immune cell type"):
        temp = list(set(df["Gene"].values) & set(data.columns))
        if len(temp) > 0:
            pathways_dic[pa] = temp
            temp_keys.append(pa)
    if len(pathways_dic) > 0:
        weights = (_DATA_IMS_META.copy().
                   set_index('Immune Cell').loc[temp_keys, "HR"] > 1).replace({True: 1, False: -1})
        min_size = min(15, np.min([len(genes) for _, genes in pathways_dic.items()]))

        def fun(row):
            ss_gsea = gp.ssgsea(data=row.squeeze(), gene_sets=pathways_dic, min_size=min_size)
            res = ss_gsea.res2d.pivot(index="Name", columns="Term", values="NES").T
            res.index.name = None
            res.columns.name = None
            return (weights*res.squeeze()).sum()
    else:

        def fun(row):
            return np.nan

    return fun


def get_PASSPRE_score(data):
    """ Get the PASSPRE score

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
        PASSPRE score for each sample.

    Notes
    -----
    Description: Pathway-based signature of upregulated pathways in pre-treatment samples of immunotherapy responders.
    Cancer type: Melanoma
    Antibodies: anti PD-1
    References (DOIs): 10.1038/s41467-021-26299-4
    """
    pathways_dic = {}
    temp_keys = []
    df_weights = _DATA_PASS_PRE_COEF.copy().set_index('Name')
    for pa, genes in _DIC_PASS_PRE_SIG.items():
        temp = list(set(genes) & set(data.columns))
        if (len(temp) > 0) and (pa in df_weights.index):
            pathways_dic[pa] = temp
            temp_keys.append(pa)

    if len(pathways_dic) > 0:
        weights = df_weights.loc[temp_keys, "Weight"]
        min_size = min(15, np.min([len(genes) for _, genes in pathways_dic.items()]))

        def fun(row):
            ss_gsea = gp.ssgsea(data=row.squeeze(), gene_sets=pathways_dic, min_size=min_size)
            res = ss_gsea.res2d.pivot(index="Name", columns="Term", values="NES").T
            res.index.name = None
            res.columns.name = None
            return (weights * res.squeeze()).sum()
    else:

        def fun(row):
            return np.nan

    return fun


def get_PASSON_score(data):
    raise NotImplementedError


def get_MIAS_score(data):
    """ Get the MIAS score

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
        MIAS score for each sample.

    Notes
    -----
    Description: Pathway-based score associated with the MHC class I antigen.
    Cancer type: Melanoma
    Antibodies: anti PD-1
    References (DOIs): 10.1038/s41467-021-27651-4
    """
    pathways_dic = {"MIAS_pathway": list(set(_DATA_MIAS) & set(data.columns))}
    if len(pathways_dic["MIAS_pathway"]) > 0:
        min_size = len(pathways_dic["MIAS_pathway"])

        def fun(row):
            ss_gsea = gp.ssgsea(data=row.squeeze(), gene_sets=pathways_dic, min_size=min_size)
            res = ss_gsea.res2d.pivot(index="Name", columns="Term", values="ES")
            res.index.name = None
            res.columns.name = None
            return res.values[0, 0]
    else:

        def fun(row):
            return np.nan

    return fun


def get_MFP_score(data):
    """ Get the MFP score

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
        MFP score for each sample.

    Notes
    -----
    Description: Classification into four tumor microenvironment (TME) subtypes using 29 functionnal gene expression
                 signatures representing the major functional components and immune, stromal,
                 and other cellular populations of the tumor.
    Cancer type: Multiple
    Antibodies: anti PD-1, anti PD-L1, anti CTLA-4
    References (DOIs): 10.1016/j.ccell.2021.04.014
    """
    pathways_dic = {}
    temp_keys = []
    for pa, genes in _DIC_MFP.items():
        temp = list(set(genes) & set(data.columns))
        if len(temp) > 0:
            pathways_dic[pa] = temp
            temp_keys.append(pa)

    if len(pathways_dic) > 0:
        min_size = min(15, np.min([len(genes) for _, genes in pathways_dic.items()]))
        knn_clf = KNeighborsClassifier(n_neighbors=5).fit(_DATA_MFP.iloc[:, :29].values, _DATA_MFP["clusters"].values)

        def fun(row):
            ss_gsea = gp.ssgsea(data=row.squeeze(), gene_sets=pathways_dic, min_size=min_size)
            res = ss_gsea.res2d.pivot(index="Name", columns="Term", values="NES")
            res.index.name = None
            res.columns.name = None
            return knn_clf.predict(res.values.reshape(1, -1))[0]

    else:

        def fun(row):
            return np.nan

    return fun


def get_TIRP_score(data):
    """ Get the TIRP score

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
        TIRP score for each sample.

    Notes
    -----
    Description:
    Cancer type:
    Antibodies:
    References (DOIs):
    """
    raise NotImplementedError

