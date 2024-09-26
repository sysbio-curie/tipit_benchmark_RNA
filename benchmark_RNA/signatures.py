# Code adapted from Kang H, et al. A Comprehensive Benchmark of Transcriptomic Biomarkers for
# Immune Checkpoint Blockades. Cancers. 2023

# R source code (Kang H, et al.): https://ngdc.cncb.ac.cn/icb/resources


import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

dirname = os.path.dirname(__file__)

_DATA_immunopheno = pd.read_csv(os.path.join(dirname, "data/marker_signatures/IPS_genes.txt"), sep="\t", index_col=0)
_DATA_TME = pd.read_csv(os.path.join(dirname, "data/marker_signatures/TMEScore.txt"), sep="\t", index_col=0)
_DATA_MPS = pd.read_csv(os.path.join(dirname, "data/marker_signatures/MPS_gene_list.txt"), sep="\t")


def get_PDL1_score(data):
    """ Get the PDL1 scoring function

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
        PDL1 score for each sample.

    Notes
    -----
    Description: Expression of PD-L1
    Cancer type: Multiple
    Antibodies: anti PD-1, anti PD-L1
    References (DOIs): 10.1038/nature14011
    """
    gene = "CD274"
    if gene in data.columns:
        def fun(row):
            return row[gene]
    else:
        def fun(row):
            return np.nan
    return fun


def get_PD1_score(data):
    """ Get the PD1 scoring function

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
        PD1 score for each sample.

    Notes
    -----
    Description: Expression of PD-1
    Cancer type: Multiple
    Antibodies: anti PD-1
    References (DOIs): 10.1158/1078-0432.ccr-13-3271
    """
    gene = "PDCD1"
    if gene in data.columns:
        def fun(row):
            return row[gene]
    else:
        def fun(row):
            return np.nan
    return fun


def get_PDL2_score(data):
    """ Get the PDL2 scoring function

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
        PDL2 score for each sample.

    Notes
    -----
    Description: Expression of PD-L2
    Cancer type: Multiple
    Antibodies: anti PD-1
    References (DOIs): 10.1158/1078-0432.ccr-16-1761
    """
    gene = "PDCD1LG2"
    if gene in data.columns:
        def fun(row):
            return row[gene]
    else:
        def fun(row):
            return np.nan
    return fun


def get_CX3CL1_score(data):
    """ Get the CX3CL1 scoring function

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
        CX3CL1 score for each sample.

    Notes
    -----
    Description: Expression of CX3CL1
    Cancer type: Multiple
    Antibodies: anti PD-L1
    References (DOIs): 10.1038/nature14011
    """
    gene = "CX3CL1"
    if gene in data.columns:
        def fun(row):
            return row[gene]
    else:
        def fun(row):
            return np.nan
    return fun


def get_CTLA4_score(data):
    """ Get the CTLA4 scoring function

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
        CTLA4 score for each sample.

    Notes
    -----
    Description: Expression of CTLA-4
    Cancer type: Multiple
    Antibodies: anti PD-L1
    References (DOIs): 10.1038/nature14011
    """
    gene = "CTLA4"
    if gene in data.columns:
        def fun(row):
            return row[gene]
    else:
        def fun(row):
            return np.nan
    return fun


def get_HLADRA_score(data):
    """ Get the HLA-DRA score

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
        HLA-DRA score for each sample.

    Notes
    -----
    Description: Expression of HLA-DRA
    Cancer type: Melanoma
    Antibodies: anti PD-L1, anti-PD-1
    References (DOIs): 10.1038/ncomms10582
    """
    gene = "HLA-DRA"
    if gene in data.columns:
        def fun(row):
            return row[gene]
    else:
        def fun(row):
            return np.nan
    return fun


def get_HRH1_score(data):
    """ Get the HRH1 score

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
        HRH1 score for each sample.

    Notes
    -----
    Description: Expression of HRH1
    Cancer type: Multiple
    Antibodies: anti PD-L1, anti PD-1, anti CTLA-4
    References (DOIs): 10.1016/j.ccell.2021.11.002
    """
    gene = "HRH1"
    if gene in data.columns:
        def fun(row):
            return row[gene]
    else:
        def fun(row):
            return np.nan
    return fun


def get_CXCL9_score(data):
    """ Get the CXCL9 score

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
        CXCL9 score for each sample.

    Notes
    -----
    Description: Expression of CXCL9
    Cancer type: Melanoma
    Antibodies: anti PD-L1
    References (DOIs): 10.1016/j.celrep.2020.107873
    """
    gene = "CXCL9"
    if gene in data.columns:
        def fun(row):
            return row[gene]
    else:
        def fun(row):
            return np.nan
    return fun


def get_CYT_score(data):
    """ Get the CYT score

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
        CYT score for each sample.

    Notes
    -----
    Description: Quantitative measure of immune cytolytic activity based on transcript levels of two key cytolytic
                 effectors, granzyme A (GZMA) and perforin (PRF1).
    Cancer type: Multiple
    Antibodies: anti PD-1, anti-CTLA4
    References (DOIs): 10.1016/j.cell.2014.12.033
    """
    genes = ["GZMA", "PRF1"]
    genes = set(genes) & set(data.columns)
    if len(genes) > 0:
        def fun(row):
            return row[list(genes)].mean()
    else:
        def fun(row):
            return np.nan
    return fun


def get_IFNgamma_score(data):
    """ Get the IFNgamma score

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
        IFNgamma score for each sample.

    Notes
    -----
    Description: Gene expression signature associated with IFN-gamma signaling.
    Cancer type: Multiple
    Antibodies: anti PD-1
    References (DOIs): 10.1172/jci91190
    """
    genes = ["IDO1", "CXCL10", "CXCL9", "HLA-DRA", "STAT1", "IFNG"]
    genes = set(genes) & set(data.columns)
    if len(genes) > 0:
        def fun(row):
            return row[list(genes)].mean()
    else:
        def fun(row):
            return np.nan
    return fun


def get_EIGS_score(data):
    """ Get the EIGS score

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
        EIGS score for each sample.

    Notes
    -----
    Description: Gene expression signature associated with 18 immune-related genes (Expanded Immune Gene Signature).
    Cancer type: Multiple
    Antibodies: anti PD-1
    References (DOIs): 10.1172/jci91190
    """
    genes = ["CD3D", "IDO1", "CIITA", "CD3E", "CCL5", "GZMK", "CD2", "HLA-DRA", "CXCL13", "IL2RG", "NKG7", "HLA-E",
             "CXCR6", "LAG3", "TAGAP", "CXCL10", "STAT1", "GZMB"]
    genes = set(genes) & set(data.columns)
    if len(genes) > 0:
        def fun(row):
            return row[list(genes)].mean()
    else:
        def fun(row):
            return np.nan
    return fun


def get_TIG_score(data):
    """ Get the TIG score

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
        TIG score for each sample.

    Notes
    -----
    Description: T cell–inflamed gene expression profile
    Cancer type: Multiple
    Antibodies: anti PD-1
    References (DOIs): 10.1172/jci91190, 10.1126/science.aar3593
    """
    genes = ["TIGIT", "PDCD1LG2", "CD27", "CD8A", "LAG3", "CD274", "CXCR6", "CMKLR1", "NKG7", "CCL5", "PSMB10", "IDO1",
             "CXCL9", "HLA-DQA1", "CD276", "STAT1", "HLA-DRB1", "HLA-E"]
    weights = pd.Series([0.084767, 0.003734, 0.072293, 0.031021, 0.123895, 0.042853, 0.004313, 0.151253,
                         0.075524, 0.008346, 0.032999, 0.060679, 0.074135, 0.020091, -0.0239, 0.250229,
                         0.058806, 0.07175],
                        index=genes)
    genes = list(set(genes) & set(data.columns))
    if len(genes) > 0:
        def fun(row):
            return (weights[genes].values * row[genes]).sum()
    else:
        def fun(row):
            return np.nan
    return fun


def get_Immunopheno_score(data):
    """ Get the Immunopheno score

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
        Immunopheno score for each sample.

    Notes
    -----
    Description: Scoring of tumor immunogenicity (i.e., ability of a tumor to induce an immune response).
    Cancer type: Multiple
    Antibodies: anti PD-1, anti CTLA-4
    References (DOIs): 10.1016/j.celrep.2016.12.019
    """
    ips_genes = _DATA_immunopheno.copy()
    ips_genes = ips_genes.loc[list(set(ips_genes.index) & set(data.columns))]
    if len(ips_genes) > 0:
        def fun(row):
            gene_expr = row[ips_genes.index]
            gene_expr = (gene_expr - gene_expr.mean()) / gene_expr.std()
            weights, expr = [], []
            for _, d in ips_genes.groupby("NAME"):
                expr.append(gene_expr[d.index].mean())
                weights.append(d["WEIGHT"].mean())
            weighted_expr = np.array(weights) * np.array(expr)
            return (np.mean(weighted_expr[:10]) + np.mean(weighted_expr[10:20]) + np.mean(weighted_expr[20:24])
                    + np.mean(weighted_expr[24:26])
                    )
    else:
        def fun(row):
            return np.nan
    return fun


def get_IMPRES_score(data):
    """ Get the IMPRES score

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
        IMPRES score for each sample.

    Notes
    -----
    Description: Score based on the pairwise relation between the expression of immune checkpoint genes.
    Cancer type: Melanoma
    Antibodies: anti PD-1, anti CTLA-4
    References (DOIs): 10.1038/s41591-018-0157-9
    """
    gene_pairs = [("PDCD1", "TNFSF4"), ("CD27", "PDCD1"), ("CTLA4", "TNFSF4"), ("CD40", "CD28"), ("CD86", "TNFSF4"),
                  ("CD28", "CD86"), ("CD80", "TNFSF9"), ("CD274", "VSIR"), ("CD86", "HAVCR2"), ("CD40", "PDCD1"),
                  ("CD86", "CD200"), ("CD40", "CD80"), ("CD28", "CD276"), ("CD40", "CD274"), ("TNFRSF14", "CD86")]
    gene_pairs = [p for p in gene_pairs if (p[0] in data.columns) & (p[1] in data.columns)]
    if len(gene_pairs) > 0:
        def fun(row):
            return sum([row[p[0]] < row[p[1]] for p in gene_pairs])
    else:
        def fun(row):
            return np.nan
    return fun


def get_CRMA_score(data):
    """ Get the CRMA score

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
        CRMA score for each sample.

    Notes
    -----
    Description: Expression of cancer germline antigens.
    Cancer type: Melanoma
    Antibodies: anti CTLA-4
    References (DOIs): 10.1016/j.cell.2018.03.026
    """
    genes = ["MAGEA3", "CSAG3", "CSAG2", "MAGEA2", "MAGEA2B", "CSAG1", "MAGEA12", "MAGEA6"]
    genes = set(genes) & set(data.columns)
    if len(genes) > 0:
        def fun(row):
            return row[list(genes)].mean()
    else:
        def fun(row):
            return np.nan
    return fun


def get_ESCS_score(data):
    """ Get the ESCS score

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
        ESCS score for each sample.

    Notes
    -----
    Description: EMT- and stroma-related gene expression.
    Cancer type: Urothelial
    Antibodies: anti PD-1
    References (DOIs): 10.1038/s41467-018-05992-x
    """
    genes = ["FLNA", "EMP3", "CALD1", "FN1", "FOXC2", "LOX", "FBN1", "TNC"]
    genes = set(genes) & set(data.columns)
    if len(genes) > 0:
        def fun(row):
            return row[list(genes)].mean()
    else:
        def fun(row):
            return np.nan
    return fun


def get_FTBRS_score(data):
    """ Get the FTBRS score

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
        FTBRS score for each sample.
    Notes
    -----
    Description: Gene expression signature associated with TGFβ signaling in fibroblasts.
    Cancer type: Multiple
    Antibodies: anti PD-L1
    References (DOIs): 10.1038/nature25501
    """
    genes = ["ACTA2", "ACTG2", "ADAM12", "ADAM19", "CNN1", "COL4A1", "CCN2", "CTPS1",
             "RFLNB", "FSTL3", "HSPB1", "IGFBP3", "PXDC1", "SEMA7A", "SH3PXD2A", "TAGLN",
             "TGFBI", "TNS1", "TPM1"]
    genes = set(genes) & set(data.columns)
    if len(genes) > 0:
        scaling = StandardScaler()
        data_scaled = scaling.fit_transform(data[list(genes)].values)
        pca = PCA().fit(data_scaled)

        def fun(row):
            return pca.transform(scaling.transform(row[list(genes)].values.reshape(1, -1)))[0, 0]
    else:
        def fun(row):
            return np.nan

    return fun


def get_TME_score(data):
    """ Get the TME score

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
        TME score for each sample.

    Notes
    -----
    Description: Characterization of the tumor microenvironnment (TME) infiltration.
    Cancer type: Gastric cancer
    Antibodies: anti PD-L1, anti PD-1, anti CTLA-4
    References (DOIs): 10.1158/2326-6066.cir-18-0436
    """
    tme_gene_immune = _DATA_TME[_DATA_TME["TME-signature-group"] == "TME-gene-A"].index
    tme_gene_stroma = _DATA_TME[_DATA_TME["TME-signature-group"] == "TME-gene-B"].index

    tme_gene_immune = list(set(tme_gene_immune) & set(data.columns))
    tme_gene_stroma = list(set(tme_gene_stroma) & set(data.columns))

    if (len(tme_gene_immune) > 0) & (len(tme_gene_stroma) > 0):
        # Filter genes with zero expression
        temp = data[tme_gene_immune + tme_gene_stroma].T.std(axis=1)
        temp = temp[temp > 0]
        tme_gene_stroma = list(set(tme_gene_stroma) & set(temp.index))
        tme_gene_immune = list(set(tme_gene_immune) & set(temp.index))
        # Scale data
        scaling = StandardScaler()
        data_scaled = pd.DataFrame(scaling.fit_transform(data[tme_gene_immune + tme_gene_stroma].values),
                                   index=data.index,
                                   columns=tme_gene_immune + tme_gene_stroma)
        # Train PCA
        pca_immune = PCA().fit(data_scaled[tme_gene_immune].values)
        pca_stroma = PCA().fit(data_scaled[tme_gene_stroma].values)

        def fun(row):
            row_scaled = scaling.transform(row[tme_gene_immune + tme_gene_stroma].values.reshape(1, -1))

            return (pca_immune.transform(row_scaled[:, :len(tme_gene_immune)])[0, 0]
                    - pca_stroma.transform(row_scaled[:, len(tme_gene_immune):])[0, 0]
                    )
    else:
        def fun(row):
            return np.nan
    return fun


def get_IRG_score(data):
    """ Get the IRG score

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
        IRG score for each sample.

    Notes
    -----
    Description: Weighted expression of immune-related genes (IRG).
    Cancer type: Cervical cancer
    Antibodies: anti PD-L1, anti PD-1, anti CTLA-4
    References (DOIs): 10.1080/2162402x.2019.1659094
    """
    genes = ["LEPR", "PRLHR", "NR2F2", "PRL", "NRP1", "ANGPTL5", "IGF1", "TNFRSF10B", "TNFRSF10A", "PLAU",
             "IFI30"]
    weights = pd.Series([0.32196, -0.64921, -0.32677, 0.23573, 0.39005, 0.38166, -0.03522, 0.02975, 0.39830, 0.14607,
                         -0.68625],
                        index=genes)
    genes = list(set(genes) & set(data.columns))
    if len(genes) > 0:
        def fun(row):
            return (weights[genes].values * row[genes]).sum()
    else:
        def fun(row):
            return np.nan
    return fun


def get_TLS_score(data):
    """ Get the TLS score

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
        TLS score for each sample.

    Notes
    -----
    Description: Gene signature associated with tertiary lymphoid structures (TLS).
    Cancer type: Melanoma
    Antibodies: anti PD-1, anti CTLA-4
    References (DOIs): 10.1038/s41586-019-1914-8
    """
    genes = ["CD79B", "CD1D", "CCR6", "LAT", "SKAP1", "CETP", "EIF1AY", "RBP5", "PTGDS"]
    genes = set(genes) & set(data.columns)
    if len(genes) > 0:
        def fun(row):
            return row[list(genes)].mean()
    else:
        def fun(row):
            return np.nan
    return fun


def get_MPS_score(data):
    """ Get the MPS score

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
        MPS score for each sample.

    Notes
    -----
    Description: Gene expression signature of melanocytic plasticity (extent of differentiation of melanocytic lineage).
    Cancer type: Melanoma
    Antibodies: anti PD-1, anti CTLA-4
    References (DOIs): 10.1038/s41591-020-0818-3
    """
    genes = list(_DATA_MPS["Gene Symbol"].values)
    weights = _DATA_MPS[["Sign in the signature", "Gene Symbol"]].set_index('Gene Symbol').squeeze()
    genes = list(set(genes) & set(data.columns))
    if len(genes) > 0:
        def fun(row):
            return (weights[genes] * row[genes]).sum()
    else:
        def fun(row):
            return np.nan
    return fun


def get_Renal101_score(data):
    """ Get the Renal101 score

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
        Renal101 score for each sample.

    Notes
    -----
    Description: Expression of genes involved in innate immune responses, cell trafficking, and inflammation.
    Cancer type: Renal Cell Carcinoma
    Antibodies: anti PD-L1, anti PD-1
    References (DOIs): 10.1038/s41591-020-1044-8
    """
    genes = ["CD3G", "CD3E", "CD8B", "THEMIS", "TRAT1", "GRAP2", "CD247", "CD2", "CD96", "PRF1", "CD6", "IL7R",
             "ITK", "GPR18", "EOMES", "SIT1", "NLRC3", "CD244", "KLRD1", "SH2D1A", "CCL5", "XCL2", "CST7", "GFI1",
             "KCNA3", "PSTPIP1"]
    genes = set(genes) & set(data.columns)
    if len(genes) > 0:
        def fun(row):
            return row[list(genes)].mean()
    else:
        def fun(row):
            return np.nan
    return fun


def get_TIDE_score(data):
    """ Get the TIDE score

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
        TIDE score for each sample.

    Notes
    -----
    Description:
    Cancer type:
    Antibodies:
    References (DOIs):
    """
    raise NotImplementedError
