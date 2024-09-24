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


# def pass_on(data):
#     name = "PASS-ON"
#     return None, name


def get_MIAS_score(data):
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
    raise NotImplementedError

