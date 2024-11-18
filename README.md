# tipit_benchmark_RNA

This repository provides a Python implementation of several transcriptomic signatures that were associated with
immunotherapy response in the literature, for different cancer types and checkpoint inhibitors.

It contains the code used in our study to perform a benchmark of transcriptomic signatures to predict immunotherapy
outcome in non-small cell lung cancer:

"Integration of clinical, pathological, radiological, and transcriptomic data improves the prediction of first-line 
immunotherapy outcome in metastatic non-small cell lung cancer"

**Preprint:** [https://doi.org/10.1101/2024.06.27.24309583](https://doi.org/10.1101/2024.06.27.24309583)


**Note:** The transcriptomic signatures were selected based on the work of [Kang *et al.* 2023](https://doi.org/10.3390/cancers15164094).
## Installation

### Dependencies
- gseapy (=1.1.3)
- pandas (= 1.5.3)
- pyyaml (>= 6.0)
- scikit-learn (>= 1.2.0)

Optional (to run the scripts):
- scikit-survival (>= 0.21.0)
- tqdm (>= 4.63.0)
- xgboost (>= 1.7.5)

### Install from source

Clone the repository: 

```
git clone https://github.com/sysbio-curie/tipit_benchmark_RNA
```

## Examples 
**Define and compute a transcriptomic signature**
```python
import pandas as pd
from benchmark_RNA.signatures import get_CYT_score

data = pd.read_csv("data/transcritpomic_data.csv", index_col=0)

#1. Define the function score
CYT_fun = get_CYT_score(data)

#2. Compute the scores
CYT_scores = data.agg(CYT_fun, axis=1)
```

**Note:** *data* should be a pandas DataFrame with samples in rows and genes in columns. Columns names should be gene
symbols.   


Some signatures include training pre-processing steps in their definition such as PCA (e.g., FTBRS, TME),
scaling (e.g., TIS, IIS), or KNN (MFP). It may be required to define them and compute their values with different
datasets.

**Define and compute a transcriptomic signature with train and test data**
```python
import pandas as pd
from benchmark_RNA.signatures_gsea import get_MFP_score

data_train = pd.read_csv("data/transcritpomic_data_train.csv", index_col=0)
data_test = pd.read_csv("data/transcritpomic_data_test.csv", index_col=0)

#1. Define the function score
MFP_fun = get_MFP_score(data_train)

#2. Compute the scores
MFP_scores = data_test.agg(MFP_fun, axis=1)
```

## Scripts

We provide [a Python script](scripts/extract_signatures.py) to reproduce the benchmark of transcriptomic signatures for
the prediction of immunotherapy outcome in lung cancer in [our paper](https://doi.org/10.1101/2024.06.27.24309583). It
defines and tests the different signatures across the fold of a repeated cross-validation scheme.

## Available transcriptomic signatures


<div style="height:200px;overflow:auto;">

| Name                                                         | Signature type | Cancer type          | Immmune Checkpoint  | References                                                           |
|--------------------------------------------------------------|----------------|----------------------|---------------------|----------------------------------------------------------------------|
| [CRMA](/benchmark_RNA/signatures.py#L507)                    | Marker genes   | Melanoma             | CTLA-4              | [Shukla *et al.*](https://doi.org/10.1016/j.cell.2018.03.026)        |
| [CTLA4](/benchmark_RNA/signatures.py#L153)                   | Marker genes   | Multiple             | PD-L1               | [Herbst *et al.*](https://doi.org/10.1038/nature14011)               |
| [CX3CL1](/benchmark_RNA/signatures.py#L120)                  | Marker genes   | Multiple             | PD-L1               | [Herbst *et al.*](https://doi.org/10.1038/nature14011)               |
| [CXCL9](/benchmark_RNA/signatures.py#L252)                   | Marker genes   | Melanoma             | PD-L1               | [Qu *et al.*](https://doi.org/10.1016/j.celrep.2020.107873)          |
| [CYT](/benchmark_RNA/signatures.py#L285)                     | Marker genes   | Multiple             | PD-1, CTLA-4        | [Rooney *et al.*](https://doi.org/10.1016/j.cell.2014.12.033)        |
| [EIGS](/benchmark_RNA/signatures.py#L354)                    | Marker genes   | Multiple             | PD-1                | [Ayers *et al.*](https://doi.org/10.1172/jci91190)                   |
| [ESCS](/benchmark_RNA/signatures.py#L541)                    | Marker genes   | Urothelial cancer    | PD-1                | [Wang *et al.*](https://doi.org/10.1038/s41467-018-05992-x)          |
| [FTBRS](/benchmark_RNA/signatures.py#L575)                   | Marker genes   | Multiple             | PD-L1               | [Mariathasan *et al.*](https://doi.org/10.1038/nature25501)          |
| [HLADRA](/benchmark_RNA/signatures.py#L186)                  | Marker genes   | Melanoma             | PD-1, PD-L1         | [Johnson *et al.*](https://doi.org/10.1038/ncomms10582)              |
| [HRH1](/benchmark_RNA/signatures.py#L219)                    | Marker genes   | Multiple             | PD-1, PD-L1, CTLA-4 | [Li *et al.*](https://doi.org/10.1016/j.ccell.2021.11.002)           |
| [IFNgamma](/benchmark_RNA/signatures.py#L320)                | Marker genes   | Multiple             | PD-1                | [Ayers *et al.*](https://doi.org/10.1172/jci91190)                   |
| [Immunopheno](/benchmark_RNA/signatures.py#L428)             | Marker genes   | Multiple             | PD-1, CTLA-4        | [Charoentong *et al.*](https://doi.org/10.1016/j.celrep.2016.12.019) |
| [IMPRES](/benchmark_RNA/signatures.py#L471)                  | Marker genes   | Melanoma             | PD-1, CTLA-4        | [Auslander *et al.*](https://doi.org/10.1038/s41591-018-0157-9)      |
| [IRG](/benchmark_RNA/signatures.py#L671)                     | Marker genes   | Cervical cancer      | PD-1, PD-L1, CTLA-4 | [Yang *et al.*](https://doi.org/10.1080/2162402x.2019.1659094)       |
| [MPS](/benchmark_RNA/signatures.py#L743)                     | Marker genes   | Melanoma             | PD-1, CTLA-4        | [Pérez-Guijarro *et al.*](https://doi.org/10.1038/s41591-020-0818-3) |
| [PD1](/benchmark_RNA/signatures.py#L54)                      | Marker genes   | Multiple             | PD-1                | [Taube *et al.*](https://doi.org/10.1158/1078-0432.ccr-13-3271)      |
| [PDL1](/benchmark_RNA/signatures.py#L21)                     | Marker genes   | Multiple             | PD-1, PD-L1         | [Herbst *et al.*](https://doi.org/10.1038/nature14011)               |
| [PDL2](/benchmark_RNA/signatures.py#L87)                     | Marker genes   | Multiple             | PD-1                | [Yearley *et al.*](https://doi.org/10.1158/1078-0432.ccr-16-1761)    |
| [Renal101](/benchmark_RNA/signatures.py#L778)                | Marker genes   | Renal cell carcinoma | PD-1, PD-L1         | [Motzer *et al.*](https://doi.org/10.1038/s41591-020-1044-8)         |
| [TIG](/benchmark_RNA/signatures.py#L389)                     | Marker genes   | Multiple             | PD-1                | [Cristescu *et al.*](https://doi.org/10.1126/science.aar3593)        |
| [TLS](/benchmark_RNA/signatures.py#L709)                     | Marker genes   | Melanoma             | PD-1, CTLA-4        | [Cabrita *et al.*](https://doi.org/10.1038/s41586-019-1914-8)        | 
| [TME](/benchmark_RNA/signatures.py#L615)                     | Marker genes   | Gastric cancer       | PD-1, PD-L1, CTLA-4 | [Zeng *et al.*](https://doi.org/10.1158/2326-6066.cir-18-0436)       |
| [APM](/benchmark_RNA/signatures_gsea.py#L264)                | GSEA           | Renal cell carcinoma | PD-1                | [Senbabaoglu *et al.*](https://doi.org/10.1186/s13059-016-1092-z)    |
| [CECMdown](/benchmark_RNA/signatures_gsea.py#L404)           | GSEA           | Multiple             | PD-1                | [Chakravarthy *et al.*](https://doi.org/10.1038/s41467-018-06654-8)  |
| [CECMup](/benchmark_RNA/signatures_gsea.py#L360)             | GSEA           | Multiple             | PD-1                | [Chakravarthy *et al.*](https://doi.org/10.1038/s41467-018-06654-8)  |
| [IIS](/benchmark_RNA/signatures_gsea.py#L213)                | GSEA           | Renal cell carcinoma | PD-1                | [Senbabaoglu *et al.*](https://doi.org/10.1186/s13059-016-1092-z)    |
| [IMS](/benchmark_RNA/signatures_gsea.py#L448)                | GSEA           | Gastric cancer       | PD-1, PD-L1         | [Lin *et al.*](https://doi.org/10.1038/s41525-021-00249-x)           |
| [IPRES](/benchmark_RNA/signatures_gsea.py#L307)              | GSEA           | Multiple             | PD-1                | [Hugo *et al.*](https://doi.org/10.1016/j.cell.2016.02.065)          |
| [MFP](/benchmark_RNA/signatures_gsea.py#L474)                | GSEA           | Multiple             | PD-1, PD-L1, CTLA-4 | [Bagaev *et al.*](https://doi.org/10.1016/j.ccell.2021.04.014)       |
| [MIAS](/benchmark_RNA/signatures_gsea.py#L433)               | GSEA           | Melanoma             | PD-1                | [Wu *et al.*](https://doi.org/10.1038/s41467-021-27651-4)            |
| [PASSPRE](/benchmark_RNA/signatures_gsea.py#L379)            | GSEA           | Melanoma             | PD-1                | [Du *et al.*](https://doi.org/10.1038/s41467-021-26299-4)            |
| [TIS](/benchmark_RNA/signatures_gsea.py#L39)                 | GSEA           | Renal cell carcinoma | PD-1                | [Senbabaoglu *et al.*](https://doi.org/10.1186/s13059-016-1092-z)    |
| [CD8T_CIBERSORT](/benchmark_RNA/signatures_deconv.py#L131)   | Deconvolution  | Multiple             | PD-1                | [Tumeh *et al.*](https://doi.org/10.1038/nature13954)                |
| [CD8T_MCPcounter](/benchmark_RNA/signatures_deconv.py#L36)   | Deconvolution  | Multiple             | PD-1                | [Tumeh *et al.*](https://doi.org/10.1038/nature13954)                |
| [CD8T_Xcell](/benchmark_RNA/signatures_deconv.py#L69)        | Deconvolution  | Multiple             | PD-1                | [Tumeh *et al.*](https://doi.org/10.1038/nature13954)                |
| [Immuno_CIBERSORT](/benchmark_RNA/signatures_deconv.py#L172) | Deconvolution  | Melanoma             | PD-1                | [Nie *et al.*](https://doi.org/10.18632/aging.102556)                |


</div>

## Acknowledgements

This repository was created as part of the PhD project of [Nicolas Captier](https://ncaptier.github.io/) in the 
[Computational Systems Biologie of Cancer group](https://institut-curie.org/team/barillot) and the
[ Laboratory of Translational Imaging in Oncology (LITO)](https://www.lito-web.fr/en/) of Institut Curie.
