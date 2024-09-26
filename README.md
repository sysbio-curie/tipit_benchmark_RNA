# tipit_benchmark_RNA

This repository provides a Python implementation of several transcriptomic signatures that were associated with
immunotherapy response in the literature, for different cancer types and checkpoint inhibitors.

It contains the code used in our study to perform a benchmark of transcriptomic signatures to predict immunotherapy
outcome in non-small cell lung cancer:

"Integration of clinical, pathological, radiological, and transcriptomic data improves the prediction of first-line immunotherapy outcome in metastatic non-small cell lung cancer"

**Preprint:** https://doi.org/10.1101/2024.06.27.24309583


**Note:** The transcriptomic signatures were selected based on the work of [Kang *et al.* 2023](https://doi.org/10.3390/cancers15164094).
## Installation

### Dependencies
- gseapy (=1.1.3)
- pandas (= 1.5.3)
- pyyaml (>= 6.0)
- scikit-learn (>= 1.2.0)

Optional (to run the scrits):
- scikit-survival (>= 0.21.0)
- tqdm (>= 4.63.0)
- xgboost (>= 1.7.5)

### Install from source

Clone the repository: 

```
git clone https://github.com/ncaptier/tipit_benchmark_RNA
```

## Available transcriptomic signatures

<div style="height:200px;overflow:auto;">

| Name                                 | Signature type | Cancer type          | Immmune Checkpoint  | References                                                           |
|--------------------------------------|----------------|----------------------|---------------------|----------------------------------------------------------------------|
| [CRMA](/benchmark_RNA/signatures.py) | Marker genes   | Melanoma             | CTLA-4              | [Shukla *et al.*](https://doi.org/10.1016/j.cell.2018.03.026)        |
| [CTLA4]()                            | Marker genes   | Multiple             | PD-L1               | [Herbst *et al.*](https://doi.org/10.1038/nature14011)               |
| [CX3CL1]()                           | Marker genes   | Multiple             | PD-L1               | [Herbst *et al.*](https://doi.org/10.1038/nature14011)               |
| [CXCL9]()                            | Marker genes   | Melanoma             | PD-L1               | [Qu *et al.*](https://doi.org/10.1016/j.celrep.2020.107873)          |
| [CYT]()                              | Marker genes   | Multiple             | PD-1, CTLA-4        | [Rooney *et al.*](https://doi.org/10.1016/j.cell.2014.12.033)        |
| [EIGS]()                             | Marker genes   | Multiple             | PD-1                | [Ayers *et al.*](https://doi.org/10.1172/jci91190)                   |
| [ESCS]()                             | Marker genes   | Urothelial cancer    | PD-1                | [Wang *et al.*](https://doi.org/10.1038/s41467-018-05992-x)          |
| [FTBRS]()                            | Marker genes   | Multiple             | PD-L1               | [Mariathasan *et al.*](https://doi.org/10.1038/nature25501)          |
| [HLADRA]()                           | Marker genes   | Melanoma             | PD-1, PD-L1         | [Johnson *et al.*](https://doi.org/10.1038/ncomms10582)              |
| [HRH1]()                             | Marker genes   | Multiple             | PD-1, PD-L1, CTLA-4 | [Li *et al.*](https://doi.org/10.1016/j.ccell.2021.11.002)           |
| [IFNgamma]()                         | Marker genes   | Multiple             | PD-1                | [Ayers *et al.*](https://doi.org/10.1172/jci91190)                   |
| [Immunopheno]()                      | Marker genes   | Multiple             | PD-1, CTLA-4        | [Charoentong *et al.*](https://doi.org/10.1016/j.celrep.2016.12.019) |
| [IMPRES]()                           | Marker genes   | Melanoma             | PD-1, CTLA-4        | [Auslander *et al.*](https://doi.org/10.1038/s41591-018-0157-9)      |
| [IRG]()                              | Marker genes   | Cervical cancer      | PD-1, PD-L1, CTLA-4 | [Yang *et al.*](https://doi.org/10.1080/2162402x.2019.1659094)       |
| [MPS]()                              | Marker genes   | Melanoma             | PD-1, CTLA-4        | [Pérez-Guijarro *et al.*](https://doi.org/10.1038/s41591-020-0818-3) |
| [PD1]()                              | Marker genes   | Multiple             | PD-1                | [Taube *et al.*](https://doi.org/10.1158/1078-0432.ccr-13-3271)      |
| [PDL1]()                             | Marker genes   | Multiple             | PD-1, PD-L1         | [Herbst *et al.*](https://doi.org/10.1038/nature14011)               |
| [PDL2]()                             | Marker genes   | Multiple             | PD-1                | [Yearley *et al.*](https://doi.org/10.1158/1078-0432.ccr-16-1761)    |
| [Renal101]()                         | Marker genes   | Renal cell carcinoma | PD-1, PD-L1         | [Motzer *et al.*](https://doi.org/10.1038/s41591-020-1044-8)         |
| [TIG]()                              | Marker genes   | Multiple             | PD-1                | [Cristescu *et al.*](https://doi.org/10.1126/science.aar3593)        |
| [TLS]()                              | Marker genes   | Melanoma             | PD-1, CTLA-4        | [Cabrita *et al.*](https://doi.org/10.1038/s41586-019-1914-8)        | 
| [TME]()                              | Marker genes   | Gastric cancer       | PD-1, PD-L1, CTLA-4 | [Zeng *et al.*](https://doi.org/10.1158/2326-6066.cir-18-0436)       |
| [APM]()                              | GSEA           | Renal cell carcinoma | PD-1                | [Senbabaoglu *et al.*](https://doi.org/10.1186/s13059-016-1092-z)    |
| [CECMdown]()                         | GSEA           | Multiple             | PD-1                | [Chakravarthy *et al.*](https://doi.org/10.1038/s41467-018-06654-8)  |
| [CECMup]()                           | GSEA           | Multiple             | PD-1                | [Chakravarthy *et al.*](https://doi.org/10.1038/s41467-018-06654-8)  |
| [IIS]()                              | GSEA           | Renal cell carcinoma | PD-1                | [Senbabaoglu *et al.*](https://doi.org/10.1186/s13059-016-1092-z)    |
| [IMS]()                              | GSEA           | Gastric cancer       | PD-1, PD-L1         | [Lin *et al.*](https://doi.org/10.1038/s41525-021-00249-x)           |
| [IPRES]()                            | GSEA           | Multiple             | PD-1                | [Hugo *et al.*](https://doi.org/10.1016/j.cell.2016.02.065)          |
| [MFP]()                              | GSEA           | Multiple             | PD-1, PD-L1, CTLA-4 | [Bagaev *et al.*](https://doi.org/10.1016/j.ccell.2021.04.014)       |
| [MIAS]()                             | GSEA           | Melanoma             | PD-1                | [Wu *et al.*](https://doi.org/10.1038/s41467-021-27651-4)            |
| [PASSPRE]()                          | GSEA           | Melanoma             | PD-1                | [Du *et al.*](https://doi.org/10.1038/s41467-021-26299-4)            |
| [TIS]()                              | GSEA           | Renal cell carcinoma | PD-1                | [Senbabaoglu *et al.*](https://doi.org/10.1186/s13059-016-1092-z)    |
| [CD8T_CIBERSORT]()                   | Deconvolution  | Multiple             | PD-1                | [Tumeh *et al.*](https://doi.org/10.1038/nature13954)                |
| [CD8T_MCPcounter]()                  | Deconvolution  | Multiple             | PD-1                | [Tumeh *et al.*](https://doi.org/10.1038/nature13954)                |
| [CD8T_Xcell]()                       | Deconvolution  | Multiple             | PD-1                | [Tumeh *et al.*](https://doi.org/10.1038/nature13954)                |
| [Immuno_CIBERSORT]()                 | Deconvolution  | Melanoma             | PD-1                | [Nie *et al.*](https://doi.org/10.18632/aging.102556)                |


</div>

## Examples 

## Acknowledgements

This repository was created as part of the PhD project of Nicolas Captier in the 
[Computational Systems Biologie of Cancer group](https://institut-curie.org/team/barillot) and the [ Laboratory of Translational Imaging in Oncology (LITO)](https://www.lito-web.fr/en/) of Institut Curie.