import argparse
import inspect
import os
import sys

import pandas as pd
from joblib import delayed
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

from _utils import read_yaml, ProgressParallel, CustomXGBoostClassifier

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import benchmark_RNA.signatures as sig
import benchmark_RNA.signatures_gsea as sig_gsea
import benchmark_RNA.signatures_deconv as sig_deconv


def main(params):
    """
    Collect values of transcriptomic signatures for immunotherapy outcome across multiple repetitions of a
    validation scheme.
    """

    # 0. Load configuration file
    config = read_yaml(params.config)

    # 1. Get all the signatures from the benchmark_RNA module
    func_sig = inspect.getmembers(sig, inspect.isfunction)
    func_gsea = inspect.getmembers(sig_gsea, inspect.isfunction)
    func_deconv = inspect.getmembers(sig_deconv, inspect.isfunction)
    funcs = func_sig + func_gsea + func_deconv

    # 2. Load transcriptomic data and fold indices for the different cross-validation schemes
    data_RNA = pd.read_csv(config["data_RNA_path"], index_col=0).iloc[:, 1:]
    data_repeats = pd.read_csv(config["data_repeats_path"]).set_index("samples")
    common_index = list(set(data_RNA.index) & set(data_repeats.index))

    data_repeats = data_repeats.loc[common_index]
    data_RNA = data_RNA.loc[common_index]

    # 3. Define and test signatures across the different repetitions of cross-validation (with parallel computing)
    parallel = ProgressParallel(
        n_jobs=config["n_jobs_repeats"],
        total=len(data_repeats["repeat"].unique()),
    )

    results_parallel = parallel(delayed(_extract_repeat)(
        data_RNA,
        data_repeats,
        funcs,
        config["ml_score"],
        r,
        config["classif"]
    )
                                for r in data_repeats["repeat"].unique()
                                )

    # 4. Collect and save results
    results = pd.concat(results_parallel, axis=0)
    save_path = os.path.join(config["save_dir"], config["save_name"])
    results.to_csv(save_path)
    return


def _extract_repeat(data_RNA, data_repeats, function_list, ml_score, repeat, classif=True):
    """ Compute the values of the transcriptomic signatures for a single repetition of a cross-validation scheme.

    Parameters
    ----------
    data_RNA: pandas DataFrame
        Transcriptomic data to train and test the signatures on.
            - Samples are in row and genes in columns.
            - Column names should be gene symbols.

    data_repeats: pandas DataFrame
        DataFrame containing information for each repetition of the cross-validation scheme, specifying the test fold
        assignment for each sample. It includes the following columns:
            - **repeat**: Index indicating which repetition of the cross-validation scheme the entry belongs to.
            - **fold_index**: Index specifying the test fold to which each sample is assigned for a given repeat.
            - **label**: Binary classification label for each sample.

        For survival prediction tasks, the **label** column is replaced by:
            - **label.event**: Indicates the occurrence of the event of interest (e.g., death, progression).
            - **label.time**: Time until the event or censoring.

    function_list: List of callables
        List of functions of the form get_name_score from the benchmark_RNA module

    ml_score: boolean
        If True, a machine learning model is trained to predict the target variable using the computed transcriptomic
        signatures as input.

    repeat: int
        Index indicating a specific repetition of the cross-validation scheme.

    classif: boolean, default=True
        Specifies the type of prediction task:
            - If True, the task is binary classification.
            - If False, the task is survival prediction.

    Returns
    -------
    pandas DataFrame
        DataFrame containing the values of the different transcriptomic signatures for each sample.
        For each sample, the test fold it belongs to in the cross-validation scheme is considered (for the repeat of
        interest). The signatures are defined/fitted on the corresponding training set and applied to the sample.
    """
    data_folds = data_repeats[data_repeats["repeat"] == repeat]
    set_indexes = set(data_RNA.index)
    list_extract = []
    for fold, df in data_folds.groupby("fold_index"):
        test_samples = df.index
        train_samples = list(set_indexes - set(test_samples))
        data_train = data_RNA.copy().loc[train_samples]
        data_test = data_RNA.copy().loc[test_samples]

        trained_functions = _get_function_list(data=data_train, raw_function_list=function_list)

        signatures_test = data_test.agg(trained_functions, axis=1)

        if ml_score:
            signatures_train = data_train.agg(trained_functions, axis=1)
            if classif:
                clf = CustomXGBoostClassifier(n_jobs=1)
                labels = data_folds.loc[signatures_train.index, "label"].values
                clf.fit(signatures_train.values, labels)
                signatures_test["ml_score"] = clf.predict_proba(signatures_test.copy().values)[:, 1]
            else:
                clf_surv = RandomSurvivalForest(max_depth=6)
                labels_surv = Surv().from_arrays(
                    event=data_folds.loc[signatures_train.index, "label.event"].values,
                    time=data_folds.loc[signatures_train.index, "label.time"].values,
                )
                clf_surv.fit(signatures_train.values, labels_surv)
                signatures_test["ml_score"] = clf_surv.predict(signatures_test.copy().values)

        signatures_test["fold_index"] = fold

        if classif:
            signatures_test["label"] = data_folds.loc[signatures_test.index, "label"]
        else:
            signatures_test["label.time"] = data_folds.loc[signatures_test.index, "label.time"]
            signatures_test["label.event"] = data_folds.loc[signatures_test.index, "label.event"]

        signatures_test["repeat"] = repeat

        list_extract.append(signatures_test)

    return pd.concat(list_extract, axis=0)


def _get_function_list(data, raw_function_list):
    """ Extract all the functions of the form get_name_score from raw_function_list and apply them to transcriptomic
    data to define the different signatures.

    Parameters
    ----------
    data: pandas DataFrame
        A DataFrame of shape (n_samples, n_genes) containing transcriptomic data used to define and fit the signature.
            - Samples are in row and genes in columns.
            - Column names should be gene symbols.

    raw_function_list: List of callables

    Returns
    -------
    List of callables
        List of defined/fitted transcriptomic signatures
    """
    t = []
    for f in raw_function_list:
        if f[0].split("_")[0] == "get":
            try:
                fun = f[1](data)
            except NotImplementedError:
                pass
            else:
                fun.__name__ = '_'.join(f[0].split('_')[1:-1])
                t.append(fun)
    return t


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Late fusion")
    args.add_argument(
        "-c",
        "--config",
        type=str,
        help="config file path",
    )

    main(params=args.parse_args())
