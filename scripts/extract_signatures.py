import argparse
import yaml
import inspect
import os
import sys
import pandas as pd
from sksurv.util import Surv
from sksurv.ensemble import RandomSurvivalForest

from sklearn.metrics import roc_auc_score

from joblib import Parallel, delayed
from tqdm.auto import tqdm

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from benchmark_RNA.utils.custom_models import CustomXGBoostClassifier
import benchmark_RNA.signatures as sig
import benchmark_RNA.signatures_gsea as sig_gsea
import benchmark_RNA.signatures_deconv as sig_deconv


def main(params):
    config = read_yaml(params.config)

    func_sig = inspect.getmembers(sig, inspect.isfunction)
    func_gsea = inspect.getmembers(sig_gsea, inspect.isfunction)
    func_deconv = inspect.getmembers(sig_deconv, inspect.isfunction)
    funcs = func_sig + func_gsea + func_deconv

    data_RNA = pd.read_csv(config["data_RNA_path"], index_col=0).iloc[:, 1:]
    data_repeats = pd.read_csv(config["data_repeats_path"]).set_index("samples")
    common_index = list(set(data_RNA.index) & set(data_repeats.index))

    data_repeats = data_repeats.loc[common_index]
    data_RNA = data_RNA.loc[common_index]

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

    results = pd.concat(results_parallel, axis=0)
    save_path = os.path.join(config["save_dir"], config["save_name"])
    results.to_csv(save_path)

    return


def _extract_repeat(data_RNA, data_repeats, function_list, ml_score, repeat, classif=True):
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

                # print("Train AUC: ", roc_auc_score(labels, clf.predict_proba(signatures_train.copy().values)[:, 1]))
                # print("Test AUC: ", roc_auc_score(data_folds.loc[signatures_test.index, "label"].values,
                #                                   clf.predict_proba(signatures_test.copy().values)[:, 1]))

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

        # print("Test AUC bis: ", roc_auc_score(signatures_test["label"].values, signatures_test["ml_score"].values))
        # print(signatures_test[["ml_score", "label"]])

        signatures_test["repeat"] = repeat

        list_extract.append(signatures_test)

    return pd.concat(list_extract, axis=0)


def _get_function_list(data, raw_function_list):
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


def read_yaml(fname):
    with open(fname) as yaml_file:
        return yaml.safe_load(yaml_file)


class ProgressParallel(Parallel):
    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Late fusion")
    args.add_argument(
        "-c",
        "--config",
        type=str,
        help="config file path",
    )

    main(params=args.parse_args())

