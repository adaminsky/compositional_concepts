from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def get_color_value(value, thresholds, quantile, inverse=False):
    if not inverse:
        for i, (threshold, q) in enumerate(zip(thresholds[::-1], quantile[::-1])):
            if value >= threshold:
                return int((1 - (i / len(thresholds))) * 100)
    else:
        for i, (threshold, q) in enumerate(zip(thresholds, quantile)):
            if value <= threshold:
                return int((1 - (i / len(thresholds))) * 100)
        return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()

    methods = ["gt", "pca", "ace", "dictlearn", "seminmf", "ct", "random", "ours"]

    results_clevr = pd.read_csv("results/clevr.csv")
    results_truth = pd.read_csv("results/truth.csv")
    results_cub = pd.read_csv("results/cub.csv")
    results = pd.concat([results_clevr, results_truth, results_cub], ignore_index=True)

    for metric in ["map", "cscore", "mean_cosim"]:
        truth_sub_map = results[
            (results["dataset"] == "truth_topics") & (results["model"] == "Llama2")
        ][[metric, "method", f"{metric}_std"]]
        clevr_map = results[
            (results["dataset"] == "CLEVR1") & (results["model"] == "CLIP")
        ][[metric, "method", f"{metric}_std"]]
        cub_sub_map = results[
            (results["dataset"] == "cub_subset") & (results["model"] == "CLIP")
        ][[metric, "method", f"{metric}_std"]]

        # remove gt from mean_cosim
        if metric == "mean_cosim":
            truth_sub_map = truth_sub_map[truth_sub_map["method"] != "gt"]
            clevr_map = clevr_map[clevr_map["method"] != "gt"]
            cub_sub_map = cub_sub_map[cub_sub_map["method"] != "gt"]

        # all_map_values = pd.concat([truth_sub_map["map"], clevr_map["map"], cub_sub_map["map"]], axis=1).to_numpy().flatten()
        truth_values = truth_sub_map[metric].to_numpy().flatten()
        clevr_values = clevr_map[metric].to_numpy().flatten()
        # clevr_values = clevr_values[clevr_values["method"].isin(methods)][metric].to_numpy().flatten()
        cub_sub_values = cub_sub_map[metric].to_numpy().flatten()
        thresholds = np.linspace(0, 1, 7, endpoint=False)
        truth_thresholds = np.percentile(truth_values, thresholds * 100)
        clevr_thresholds = np.percentile(clevr_values, thresholds * 100)
        cub_thresholds = np.percentile(cub_sub_values, thresholds * 100)

        if args.markdown:
            data = []
            for method in methods:
                if metric == "mean_cosim" and method == "gt":
                    continue
                clevr_map_val = float(
                    clevr_map[clevr_map["method"] == method][metric].iloc[0]
                )
                clevr_map_std = float(
                    clevr_map[clevr_map["method"] == method][f"{metric}_std"].iloc[0]
                )
                truth_map_val = float(
                    truth_sub_map[truth_sub_map["method"] == method][metric].iloc[0]
                )
                truth_map_std = float(
                    truth_sub_map[truth_sub_map["method"] == method][
                        f"{metric}_std"
                    ].iloc[0]
                )
                cub_map_val = float(
                    cub_sub_map[cub_sub_map["method"] == method][metric].iloc[0]
                )
                cub_map_std = float(
                    cub_sub_map[cub_sub_map["method"] == method][f"{metric}_std"].iloc[
                        0
                    ]
                )
                clevr_str = f"{clevr_map_val:.3f} $\pm$ {clevr_map_std:.3f}"
                truth_sub_str = f"{truth_map_val:.3f} $\pm$ {truth_map_std:.3f}"
                cub_sub_str = f"{cub_map_val:.3f} $\pm$ {cub_map_std:.3f}"
                data.append(
                    {
                        "CLEVR": clevr_str,
                        "CUB-sub": cub_sub_str,
                        "Truth-sub": truth_sub_str,
                    }
                )
            if metric != "mean_cosim":
                df = pd.DataFrame(data, index=methods)
            else:
                df = pd.DataFrame(data, index=methods[1:])
            print(df.to_markdown())
            print()
        else:
            output = ""
            for method in methods:
                if metric == "mean_cosim" and method == "gt":
                    continue
                clevr_map_val = float(
                    clevr_map[clevr_map["method"] == method][metric].iloc[0]
                )
                clevr_map_std = float(
                    clevr_map[clevr_map["method"] == method][f"{metric}_std"].iloc[0]
                )
                truth_map_val = float(
                    truth_sub_map[truth_sub_map["method"] == method][metric].iloc[0]
                )
                truth_map_std = float(
                    truth_sub_map[truth_sub_map["method"] == method][
                        f"{metric}_std"
                    ].iloc[0]
                )
                cub_map_val = float(
                    cub_sub_map[cub_sub_map["method"] == method][metric].iloc[0]
                )
                cub_map_std = float(
                    cub_sub_map[cub_sub_map["method"] == method][f"{metric}_std"].iloc[
                        0
                    ]
                )
                clevr_str = f"{clevr_map_val:.3f} $\pm$ {clevr_map_std:.3f}"
                truth_sub_str = f"{truth_map_val:.3f} $\pm$ {truth_map_std:.3f}"
                cub_sub_str = f"{cub_map_val:.3f} $\pm$ {cub_map_std:.3f}"
                output += (
                    f"{method} & \cellcolor{{pink!{get_color_value(clevr_map_val, clevr_thresholds, thresholds, inverse=metric=='cscore')}}} {clevr_str} & "
                    f"\cellcolor{{pink!{get_color_value(cub_map_val, cub_thresholds, thresholds, inverse=metric=='cscore')}}} {cub_sub_str} & "
                    f"\cellcolor{{pink!{get_color_value(truth_map_val, truth_thresholds, thresholds, inverse=metric=='cscore')}}} {truth_sub_str} \\\\\n"
                )
            print(metric)
            print(
                "\\begin{tabular}{lccc}\n\\toprule\n\\rowcolor{gray!20} & CLEVR &  CUB-sub   & Truth-sub \\\\"
            )
            print(output)
            print("\\bottomrule\n\\end{tabular}")
            print()


if __name__ == "__main__":
    main()
