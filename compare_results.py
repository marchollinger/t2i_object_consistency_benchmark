#!/usr/bin/env python
from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from run_benchmark import aggregate_object_scores


def compare_means_grouped(df: pd.DataFrame, score_name: str) -> Figure:
    """Produces a bar plot to visualize the difference between categories.

    Args:
        df: The final benchmark results for all models merged into one DataFrame
            with columns ["prompt_id", "variation", "prompt", "category", "score", "model"].
        metric: The name of the aggregation metric used to obtain the scores per prompt.

    Returns:
        A figure containing the plot.
    """
    fig, ax = plt.subplots()
    sns.barplot(df, x="model", y="score", hue="category", ax=ax)
    ax.set_ylabel("mean " + score_name + " score")
    return fig


def box_means_grouped(df: pd.DataFrame, metric: str) -> Figure:
    """Produces a box plot to visualize the difference in mean and variation between categories.

    Args:
        df: The final benchmark results for all models merged into one DataFrame
            with columns ["prompt_id", "variation", "prompt", "category", "score", "model"].
        metric: The name of the aggregation metric used to obtain the scores per prompt.

    Returns:
        A figure containing the plot.
    """
    fig, ax = plt.subplots()
    sns.boxplot(
        df,
        x="model",
        y="score",
        hue="category",
        ax=ax,
        order=["SD2", "SD2.1", "SD3-medium", "DALL-E 3"],
    )
    ax.set_ylabel(metric + " score")
    ax.set_xlabel(None)
    return fig


def compare_models(model_output_dir: Path, out_dir: Path, metric="std"):
    """Produce plots comparing all the model results in the directory.

    Args:
        model_output_dir: Path to directory containing the model results.
        out_dir: Directory to save the plots to.
        metric: Metric to use for aggregating the variation scores. Defaults to "std".
    """
    model_dirs = [m for m in model_output_dir.glob("./*") if m.is_dir()]
    model_names = list(map(lambda x: x.parts[-1], model_dirs))

    full_df = pd.DataFrame(
        {"category": ["abstract"] * 40 + ["realistic"] * 40}, index=range(0, 80)
    )
    full_df["category"] = full_df["category"].astype("category")

    for i, model_name in enumerate(model_names):
        model_df = pd.read_csv(
            model_dirs[i] / "score" / "raw_scores.csv",
            index_col=["prompt_id", "variation"],
            delimiter="\t",
        )
        model_df = aggregate_object_scores(model_df, metric=metric)
        full_df[model_name] = model_df["score"]

    full_df = full_df.melt(id_vars=["category"], var_name="model", value_name="score")
    box_means_grouped(full_df, metric).savefig(out_dir.name + f"box_plot.pdf")
    compare_means_grouped(full_df, metric).savefig(out_dir.name + f"compare_means.pdf")


def main():

    compare_models(Path("./out"), Path("./"), metric="std")


if __name__ == "__main__":
    gpt_file = "./test_model/out/csv/all_gpt4.csv"
    # gpt4_save_figures(gpt_file, "gpt4", "./")
    main()
