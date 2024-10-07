#!/usr/bin/env python
from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from run_benchmark import aggregate_object_scores


def gpt4_save_figures(data_file, score_name, out_dir):
    full_df = pd.read_csv(data_file, index_col="prompt_id", delimiter="\t")
    num_df = full_df.loc[:, full_df.columns != "prompt"]
    line_plot(num_df, score_name, out_dir)
    compare_means(num_df, score_name, out_dir)


def box_plot(df, score_name, out_path):
    fig, ax = plt.subplots()
    sns.boxplot(df, ax=ax)
    ax.set_ylabel(score_name + "_score")
    fig.savefig(out_path + f"box_plot_{score_name}.pdf")


def compare_var(df, score_name, out_path):
    fig, ax = plt.subplots()
    df = df.mean(axis=0)
    x = list(df.index)
    sns.barplot(x=x, y=df, ax=ax)
    ax.set_ylabel(score_name + "_score")
    fig.savefig(out_path + f"var_plot_{score_name}.pdf")


def compare_means(df, score_name, out_path):
    fig, ax = plt.subplots()
    df = df.mean(axis=0)
    x = list(df.index)
    sns.barplot(x=x, y=df, ax=ax)
    ax.set_ylabel(score_name + " score")
    fig.savefig(out_path + f"mean_plot_{score_name}.pdf")


def line_plot(df, metric):
    df = df
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(15, 10))
    # sns.lineplot(df["SD2"], ax=ax)
    x_axis = range(df.shape[0])
    for column in df:
        if "real" in column:
            sns.lineplot(x=x_axis, y=df[column], label=column, marker="o", ax=ax1)
        else:
            sns.lineplot(x=x_axis, y=df[column], label=column, marker="o", ax=ax2)
    ax1.set_title("real")
    ax1.set_ylabel(metric + "_score")
    ax1.legend(loc="upper right")
    ax2.set_title("simple")
    ax2.set_xlabel("prompt_id")
    ax2.set_ylabel("mean " + metric + "_score")
    return fig


def comparative_line_plot(df, metric):
    df = stack_df(df)
    n_plots = (df.shape[1] // 2) + 1
    fig, axes = plt.subplots(nrows=n_plots, ncols=1, figsize=(15, 5 * n_plots))
    for i, m in enumerate(["SD1-5", "SD2", "SD2-1"]):
        model_df = df[df["model"] == m]
        sns.lineplot(
            model_df, x="prompt_id", y="score", hue="task_type", marker="o", ax=axes[i]
        )
        axes[i].set_title(m)
        axes[i].set_xlabel("")
        axes[i].set_ylabel(metric + " score")
    axes[-1].set_xlabel("prompt_id")
    fig.tight_layout()
    return fig


def compare_means_grouped(df, score_name):
    fig, ax = plt.subplots()
    sns.barplot(df, x="model", y="score", hue="category", ax=ax)
    ax.set_ylabel("mean " + score_name + " score")
    return fig


def box_means_grouped(df, metric):
    fig, ax = plt.subplots()
    sns.boxplot(df, x="model", y="score", hue="category", ax=ax, order=["SD2", "SD2.1", "SD3-medium", "DALL-E 3"])
    ax.set_ylabel(metric + " score")
    ax.set_xlabel(None)
    return fig


def save_figures(data_file, metric, out_dir):
    full_df = pd.read_csv(data_file, index_col="prompt_id", delimiter="\t")
    num_df = full_df.loc[:, full_df.columns != "prompt"]

    # line_plot(num_df, "align", out_dir).savefig(out_dir + f"line_{metric}_plot.pdf")
    compare_means_grouped(num_df, metric).savefig(out_dir + f"bar_{metric}_plot.pdf")
    box_means_grouped(num_df, metric).savefig(out_dir + f"box_{metric}_plot.pdf")
    comparative_line_plot(num_df, metric).savefig(out_dir + f"line_{metric}_plot.pdf")


def stack_df(df):
    arrays = [
        ["SD1-5", "SD1-5", "SD2", "SD2", "SD2-1", "SD2-1"],
        ["simple", "real"] * 3,
    ]
    index = pd.MultiIndex.from_arrays(arrays, names=["model", "task_type"])
    df.columns = index
    df = df.stack().stack().to_frame().reset_index()
    df = df.rename(columns={0: "score"})
    return df


def compare_models(model_output_dir: Path, out_dir: Path, metric="std"):
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
    # clip_file = "./test_model/out/csv/all_clip.csv"
    # save_figures(clip_file, "clip", "./")
    # metrics = ["min", "mean", "std"]
    # for metric in metrics:
    #     align_file = f"./test_model/out/real-simple/{metric}_align.csv"
    #     save_figures(align_file, metric, "./plots/")

    compare_models(Path("./out"), Path("./"), metric="std")


if __name__ == "__main__":
    gpt_file = "./test_model/out/csv/all_gpt4.csv"
    # gpt4_save_figures(gpt_file, "gpt4", "./")
    main()
