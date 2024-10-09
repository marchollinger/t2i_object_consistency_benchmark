#!/usr/bin/env python
import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils import IModel
from utils.sim_scores import ALIGNScore, CLIPScore
from utils.t2i_variants import DALL_E, SD


def aggregate_object_scores(df: pd.DataFrame, metric: str = "std") -> pd.DataFrame:
    """Use a metric to aggregate the variations for each prompt.

    Args:
        df: DataFrame with raw benchmark results.
        metric:  Metric to use.

    Returns:
        A DataFrame containing a aggregated score per prompt.
    """
    agg_dict = {
        "prompt": "first",
        "category": "first",
        "score": metric,
    }
    df = df.groupby(level=0).agg(agg_dict)
    return df


def plot_result(final_df: pd.DataFrame, out_dir: Path) -> None:
    """Produces a line plot and a comparative bar plot for one model.

    Args:
        final_df: Final benchmark results.
        out_dir: Where to save the plots.
    """
    # Line plot before aggregating to show variation
    fig, ax = plt.subplots()
    sns.lineplot(
        final_df,
        x="prompt_id",
        y="score",
        hue="category",
        marker="o",
        ax=ax,
    )
    ax.set_ylabel("faithfulness")
    fig.savefig(out_dir / "line.pdf")

    # aggregate scores
    final_df = aggregate_object_scores(final_df)

    # Compare means
    fig, ax = plt.subplots()
    sns.barplot(final_df, x="category", y="score", hue="category")
    ax.set_ylabel("mean " + "variability")
    fig.savefig(out_dir / "compare_means.pdf")


def generate_and_score(
    df: pd.DataFrame, img_dir: Path, scorer: CLIPScore | ALIGNScore
) -> None:
    """Run the benchmark and add scores to the DataFrame in place.

    Args:
        df: DataFrame containing the prompts and variations with MultiIndex ["prompt_id", "variation"].
        img_dir: Where to save the generated images.
        scorer: Text-to-image similarity score to use.
    """
    os.makedirs(img_dir, exist_ok=True)
    for i, prompt in tqdm(df["prompt"].items(), total=df.shape[0]):
        img_path = img_dir / (str(i[0]) + "_" + str(i[1]) + ".png")
        model.invoke(prompt).save(img_path, format="png")
        df.loc[i, "score"] = scorer.calculate_score(img_path, prompt)


def main(
    prompts: Path, out_dir: Path, model: IModel, scorer: ALIGNScore | CLIPScore
) -> None:
    """Run the benchmark, save final_result and visualization.

    Args:
        prompts: Path to a tsv file containing the columns ["prompt_id", "variation", "prompt", "category"].
        out_dir: Output folder for the results, the directory will be populated with a folder containing
            the benchmark results.
        model: The model to be tested
        scorer: text-to-image similarity score to use.
    """

    prompt_df = pd.read_csv(
        prompts,
        index_col=["prompt_id", "variation"],
        header=0,
        delimiter="\t",
    )

    # run the benchmark
    img_dir = out_dir / model.model_name / "generated_images"
    os.makedirs(img_dir, exist_ok=True)
    csv_dir = out_dir / model.model_name / "score"
    os.makedirs(csv_dir, exist_ok=True)
    generate_and_score(prompt_df, img_dir, scorer)

    # save raw scores for visualization
    prompt_df.to_csv(csv_dir / "raw_scores.csv", sep="\t")

    # save plots for single model
    plot_result(prompt_df, csv_dir)

    # save the final score to a file
    prompt_df = aggregate_object_scores(prompt_df)
    diff = (
        prompt_df.loc[prompt_df["category"] == "realistic", "score"].mean()
        - prompt_df.loc[prompt_df["category"] == "abstract", "score"].mean()
    )
    with open(csv_dir / "final_score.txt", "w+") as f:
        print(
            diff,
            file=f,
        )

    return prompt_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run_benchmark.py",
        description="Run the object consistency benchmark for a given model.",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="stabilityai/stable-diffusion-2",
        required=True,
        help="hugging face model-ID",
    )
    parser.add_argument(
        "--test_prompts",
        type=Path,
        default="./data/abstract_vs_realistic.csv",
        help="path to a csv file containing prompts and prompt-variations"
        '(default: "data/abstract_vs_realistic.csv")',
    )
    parser.add_argument(
        "--score_name",
        type=str,
        default="align",
        help='score tetote use (default: "align")',
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        type=Path,
        default="./",
        help='where to save results (default: "./")',
    )
    args = parser.parse_args()
    if args.score_name == "clip":
        scorer = CLIPScore()
    elif args.score_name == "align":
        scorer = ALIGNScore()
    else:
        print("\n\n[Error] Could not find a similarity metric by this name")
        sys.exit(1)

    if args.model_id in ("dalle-e-2", "dall-e-3"):
        model = DALL_E(args.model_id)
    else:
        model = SD(args.model_id)

    main(args.test_prompts, args.out_dir, model, scorer)
