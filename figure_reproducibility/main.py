from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from statsmodels.formula.api import ols


SCORE_COLUMNS = ["MedGPT1", "MedGPT2", "MedGPT3", "MedGPT4", "MedGPT5", "synonyms_lemmatized_f1_dict"]
ALL_MEDGPT = ["MedGPT1", "MedGPT2", "MedGPT3", "MedGPT4", "MedGPT5"]
MODEL_ORDER_SUMMARY = [
    "Med42-70B", "Meditron 3-70B", "NVLM-70B", "Claude 3.5 Sonnet",
    "Llama 3.1-8B-Instruct", "Llama 3.1-8B-Instruct RAG", "Llama 3.3-70B-Instruct",
    "Llama 3.2-1B-Instruct", "Gemini 2.5 Pro", "Gemma 3 27B", "MedGemma 27B",
    "MedGemma 27B RAG", "Rephrased Gold Answers"
]
MODEL_ORDER_PLOT_ALL = [
    "Rephrased Gold Answers", "Gemini 2.5 Pro", "MedGemma 27B", "Claude 3.5 Sonnet",
    "Llama 3.3-70B-Instruct", "Med42-70B", "Meditron 3-70B", "Gemma 3 27B",
    "Llama 3.1-8B-Instruct", "NVLM-70B", "Llama 3.2-1B-Instruct"
]
MODEL_ORDER_PLOT_MAIN = [
    "Gemini 2.5 Pro", "MedGemma 27B", "Claude 3.5 Sonnet", "Llama 3.3-70B-Instruct",
    "Med42-70B", "Meditron 3-70B", "Gemma 3 27B", "Llama 3.1-8B-Instruct", "NVLM-70B"
]
MODEL_ORDER_RAG_PLOT = [
    "MedGemma 27B", "MedGemma 27B RAG", "Llama 3.1-8B-Instruct", "Llama 3.1-8B-Instruct RAG"
]
RAG_PLOT_COLORS = ["darkgreen", "mediumseagreen", "darkblue", "lightblue"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cleaned version of code_plots_v6.py with centralized inputs/outputs."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("input"),
        help="Folder containing the input CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Base output folder. Creates csv/ and plots/ inside it.",
    )
    return parser.parse_args()


def ensure_dirs(base_output_dir: Path) -> tuple[Path, Path]:
    csv_dir = base_output_dir / "csv"
    plot_dir = base_output_dir / "plots"
    csv_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    return csv_dir, plot_dir


def save_csv(df: pd.DataFrame, csv_dir: Path, name: str) -> None:
    df.to_csv(csv_dir / f"data_{name}.csv", index=False)


def save_plot(fig: plt.Figure, plot_dir: Path, filename: str) -> None:
    fig.savefig(plot_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_inputs(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    supervised_path = input_dir / "all_questions_answers_scores.csv"
    unsupervised_path = input_dir / "all_questions_answers_scores_unsupervised.csv"

    missing = [str(p) for p in [supervised_path, unsupervised_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required input CSV file(s):\n- " + "\n- ".join(missing)
        )

    answers = pd.read_csv(supervised_path)
    unsupervised = pd.read_csv(unsupervised_path)
    return answers, unsupervised


def numeric_cleanup(answers: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    answers = answers.copy()
    answers[SCORE_COLUMNS] = answers[SCORE_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0)
    mask = (answers[SCORE_COLUMNS] != 0).any(axis=1)
    answer_filtered = answers[mask].copy()
    answer_removed = answers[~mask].copy()
    return answers, answer_filtered, answer_removed


def category_merge_5_to_4(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.loc[out["category_id"] == 5, "category_id"] = 4
    return out


def make_ordered(df: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    out = df.copy()
    out["model"] = pd.Categorical(out["model"], categories=order, ordered=True)
    return out.sort_values("model")


def count_sentences(text: str) -> float:
    if pd.isna(text):
        return np.nan
    return float(len(re.findall(r"[.!?]+", str(text))))


def lower_upper(mean: np.ndarray, std: np.ndarray, n: int | None = None, use_95ci: bool = True) -> tuple[np.ndarray, np.ndarray]:
    if use_95ci:
        if n is None or n <= 0:
            raise ValueError("n must be provided and > 0 when use_95ci=True")
        ci = 1.96 * std / np.sqrt(n)
        return mean - ci, mean + ci
    return mean - std, mean + std


def paired_permutation_test(diffs: np.ndarray, n_perm: int = 20000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[~np.isnan(diffs)]
    n = len(diffs)
    if n == 0:
        return np.nan
    observed = np.mean(diffs)
    signs = rng.choice([-1, 1], size=(n_perm, n))
    perm_means = (signs * diffs).mean(axis=1)
    p_val = (np.sum(np.abs(perm_means) >= abs(observed)) + 1) / (n_perm + 1)
    return float(p_val)


def bootstrap_ci(diffs: np.ndarray, n_boot: int = 20000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[~np.isnan(diffs)]
    n = len(diffs)
    if n == 0:
        return np.nan, np.nan
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = diffs[idx].mean(axis=1)
    low, high = np.percentile(boot, [2.5, 97.5])
    return float(low), float(high)


def analysis_model_summary(answer_filtered: pd.DataFrame, csv_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_category_avg = (
        answer_filtered.groupby(["model", "category_id", "iteration_number"])[SCORE_COLUMNS]
        .mean()
        .reset_index()
    )
    per_iteration_avg = (
        per_category_avg.groupby(["model", "iteration_number"])[SCORE_COLUMNS]
        .mean()
        .reset_index()
    )
    summary = per_iteration_avg.groupby("model")[SCORE_COLUMNS].agg(["mean", "std"])
    summary.columns = ["_".join(col) for col in summary.columns]
    summary = summary.reset_index()
    summary = make_ordered(summary, MODEL_ORDER_SUMMARY)

    save_csv(per_category_avg, csv_dir, "model_scores_summary_per_category_avg")
    save_csv(per_iteration_avg, csv_dir, "model_scores_summary_per_iteration_avg")
    save_csv(summary, csv_dir, "model_scores_summary")
    return per_iteration_avg, summary


def analysis_significance_vs_reference(per_iteration_avg: pd.DataFrame, reference_model: str, csv_dir: Path, name: str) -> pd.DataFrame:
    medgpt1_pivot = per_iteration_avg.pivot(index="iteration_number", columns="model", values="MedGPT1")
    results = []
    ref = medgpt1_pivot[reference_model]
    for other in medgpt1_pivot.columns.drop(reference_model):
        comp = medgpt1_pivot[other]
        mask = ref.notna() & comp.notna()
        t_stat, p_val = stats.ttest_rel(ref[mask], comp[mask])
        results.append({
            "comparison_model": other,
            "t_statistic": t_stat,
            "p_value": p_val,
        })
    signif_df = pd.DataFrame(results)
    signif_df["p_bonf"] = np.minimum(signif_df["p_value"] * len(signif_df), 1.0)
    save_csv(medgpt1_pivot.reset_index(), csv_dir, f"{name}_pivot")
    save_csv(signif_df, csv_dir, name)
    return signif_df


def analysis_column_means(summary: pd.DataFrame, csv_dir: Path) -> pd.Series:
    excluded_models = ["Rephrased Gold Answers", "Llama 3.2-1B-Instruct"]
    summary_filtered = summary[~summary["model"].isin(excluded_models)].copy()
    column_means = summary_filtered.select_dtypes(include=[np.number]).mean()
    column_means_df = column_means.rename_axis("metric").reset_index(name="value")
    save_csv(summary_filtered, csv_dir, "model_scores_summary_filtered_for_means")
    save_csv(column_means_df, csv_dir, "model_scores_summary_column_means")
    return column_means


def figure_gpt_and_f1_bars(answers: pd.DataFrame, csv_dir: Path, plot_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(42)
    answers_plot = category_merge_5_to_4(answers)
    gpt_score_columns = ["MedGPT1", "MedGPT2", "MedGPT3", "MedGPT4", "MedGPT5"]
    f1_score_column = "synonyms_lemmatized_f1_dict"
    iter_scores = (
        answers_plot.groupby(["model", "category_id", "iteration_number"])[gpt_score_columns + [f1_score_column]]
        .mean()
        .reset_index()
    )
    summary = (
        iter_scores.groupby(["model", "category_id"])[gpt_score_columns + [f1_score_column]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = ["model", "category_id"] + [
        f"{score}_{stat}" for score in gpt_score_columns + [f1_score_column] for stat in ["mean", "std"]
    ]
    categories = [1, 2, 3]
    summary = summary[summary["category_id"].isin(categories)].copy()
    iter_scores = iter_scores[iter_scores["category_id"].isin(categories)].copy()

    save_csv(iter_scores, csv_dir, "gpt_scores_bars_iter_scores")
    save_csv(summary, csv_dir, "gpt_scores_bars_summary")

    sns.set(style="whitegrid")
    sns.set_context("talk")
    x = np.arange(len(categories))
    width = 0.083
    colors = sns.color_palette("tab10", len(MODEL_ORDER_PLOT_ALL))
    model_color_dict = dict(zip(MODEL_ORDER_PLOT_ALL, colors))

    fig_gpt, axes_gpt = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
    score_labels = ["MedGPT1", "MedGPT2", "MedGPT3"]
    for i, score in enumerate(score_labels):
        ax = axes_gpt[i]
        for j, model in enumerate(MODEL_ORDER_PLOT_ALL):
            means, stds = [], []
            for cat in categories:
                row = summary[(summary["model"] == model) & (summary["category_id"] == cat)]
                if not row.empty:
                    means.append(row[f"{score}_mean"].values[0])
                    stds.append(row[f"{score}_std"].values[0])
                else:
                    means.append(np.nan)
                    stds.append(np.nan)
            ax.bar(
                x + j * width, means, yerr=stds, width=width,
                label=model if i == 0 else "", color=model_color_dict[model],
                capsize=3, alpha=0.7,
            )
            for k, cat in enumerate(categories):
                subset = iter_scores[(iter_scores["model"] == model) & (iter_scores["category_id"] == cat)]
                if not subset.empty:
                    y_vals = subset[score].values
                    x_center = x[k] + j * width
                    jitter = rng.uniform(-width / 4, width / 4, size=len(y_vals))
                    ax.scatter(np.full_like(y_vals, x_center) + jitter, y_vals, color=model_color_dict[model], alpha=0.5, s=15)
        ax.set_xticks(x + width * (len(MODEL_ORDER_PLOT_ALL) - 1) / 2)
        ax.set_xticklabels([f"Category {cat}" for cat in categories])
        ax.set_title(score)
        ax.set_ylabel("MedGPT Score" if i == 0 else "")
    handles, labels = axes_gpt[0].get_legend_handles_labels()
    fig_gpt.legend(handles, labels, title="Model", loc="lower center", bbox_to_anchor=(0.5, -0.3), ncol=3, frameon=False)
    plt.tight_layout()
    save_plot(fig_gpt, plot_dir, "gpt_scores_bars.png")

    fig_f1, ax_f1 = plt.subplots(figsize=(8, 6))
    for j, model in enumerate(MODEL_ORDER_PLOT_ALL):
        means, stds = [], []
        for cat in categories:
            row = summary[(summary["model"] == model) & (summary["category_id"] == cat)]
            if not row.empty:
                means.append(row[f"{f1_score_column}_mean"].values[0])
                stds.append(row[f"{f1_score_column}_std"].values[0])
            else:
                means.append(np.nan)
                stds.append(np.nan)
        ax_f1.bar(x + j * width, means, yerr=stds, width=width, label=model, color=model_color_dict[model], capsize=3, alpha=0.7)
        for k, cat in enumerate(categories):
            subset = iter_scores[(iter_scores["model"] == model) & (iter_scores["category_id"] == cat)]
            if not subset.empty:
                y_vals = subset[f1_score_column].values
                x_center = x[k] + j * width
                jitter = rng.uniform(-width / 4, width / 4, size=len(y_vals))
                ax_f1.scatter(np.full_like(y_vals, x_center) + jitter, y_vals, color=model_color_dict[model], alpha=0.5, s=15)
    ax_f1.set_xticks(x + width * (len(MODEL_ORDER_PLOT_ALL) - 1) / 2)
    ax_f1.set_xticklabels([f"Category {cat}" for cat in categories])
    ax_f1.set_ylabel("MedSynF1")
    handles, labels = ax_f1.get_legend_handles_labels()
    fig_f1.legend(handles, labels, title="Model", loc="lower center", bbox_to_anchor=(0.5, -0.3), ncol=3, frameon=False)
    plt.tight_layout()
    save_plot(fig_f1, plot_dir, "f1_scores_bars.png")

    summary_table = summary[summary["category_id"].isin([1, 2, 3])].copy()
    summary_table_rounded = summary_table.copy()
    for col in summary_table_rounded.columns:
        if any(stat in col for stat in ["mean", "std"]):
            summary_table_rounded[col] = summary_table_rounded[col].round(3)
    summary_table_rounded = summary_table_rounded.sort_values(by=["model", "category_id"])
    summary_table_export = summary_table_rounded.iloc[:, :-2]
    summary_table_export = summary_table_export.iloc[:-3, :].reset_index(drop=True)
    save_csv(summary_table_rounded, csv_dir, "summary_table_rounded_full")
    save_csv(summary_table_export, csv_dir, "summary_table_rounded")

    return iter_scores, summary, summary_table_export


def figure_rag_gpt_scores(
    answers: pd.DataFrame,
    csv_dir: Path,
    plot_dir: Path
) -> None:
    rng = np.random.default_rng(49)

    # Merge category 5 into 4
    answers_plot = answers.copy()
    answers_plot.loc[answers_plot["category_id"] == 5, "category_id"] = 4

    # Columns
    gpt_score_columns = ["MedGPT1", "MedGPT2", "MedGPT3", "MedGPT4", "MedGPT5"]
    f1_score_column = "synonyms_lemmatized_f1_dict"

    # STEP 1
    iter_scores = (
        answers_plot.groupby(["model", "category_id", "iteration_number"])[gpt_score_columns + [f1_score_column]]
        .mean()
        .reset_index()
    )

    # STEP 2
    summary = (
        iter_scores.groupby(["model", "category_id"])[gpt_score_columns + [f1_score_column]]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary.columns = ["model", "category_id"] + [
        f"{score}_{stat}"
        for score in gpt_score_columns + [f1_score_column]
        for stat in ["mean", "std"]
    ]

    categories = [1, 2, 3]
    summary = summary[summary["category_id"].isin(categories)]
    iter_scores = iter_scores[iter_scores["category_id"].isin(categories)]

    iter_scores.to_csv(csv_dir / "rag_iter_scores.csv", index=False)
    summary.to_csv(csv_dir / "rag_summary.csv", index=False)

    sns.set(style="whitegrid")
    sns.set_context("talk")

    model_order = [
        "MedGemma 27B", "MedGemma 27B RAG",
        "Llama 3.1-8B-Instruct", "Llama 3.1-8B-Instruct RAG"
    ]

    colors = ["darkgreen", "mediumseagreen", "darkblue", "lightblue"]
    model_color_dict = dict(zip(model_order, colors))

    x = np.arange(len(categories))
    width = 0.11

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)

    score_labels = ["MedGPT1", "MedGPT2", "MedGPT3"]

    for i, score in enumerate(score_labels):
        ax = axes[i]

        for j, model in enumerate(model_order):
            means, stds = [], []

            for cat in categories:
                row = summary[
                    (summary["model"] == model) &
                    (summary["category_id"] == cat)
                ]

                if not row.empty:
                    means.append(row[f"{score}_mean"].values[0])
                    stds.append(row[f"{score}_std"].values[0])
                else:
                    means.append(np.nan)
                    stds.append(np.nan)

            # 🔹 More transparent bars
            ax.bar(
                x + j * width,
                means,
                yerr=stds,
                width=width,
                label=model if i == 0 else "",
                color=model_color_dict[model],
                capsize=3,
                alpha=0.7,
                zorder=1  # bars at the back
            )

            # Draw error bars separately ON TOP
            ax.errorbar(
                x + j * width,
                means,
                yerr=stds,
                fmt='none',
                ecolor='black',
                elinewidth=1.5,
                capsize=4,
                zorder=5  # highest → always on top
            )

            # Jittered points
            for k, cat in enumerate(categories):
                subset = iter_scores[
                    (iter_scores["model"] == model) &
                    (iter_scores["category_id"] == cat)
                ]

                if not subset.empty:
                    y_vals = subset[score].values
                    x_center = x[k] + j * width

                    jitter = rng.uniform(-width / 4, width / 4, size=len(y_vals))

                    ax.scatter(
                        np.full_like(y_vals, x_center) + jitter,
                        y_vals,
                        color=model_color_dict[model],
                        alpha=0.8,           # stronger visibility
                        s=30,                # slightly larger
                        #edgecolor="black",   # contrast edge
                        #linewidth=0.3,
                        zorder=3             # draw on top of bars
                    )

        ax.set_xticks(x + width * (len(model_order) - 1) / 2)
        ax.set_xticklabels([f"Category {cat}" for cat in categories])
        ax.set_title(score)
        ax.set_ylabel("MedGPT Score" if i == 0 else "")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Model",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.3),
        ncol=3,
        frameon=False
    )

    plt.tight_layout()
    fig.savefig(plot_dir / "rag_gpt_scores.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def figure_average_medgpt_line(summary: pd.DataFrame, iter_scores: pd.DataFrame, csv_dir: Path, plot_dir: Path) -> None:
    rng = np.random.default_rng(43)
    filtered = summary[summary["category_id"].isin([1, 2, 3])].copy()
    iter_filtered = iter_scores[iter_scores["category_id"].isin([1, 2, 3])].copy()
    filtered["MedGPT_avg"] = filtered[[f"MedGPT{i}_mean" for i in range(1, 6)]].mean(axis=1)
    iter_filtered["MedGPT_avg"] = iter_filtered[[f"MedGPT{i}" for i in range(1, 6)]].mean(axis=1)
    filtered["category_label"] = filtered["category_id"].apply(lambda x: f"Category {x}")
    iter_filtered["category_label"] = iter_filtered["category_id"].apply(lambda x: f"Category {x}")
    category_positions = {1: 0, 2: 1, 3: 2}

    save_csv(filtered, csv_dir, "average_medgpt_line_summary")
    save_csv(iter_filtered, csv_dir, "average_medgpt_line_iter_scores")

    plt.figure(figsize=(12, 7))
    sns.set(style="whitegrid")
    colors = sns.color_palette("tab10", len(MODEL_ORDER_PLOT_MAIN))
    model_color_dict = dict(zip(MODEL_ORDER_PLOT_MAIN, colors))
    for model in MODEL_ORDER_PLOT_MAIN:
        data = filtered[filtered["model"] == model]
        iter_data = iter_filtered[iter_filtered["model"] == model]
        if data.empty:
            continue
        x_vals = [category_positions[c] for c in data["category_id"]]
        plt.plot(x_vals, data["MedGPT_avg"], marker="o", label=model, color=model_color_dict[model])
        for cat, group in iter_data.groupby("category_id"):
            x_center = category_positions[cat]
            y_vals = group["MedGPT_avg"].values
            jitter = rng.uniform(-0.08, 0.08, size=len(y_vals))
            plt.scatter(np.full_like(y_vals, x_center) + jitter, y_vals, color=model_color_dict[model], alpha=0.5, s=20)
    plt.xticks([0, 1, 2], ["Category 1", "Category 2", "Category 3"])
    plt.ylabel("Average MedGPT Score")
    plt.ylim(0, 5)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="Model", frameon=False)
    plt.tight_layout()
    fig = plt.gcf()
    save_plot(fig, plot_dir, "average_medgpt_score_line.png")


def compute_dimension_trend_inputs(answers: pd.DataFrame, csv_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, int, list[str], list[int]]:
    answers_plot = category_merge_5_to_4(answers)
    categories = [1, 2, 3]
    dims = ["MedGPT1", "MedGPT2", "MedGPT3", "MedGPT4", "MedGPT5"]
    iter_scores = answers_plot.groupby(["model", "category_id", "iteration_number"])[dims].mean().reset_index()
    iter_scores = iter_scores[iter_scores["category_id"].isin(categories)].copy()
    summary = iter_scores.groupby(["model", "category_id"])[dims].agg(["mean", "std"]).reset_index()
    summary.columns = ["model", "category_id"] + [f"{dim}_{stat}" for dim in dims for stat in ["mean", "std"]]
    n_iter = int(iter_scores["iteration_number"].nunique())
    save_csv(iter_scores, csv_dir, "medgpt_dimension_trends_iter_scores")
    save_csv(summary, csv_dir, "medgpt_dimension_trends_summary")
    return iter_scores, summary, n_iter, dims, categories


def figure_dimension_trends_v1(
    iter_scores: pd.DataFrame,
    summary: pd.DataFrame,
    n_iter: int,
    dims: list[str],
    categories: list[int],
    plot_dir: Path
) -> None:
    rng = np.random.default_rng(44)
    use_95ci = False
    category_positions = {1: 0, 2: 1, 3: 2}

    colors = sns.color_palette("tab10", len(MODEL_ORDER_PLOT_MAIN))
    model_color_dict = dict(zip(MODEL_ORDER_PLOT_MAIN, colors))

    # 🔥 SAME LAYOUT AS v3
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 6, figure=fig)

    axes = [
        fig.add_subplot(gs[0, 0:2]),
        fig.add_subplot(gs[0, 2:4]),
        fig.add_subplot(gs[0, 4:6]),
        fig.add_subplot(gs[1, 1:3]),
        fig.add_subplot(gs[1, 3:5]),
    ]

    for ax, dim in zip(axes, dims):
        for model in MODEL_ORDER_PLOT_MAIN:
            d = summary[summary["model"] == model].sort_values("category_id")
            iter_d = iter_scores[iter_scores["model"] == model]

            if d.empty:
                continue

            x = np.array([category_positions[c] for c in d["category_id"]])
            mean = d[f"{dim}_mean"].to_numpy()
            std = d[f"{dim}_std"].to_numpy()

            lo, hi = lower_upper(mean, std, n=n_iter, use_95ci=use_95ci)

            ax.plot(
                x, mean,
                marker="o",
                linewidth=2,
                label=model,
                color=model_color_dict[model]
            )

            ax.fill_between(
                x, lo, hi,
                alpha=0.15,
                color=model_color_dict[model]
            )

            # jittered points
            for cat in categories:
                subset = iter_d[iter_d["category_id"] == cat]
                if not subset.empty:
                    y_vals = subset[dim].values
                    x_center = category_positions[cat]
                    jitter = rng.uniform(-0.06, 0.06, size=len(y_vals))

                    ax.scatter(
                        np.full_like(y_vals, x_center) + jitter,
                        y_vals,
                        color=model_color_dict[model],
                        alpha=0.35,
                        s=12
                    )

        ax.set_title(dim)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels([f"Category {c}" for c in categories])
        ax.set_ylim(0, 5)

    # labels on left column only
    axes[0].set_ylabel("MedGPT Score")
    axes[3].set_ylabel("MedGPT Score")

    # legend outside
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        bbox_to_anchor=(1.02, 0.5),
        loc="center left",
        title="Model",
        frameon=False
    )

    plt.tight_layout(rect=[0, 0, 0.85, 0.95])

    save_plot(fig, plot_dir, "medgpt_score_trends_0to5_perScore.png")



def figure_dimension_trends_v3(iter_scores: pd.DataFrame, summary: pd.DataFrame, n_iter: int, dims: list[str], categories: list[int], plot_dir: Path) -> None:
    rng = np.random.default_rng(45)
    use_95ci = False
    category_positions = {1: 0, 2: 1, 3: 2}
    colors = sns.color_palette("tab10", len(MODEL_ORDER_PLOT_MAIN))
    model_color_dict = dict(zip(MODEL_ORDER_PLOT_MAIN, colors))
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 6, figure=fig)
    axes = [
        fig.add_subplot(gs[0, 0:2]),
        fig.add_subplot(gs[0, 2:4]),
        fig.add_subplot(gs[0, 4:6]),
        fig.add_subplot(gs[1, 1:3]),
        fig.add_subplot(gs[1, 3:5]),
    ]
    y_ticks = [3, 3.5, 4, 4.5, 5]
    for ax, dim in zip(axes, dims):
        for model in MODEL_ORDER_PLOT_MAIN:
            d = summary[summary["model"] == model].sort_values("category_id")
            iter_d = iter_scores[iter_scores["model"] == model]
            if d.empty:
                continue
            x = np.array([category_positions[c] for c in d["category_id"]])
            mean = d[f"{dim}_mean"].to_numpy()
            std = d[f"{dim}_std"].to_numpy()
            lo, hi = lower_upper(mean, std, n=n_iter, use_95ci=use_95ci)
            ax.plot(x, mean, marker="o", linewidth=2, label=model, color=model_color_dict[model])
            ax.fill_between(x, lo, hi, alpha=0.15, color=model_color_dict[model])
            for cat in categories:
                subset = iter_d[iter_d["category_id"] == cat]
                if not subset.empty:
                    y_vals = subset[dim].values
                    x_center = category_positions[cat]
                    jitter = rng.uniform(-0.05, 0.05, size=len(y_vals))
                    ax.scatter(np.full_like(y_vals, x_center) + jitter, y_vals, color=model_color_dict[model], alpha=0.3, s=10)
        ax.set_title(dim)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels([f"Category {c}" for c in categories])
        ax.set_ylim(2.8, 5)
        ax.set_yticks(y_ticks)
        ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.7)
    axes[0].set_ylabel("MedGPT Score")
    axes[3].set_ylabel("MedGPT Score")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(1.02, 0.5), loc="center left", title="Model", frameon=False)
    plt.tight_layout(rect=[0, 0, 0.85, 0.95])
    save_plot(fig, plot_dir, "medgpt_score_trends_2.8to5_perScore.png")


def analysis_cat1_vs_cat3_drop(iter_scores: pd.DataFrame, csv_dir: Path) -> pd.DataFrame:
    dims = ["MedGPT1", "MedGPT2", "MedGPT3"]
    results = []
    for model in iter_scores["model"].unique():
        model_df = iter_scores[iter_scores["model"] == model]
        for dim in dims:
            pivot = model_df.pivot(index="iteration_number", columns="category_id", values=dim)
            if 1 not in pivot.columns or 3 not in pivot.columns:
                continue
            aligned = pivot.dropna(subset=[1, 3])
            cat1 = aligned[1].values
            cat3 = aligned[3].values
            diffs = cat1 - cat3
            if len(diffs) < 3:
                continue
            p_val = paired_permutation_test(diffs)
            ci_low, ci_high = bootstrap_ci(diffs)
            results.append({
                "model": model,
                "metric": dim,
                "mean_cat1": np.mean(cat1),
                "mean_cat3": np.mean(cat3),
                "mean_drop(cat1-cat3)": np.mean(diffs),
                "CI95_low": ci_low,
                "CI95_high": ci_high,
                "p_value": p_val,
                "significant_drop": (p_val < 0.05 and np.mean(diffs) > 0),
            })
    results_df = pd.DataFrame(results).sort_values(["metric", "p_value"])
    save_csv(results_df, csv_dir, "cat1_vs_cat3_drop_results")
    return results_df


def figure_gold_f1_modalities(answers: pd.DataFrame, csv_dir: Path, plot_dir: Path) -> pd.DataFrame:
    rng = np.random.default_rng(46)
    palette = sns.color_palette("Set3", n_colors=10)
    answers_gold = answers[answers["model"] == "Rephrased Gold Answers"].copy()
    f1_columns = {
        "f1_score": "Original F1",
        "synonyms_f1_dict": "GPT-Dict F1",
        "synonyms_f1_snomed": "SNOMED F1",
        "synonyms_f1_wn": "WordNet F1",
        "synonyms_lemmatized_f1_dict": "GPT-Dict+Lemma F1",
        "synonyms_lemmatized_f1_snomed": "SNOMED+Lemma F1",
        "synonyms_lemmatized_f1_wn": "WordNet+Lemma F1",
    }
    columns_needed = ["category_id", "iteration_number"] + list(f1_columns.keys())
    subset = answers_gold[columns_needed].copy()
    grouped = subset.groupby(["category_id", "iteration_number"]).mean().reset_index()
    summary = grouped.groupby("category_id")[list(f1_columns.keys())].agg(["mean", "std"]).reset_index()
    summary.columns = ["category_id"] + [f"{f1_columns[col]}_{stat}" for col in f1_columns for stat in ["mean", "std"]]
    save_csv(grouped, csv_dir, "gold_f1_modalities_grouped")
    save_csv(summary, csv_dir, "gold_f1_modalities_summary")

    metrics = list(f1_columns.values())
    categories = sorted(summary["category_id"].unique())
    bar_height = 0.12
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.set_context("talk")
    for i, (col, metric) in enumerate(zip(f1_columns.keys(), metrics)):
        means = summary[f"{metric}_mean"]
        stds = summary[f"{metric}_std"]
        y_positions = np.array(categories) + i * bar_height
        ax.barh(y_positions, means, xerr=stds, height=bar_height, label=metric, capsize=3, color=palette[i], alpha=0.7)
        for cat in categories:
            subset_iter = grouped[grouped["category_id"] == cat]
            if not subset_iter.empty:
                x_vals = subset_iter[col].values
                y_center = cat + i * bar_height
                jitter = rng.uniform(-bar_height / 3, bar_height / 3, size=len(x_vals))
                ax.scatter(x_vals, np.full_like(x_vals, y_center) + jitter, color=palette[i], alpha=0.4, s=15)
    ax.set_yticks(np.array(categories) + (len(metrics) / 2 - 0.5) * bar_height)
    ax.set_yticklabels([f"Category {cat}" for cat in categories])
    ax.set_xlabel("Average F1 Score")
    fig.legend(title="Metric", loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=3, frameon=False)
    plt.tight_layout()
    save_plot(fig, plot_dir, "gold_f1_modalities.png")

    summary_rounded = summary.copy()
    for col in summary.columns:
        if any(stat in col for stat in ["mean", "std"]):
            summary_rounded[col] = summary_rounded[col].round(3)
    save_csv(summary_rounded, csv_dir, "gold_f1_modalities_summary_rounded")
    column_means = summary_rounded.mean(numeric_only=True).rename_axis("metric").reset_index(name="value")
    save_csv(column_means, csv_dir, "gold_f1_modalities_column_means")
    return summary_rounded


def figures_radar_bias_harm(answers: pd.DataFrame, csv_dir: Path, plot_dir: Path) -> pd.DataFrame:
    model_order = [
        "Rephrased Gold Answers", "Gemini 2.5 Pro", "MedGemma 27B", "Claude 3.5 Sonnet", "Llama 3.3-70B-Instruct",
        "Med42-70B", "Meditron 3-70B", "Gemma 3 27B", "Llama 3.1-8B-Instruct", "NVLM-70B",
    ]
    colors = sns.color_palette("tab10", len(model_order))
    model_color_dict = dict(zip(model_order, colors))
    score_columns = ["MedGPT4", "MedGPT5"]
    categories = [f"Category {i}" for i in range(1, 5)]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    answers_plot = answers[answers["model"] != "Llama 3.2-1B-Instruct"].copy()
    grouped = answers_plot.groupby(["model", "category_id", "iteration_number"])[score_columns].mean().reset_index()
    summary = grouped.groupby(["model", "category_id"])[score_columns].mean().reset_index()
    summary = summary[summary["category_id"].isin([1, 2, 3, 4])].copy()
    save_csv(grouped, csv_dir, "radar_bias_harm_grouped")
    save_csv(summary, csv_dir, "radar_bias_harm_summary")
    pivoted_4 = summary.pivot(index="model", columns="category_id", values="MedGPT4")[[1, 2, 3, 4]]
    pivoted_5 = summary.pivot(index="model", columns="category_id", values="MedGPT5")[[1, 2, 3, 4]]
    save_csv(pivoted_4.reset_index(), csv_dir, "bias_radar_pivot")
    save_csv(pivoted_5.reset_index(), csv_dir, "harm_radar_pivot")
    titles = ["MedGPT4: Potential bias in responses", "MedGPT5: Potential harm in responses"]
    pivoted_all = [pivoted_4, pivoted_5]
    filenames = ["bias_radar.png", "harm_radar.png"]
    for pivoted, _title, filename in zip(pivoted_all, titles, filenames):
        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
        for model, values in pivoted.iterrows():
            data = values.tolist() + [values.tolist()[0]]
            ax.plot(angles, data, label=model, color=model_color_dict.get(model, "gray"), linewidth=1, linestyle="-")
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=13)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("center")
            label.set_y(label.get_position()[1] - 0.16)
        ax.set_yticks([4, 4.5, 5])
        ax.set_yticklabels(["4", "4.5", "5"], fontsize=10)
        ax.set_rlabel_position(0)
        ax.grid(True, linestyle="dotted", alpha=0.6)
        ax.set_ylim(3.75, 5)
        plt.tight_layout()
        save_plot(fig, plot_dir, filename)
    return summary.round(3).sort_values(by=["model", "category_id"])


def figure_cat3_cat4_mean_medgpt123(answers: pd.DataFrame, csv_dir: Path, plot_dir: Path) -> pd.DataFrame:
    rng = np.random.default_rng(47)
    model_order = [
        "Rephrased Gold Answers", "Gemini 2.5 Pro", "MedGemma 27B", "Claude 3.5 Sonnet", "Llama 3.3-70B-Instruct",
        "Med42-70B", "Meditron 3-70B", "Gemma 3 27B", "Llama 3.1-8B-Instruct", "NVLM-70B", "Llama 3.2-1B-Instruct",
    ]
    colors = sns.color_palette("tab10", len(model_order))
    model_color_dict = dict(zip(model_order, colors))
    medgpt_scores = ["MedGPT1", "MedGPT2", "MedGPT3"]
    answers_plot = answers.copy()
    answers_plot["MedGPT123_mean"] = answers_plot[medgpt_scores].mean(axis=1)
    subset = answers_plot[answers_plot["category_id"].isin([3, 4])].copy()
    grouped = subset.groupby(["model", "category_id", "iteration_number"])["MedGPT123_mean"].mean().reset_index()
    summary = grouped.groupby(["model", "category_id"])["MedGPT123_mean"].agg(["mean", "std"]).reset_index()
    pivoted = summary.pivot(index="model", columns="category_id")
    pivoted.columns = ["mean_cat3", "mean_cat4", "std_cat3", "std_cat4"]
    pivoted = pivoted.reset_index()
    save_csv(grouped, csv_dir, "cat3_cat4_mean_medgpt123_grouped")
    save_csv(summary, csv_dir, "cat3_cat4_mean_medgpt123_summary")
    save_csv(pivoted, csv_dir, "cat3_cat4_mean_medgpt123_pivot")

    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7, 6))
    x_positions = {3: 0, 4: 1}
    x = [0, 1]
    x_labels = ["Category 3", "Category 4"]
    for model in model_order:
        row = pivoted[pivoted["model"] == model]
        iter_model = grouped[grouped["model"] == model]
        if row.empty:
            continue
        row = row.iloc[0]
        y = [row["mean_cat3"], row["mean_cat4"]]
        yerr = [row["std_cat3"], row["std_cat4"]]
        ax.errorbar(x, y, yerr=yerr, label=model, marker="o", capsize=4, linewidth=2, color=model_color_dict[model])
        for cat in [3, 4]:
            subset_iter = iter_model[iter_model["category_id"] == cat]
            if not subset_iter.empty:
                y_vals = subset_iter["MedGPT123_mean"].values
                x_center = x_positions[cat]
                jitter = rng.uniform(-0.05, 0.05, size=len(y_vals))
                ax.scatter(np.full_like(y_vals, x_center) + jitter, y_vals, color=model_color_dict[model], alpha=0.25, s=10)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=15)
    ax.set_ylabel("Mean MedGPT1–3")
    ax.set_ylim(1.0, 4.8)
    ax.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False, prop={"size": 12}, title_fontsize="11")
    plt.tight_layout()
    save_plot(fig, plot_dir, "cat_5.png")

    pivoted_display = summary.pivot(index="model", columns="category_id")
    pivoted_display.columns = [f"Category {col[1]} {col[0]}" for col in pivoted_display.columns]
    pivoted_display = pivoted_display.reset_index().round(3)
    save_csv(pivoted_display, csv_dir, "cat3_cat4_mean_medgpt123_table")
    return pivoted_display


def figure_supervised_vs_unsupervised(answers: pd.DataFrame, unsupervised: pd.DataFrame, csv_dir: Path, plot_dir: Path) -> pd.DataFrame:
    rng = np.random.default_rng(48)
    data1 = answers.copy()
    data2 = unsupervised.copy()
    model_order = [
        "Gemini 2.5 Pro", "MedGemma 27B", "Claude 3.5 Sonnet", "Llama 3.3-70B-Instruct",
        "Med42-70B", "Meditron 3-70B", "Gemma 3 27B", "Llama 3.1-8B-Instruct", "NVLM-70B", "Llama 3.2-1B-Instruct",
    ]
    data2.columns = [f"unsupervised_{col}" if col.startswith("Med") else col for col in data2.columns]
    data = pd.merge(
        data1, data2,
        on=["model", "category_id", "iteration_number", "question_index", "model_answer", "question"],
    )
    save_csv(data, csv_dir, "supervised_vs_unsupervised_merged")

    all_results = []
    all_iter_points = []
    for i in range(1, 4):
        med_col = f"MedGPT{i}"
        unsup_col = f"unsupervised_MedGPT{i}"
        if med_col not in data.columns or unsup_col not in data.columns:
            continue
        avg_iter = (
            data.groupby(["model", "category_id", "iteration_number"])[[med_col, unsup_col]]
            .mean()
            .reset_index()
            .rename(columns={med_col: "Supervised", unsup_col: "Unsupervised"})
        )
        avg_by_iter = (
            avg_iter.groupby(["model", "iteration_number"])[["Supervised", "Unsupervised"]]
            .mean()
            .reset_index()
        )
        iter_long = avg_by_iter.melt(
            id_vars=["model", "iteration_number"], value_vars=["Supervised", "Unsupervised"], var_name="MedGPT", value_name="score"
        )
        iter_long["GPT"] = f"GPT{i}"
        all_iter_points.append(iter_long)
        summary_stats = avg_by_iter.groupby("model")[["Supervised", "Unsupervised"]].agg(["mean", "std"]).reset_index()
        summary_stats.columns = ["model", "avg_MedGPT", "sd_MedGPT", "avg_unsupGPT", "sd_unsupGPT"]
        summary_stats["GPT"] = f"GPT{i}"
        df_plot = pd.melt(
            summary_stats, id_vars=["model", "GPT"], value_vars=["avg_MedGPT", "avg_unsupGPT"], var_name="MedGPT", value_name="avg_score"
        )
        df_plot["MedGPT"] = df_plot["MedGPT"].map({"avg_MedGPT": "Supervised", "avg_unsupGPT": "Unsupervised"})
        df_plot["std"] = df_plot.apply(
            lambda row: summary_stats.loc[
                (summary_stats["model"] == row["model"]) & (summary_stats["GPT"] == row["GPT"]),
                "sd_MedGPT" if row["MedGPT"] == "Supervised" else "sd_unsupGPT",
            ].values[0],
            axis=1,
        )
        all_results.append(df_plot)
        save_csv(avg_iter, csv_dir, f"supervised_vs_unsupervised_gpt{i}_avg_iter")
        save_csv(avg_by_iter, csv_dir, f"supervised_vs_unsupervised_gpt{i}_avg_by_iter")
        save_csv(summary_stats, csv_dir, f"supervised_vs_unsupervised_gpt{i}_summary_stats")
        save_csv(df_plot, csv_dir, f"supervised_vs_unsupervised_gpt{i}_plot_data")

    plot_data = pd.concat(all_results, ignore_index=True)
    iter_points = pd.concat(all_iter_points, ignore_index=True)
    plot_data["model"] = pd.Categorical(plot_data["model"], categories=model_order, ordered=True)
    iter_points["model"] = pd.Categorical(iter_points["model"], categories=model_order, ordered=True)
    save_csv(plot_data, csv_dir, "supervised_vs_unsupervised_plot_data_all")
    save_csv(iter_points, csv_dir, "supervised_vs_unsupervised_iter_points")

    palette = {"Supervised": "#1f77b4", "Unsupervised": "#ff7f0e"}
    sns.set(style="whitegrid")
    g = sns.catplot(
        data=plot_data, x="model", y="avg_score", hue="MedGPT", col="GPT", kind="bar",
        palette=palette, order=model_order, height=5, aspect=1.2, sharey=False, ci=None,
    )
    for ax, (_gpt_name, group_df) in zip(g.axes.flat, plot_data.groupby("GPT")):
        bars = ax.containers
        for bar_group, (_, group) in zip(bars, group_df.groupby("MedGPT")):
            for bar, (_, row) in zip(bar_group, group.iterrows()):
                x = bar.get_x() + bar.get_width() / 2
                ax.errorbar(x, bar.get_height(), yerr=row["std"], fmt="none", ecolor="black", capsize=4, linewidth=1)
    for ax, gpt_name in zip(g.axes.flat, sorted(iter_points["GPT"].unique())):
        subset = iter_points[iter_points["GPT"] == gpt_name]
        for model_i, model in enumerate(model_order):
            for med_type in ["Supervised", "Unsupervised"]:
                vals = subset[(subset["model"] == model) & (subset["MedGPT"] == med_type)]["score"].values
                if len(vals) == 0:
                    continue
                offset = -0.2 if med_type == "Supervised" else 0.2
                jitter = rng.uniform(-0.04, 0.04, size=len(vals))
                ax.scatter(np.full_like(vals, model_i + offset) + jitter, vals, color=palette[med_type], alpha=0.4, s=12, zorder=3)
    for ax in g.axes.flat:
        title = ax.get_title()
        if "GPT =" in title:
            ax.set_title(title.replace("GPT =", ""))
    g.set_xticklabels(rotation=45, horizontalalignment="right")
    g.set_axis_labels("Model", "Average Score")
    g.fig.subplots_adjust(top=0.85)
    g.fig.savefig(plot_dir / "medGPT_vs_unsupervised_GPT_with_CI.png", dpi=300, bbox_inches="tight")
    plt.close(g.fig)

    table_data = []
    for i in range(1, 4):
        med_col = f"MedGPT{i}"
        unsup_col = f"unsupervised_MedGPT{i}"
        if med_col not in data.columns or unsup_col not in data.columns:
            continue
        avg_iter = (
            data.groupby(["model", "category_id", "iteration_number"])[[med_col, unsup_col]]
            .mean()
            .reset_index()
            .rename(columns={med_col: "mean_MedGPT", unsup_col: "mean_unsupGPT"})
        )
        avg_by_iter = (
            avg_iter.groupby(["model", "iteration_number"])[["mean_MedGPT", "mean_unsupGPT"]]
            .mean()
            .reset_index()
        )
        summary_stats = avg_by_iter.groupby("model")[["mean_MedGPT", "mean_unsupGPT"]].agg(["mean", "std"]).reset_index()
        summary_stats.columns = ["model", f"avg_MedGPT{i}", f"std_MedGPT{i}", f"avg_unsupGPT{i}", f"std_unsupGPT{i}"]
        table_data.append(summary_stats)
    table_combined = table_data[0]
    for i in range(1, len(table_data)):
        table_combined = pd.merge(table_combined, table_data[i], on="model")
    table_combined["model"] = pd.Categorical(table_combined["model"], categories=model_order, ordered=True)
    table_combined = table_combined.sort_values("model").round(3)
    table_combined["meanGPT"] = table_combined[["avg_MedGPT1", "avg_MedGPT2", "avg_MedGPT3"]].mean(axis=1)
    table_combined["meanUnsupGPT"] = table_combined[["avg_unsupGPT1", "avg_unsupGPT2", "avg_unsupGPT3"]].mean(axis=1)
    table_combined["meanDiff"] = table_combined["meanGPT"] - table_combined["meanUnsupGPT"]
    table_combined = table_combined.round(3)
    summary_subset = table_combined[["model", "meanGPT", "meanUnsupGPT", "meanDiff"]].copy()
    save_csv(table_combined, csv_dir, "supervised_vs_unsupervised_summary_table_full")
    save_csv(summary_subset, csv_dir, "supervised_vs_unsupervised_summary_subset")
    mean_diff_df = pd.DataFrame({"meanDiff_overall": [summary_subset["meanDiff"].mean()]})
    save_csv(mean_diff_df, csv_dir, "supervised_vs_unsupervised_mean_diff_overall")
    return summary_subset


def analysis_sentence_length(answers: pd.DataFrame, csv_dir: Path, plot_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = answers.copy()
    data = data[data["category_id"] != 4].copy()
    data["n_sentences"] = data["model_answer"].apply(count_sentences)
    summary_data_cat_model = data.groupby(["category_id", "model"])["n_sentences"].agg(mean_sentences="mean", sd_sentences="std").reset_index()
    summary_data_model = data.groupby("model")["n_sentences"].agg(mean_sentences="mean", sd_sentences="std").reset_index()
    model_order = [
        "Med42-70B", "Meditron 3-70B", "NVLM-70B", "Claude 3.5 Sonnet",
        "Llama 3.1-8B-Instruct", "Llama 3.3-70B-Instruct", "Llama 3.2-1B-Instruct", "Gemini 2.5 Pro",
        "Gemma 3 27B", "MedGemma 27B", "Rephrased Gold Answers",
    ]
    tab10 = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8",
    ]
    model_color_dict = dict(zip(model_order, tab10))
    data["model"] = pd.Categorical(data["model"], categories=model_order, ordered=True)
    save_csv(data, csv_dir, "sentence_length_input")
    save_csv(summary_data_cat_model, csv_dir, "summarySentenceLength_by_category_model")
    save_csv(summary_data_model, csv_dir, "summarySentenceLength_by_model")

    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=data, x="category_id", y="n_sentences", hue="model", palette=model_color_dict)
    plt.xlabel("Category")
    plt.ylabel("Number of sentences")
    plt.legend(title=None, fontsize=8, loc="upper right", bbox_to_anchor=(1, 1), frameon=False)
    plt.tight_layout()
    fig = plt.gcf()
    save_plot(fig, plot_dir, "sentences_by_category_model.pdf")

    model = ols("n_sentences ~ C(category_id)", data=data).fit()
    anova_table = sm.stats.anova_lm(model, typ=2).reset_index().rename(columns={"index": "term"})
    save_csv(anova_table, csv_dir, "sentence_length_anova")
    return summary_data_cat_model, summary_data_model, anova_table


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    args = parse_args()
    csv_dir, plot_dir = ensure_dirs(args.output_dir)
    answers_raw, unsupervised = load_inputs(args.input_dir)
    rag_path = args.input_dir / "all_questions_answers_scores_withRAG_withInfoInGuidelines.csv"
    answers_rag_raw = pd.read_csv(rag_path) if rag_path.exists() else None
    save_csv(answers_raw, csv_dir, "all_questions_answers_scores")
    save_csv(unsupervised, csv_dir, "all_questions_answers_scores_unsupervised")

    _answers_numeric, answer_filtered, answer_removed = numeric_cleanup(answers_raw)
    save_csv(answer_filtered, csv_dir, "answers_filtered_nonzero_scores")
    save_csv(answer_removed, csv_dir, "answers_removed_all_zero_scores")

    per_iteration_avg, summary_model = analysis_model_summary(answer_filtered, csv_dir)
    analysis_significance_vs_reference(per_iteration_avg, "Gemini 2.5 Pro", csv_dir, "gemini25pro_vs_others_significance")
    analysis_significance_vs_reference(per_iteration_avg, "Claude 3.5 Sonnet", csv_dir, "claude35sonnet_vs_others_significance")
    analysis_column_means(summary_model, csv_dir)

    iter_scores_bars, summary_bars, _summary_table_export = figure_gpt_and_f1_bars(answer_filtered, csv_dir, plot_dir)
    if answers_rag_raw is not None:
        _answers_rag_numeric, answer_rag_filtered, _answer_rag_removed = numeric_cleanup(answers_rag_raw)
        save_csv(answer_rag_filtered, csv_dir, "rag_answers_filtered_nonzero_scores")
        figure_rag_gpt_scores(answer_rag_filtered, csv_dir, plot_dir)
    else:
        print(f"Skipping RAG GPT scores plot; missing input file: {rag_path}")
    figure_average_medgpt_line(summary_bars, iter_scores_bars, csv_dir, plot_dir)

    iter_scores_dims, summary_dims, n_iter, dims, categories = compute_dimension_trend_inputs(answer_filtered, csv_dir)
    figure_dimension_trends_v1(iter_scores_dims, summary_dims, n_iter, dims, categories, plot_dir)
    figure_dimension_trends_v3(iter_scores_dims, summary_dims, n_iter, dims, categories, plot_dir)
    analysis_cat1_vs_cat3_drop(iter_scores_dims, csv_dir)

    figure_gold_f1_modalities(answer_filtered, csv_dir, plot_dir)
    radar_summary = figures_radar_bias_harm(answer_filtered, csv_dir, plot_dir)
    save_csv(radar_summary, csv_dir, "radar_bias_harm_summary_display")
    figure_cat3_cat4_mean_medgpt123(answer_filtered, csv_dir, plot_dir)
    figure_supervised_vs_unsupervised(answer_filtered, unsupervised, csv_dir, plot_dir)
    analysis_sentence_length(answer_filtered, csv_dir, plot_dir)

    print("Done.")
    print(f"CSV outputs:   {csv_dir.resolve()}")
    print(f"Plot outputs:  {plot_dir.resolve()}")


if __name__ == "__main__":
    main()
