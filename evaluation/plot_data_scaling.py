"""Plot fixed-dev learning curves and final full-dev perplexity for two budgets."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path, nargs=2)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in args.metrics]
    first, second = runs
    if (first["data"]["validation"]["sha256"] != second["data"]["validation"]["sha256"]
            or first["data"]["validation"]["evaluation_offsets"] != second["data"]["validation"]["evaluation_offsets"]
            or first["config"]["model"] != second["config"]["model"]
            or first["tokenizer"] != second["tokenizer"]
            or first["config"]["seed"] != second["config"]["seed"]):
        parser.error("comparison requires identical model, tokenizer, seed and dev data/offsets")
    sizes = {r["final_full_validation"]["scored_tokens"] for r in runs}
    sample_sizes = {e["validation"]["scored_tokens"] for r in runs for e in r["evaluation_history"]}
    if len(sizes) != 1 or len(sample_sizes) != 1:
        parser.error("dev evaluation sizes differ")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
    colors = ("#2166ac", "#d6604d")
    for i, (run, color) in enumerate(zip(runs, colors)):
        label = f"{run['config']['corpus']['train_tokens'] / 1e6:g} M tokens"
        evaluations = run["evaluation_history"]
        axes[0].plot([e["target_tokens"] / 1e6 for e in evaluations],
                     [e["validation"]["loss"] for e in evaluations],
                     marker="o", markersize=3, color=color, label=label)
        perplexity = run["final_full_validation"]["perplexity"]
        bars = axes[1].bar(i, perplexity, width=0.55, color=color, label=label)
        axes[1].bar_label(bars, labels=[f"{perplexity:.2f}".replace(".", ",")], padding=5)
    axes[0].set_title(f"Dev fixe : {next(iter(sample_sizes)):,} cibles".replace(",", " "))
    axes[0].set_xlabel("Cibles d’entraînement traitées (millions)")
    axes[0].set_ylabel("Cross-entropy moyenne (nats / cible)")
    axes[0].set_xlim(left=0)
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.2)
    axes[1].set_title(f"Dev complet : {next(iter(sizes)):,} cibles".replace(",", " "))
    axes[1].set_xticks(range(2), [f"{r['config']['corpus']['train_tokens'] / 1e6:g} M tokens" for r in runs])
    axes[1].set_ylabel("Perplexité finale (plus bas = mieux)")
    axes[1].set_ylim(0, 1.16 * max(r["final_full_validation"]["perplexity"] for r in runs))
    axes[1].set_axisbelow(True)
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle(f"Budget de données · modèle fixé à {first['parameter_count'] / 1e6:.2f} M paramètres\n"
                 f"Même dev · seed {first['config']['seed']} · calendrier adapté à chaque budget", fontsize=12)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        path = args.output_prefix.with_suffix(f".{extension}")
        fig.savefig(path, dpi=160)
        if extension == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
