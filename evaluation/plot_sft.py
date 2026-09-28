"""Plot supervised response learning alongside general-domain retention."""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    run = json.loads(args.metrics.read_text(encoding="utf-8"))
    rows = run["evaluation_history"]
    per_epoch = math.ceil(run["config"]["corpus"]["train_examples"] / run["config"]["batch_size"])
    epochs = [row["step"] / per_epoch for row in rows]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
    for ax, (key, full, label, color) in zip(axes, (
        ("dev_response", "dev_response", "Dev SFT : réponses et EOS", "#2166ac"),
        ("general_validation_sample", "general_validation_full", "Dev général", "#d6604d"),
    )):
        counts = {row[key]["scored_tokens"] for row in rows}
        if len(counts) != 1:
            parser.error("evaluation target count changed during the run")
        before = run["initial"][full]["perplexity"]
        after = run["final"][full]["perplexity"]
        ax.plot(epochs, [row[key]["loss"] for row in rows], color=color, marker="o")
        ax.set_title(f"{label}\nPerplexité complète : {before:.2f} → {after:.2f}".replace(".", ","))
        ax.set_xlabel("Passes sur les 3 000 exemples d’entraînement")
        ax.set_ylabel(f"Loss sur {next(iter(counts)):,} cibles fixes (nats)".replace(",", " "))
        ax.set_xticks(range(run["config"]["epochs"] + 1))
        ax.set_xlim(left=0)
        ax.grid(alpha=0.2)
    fig.suptitle("SFT · modèle 94 M · loss limitée aux réponses · seed 0\n"
                 "Échelles verticales propres à chaque dev · qualité générative évaluée séparément", fontsize=12)
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
