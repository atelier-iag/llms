"""Plot separate paired SFT transfer conditions without pooling their scores."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.sft_transfer_data import CONDITIONS, COUNTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.analysis.read_text(encoding="utf-8"))
    labels = ("Référence", "Question\nreformulée", "Ordre\ninversé", "Nouvelles\nvaleurs")
    for model in ("initial", "sft"):
        if set(data[model]) != set(CONDITIONS):
            parser.error("expected the four frozen conditions for both models")
        for condition in CONDITIONS:
            scores = data[model][condition]["scores"]
            if any(scores[task]["examples"] != n for task, n in COUNTS.items()) or scores["total"]["examples"] != 64:
                parser.error("unexpected diagnostic task counts")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    x = np.arange(4)
    for index, (model, name, color) in enumerate((
        ("initial", "Modèle initial", "#2166ac"), ("sft", "Après SFT à 1e-4", "#1b9e77")
    )):
        correct = [data[model][condition]["scores"]["total"]["exact_matches"] for condition in CONDITIONS]
        bars = axes[0].bar(x + (index - 0.5) * 0.36, [100 * n / 64 for n in correct],
                          width=0.36, label=name, color=color)
        axes[0].bar_label(bars, labels=[f"{n}/64" for n in correct], padding=4, fontsize=9)
    axes[0].set_xticks(x, labels)
    axes[0].set(title="Résultat global, par condition", ylabel="Réponses exactes (%)", ylim=(0, 115))
    axes[0].set_yticks(range(0, 101, 20))
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].set_axisbelow(True)
    axes[0].legend(loc="upper right", fontsize=9)
    matrix = np.array([[100 * data["sft"][condition]["scores"][task]["exact_matches"] / n
                        for task, n in COUNTS.items()] for condition in CONDITIONS])
    heatmap = axes[1].imshow(matrix, vmin=0, vmax=100, cmap="YlGnBu", aspect="auto")
    for row, condition in enumerate(CONDITIONS):
        for col, (task, n) in enumerate(COUNTS.items()):
            correct = data["sft"][condition]["scores"][task]["exact_matches"]
            axes[1].text(col, row, f"{correct}/{n}", ha="center", va="center",
                         color="white" if matrix[row, col] > 60 else "#17202a", fontsize=12)
    axes[1].set_xticks(range(3), ("Copie", "Couleurs", "Prénoms"))
    axes[1].set_yticks(range(4), labels)
    axes[1].set_title("Modèle SFT : détail par tâche")
    fig.colorbar(heatmap, ax=axes[1], label="Réponses exactes (%)", shrink=0.85)
    fig.suptitle("Transfert des consignes SFT · deux checkpoints figés, aucun entraînement\n"
                 "64 cas appariés par condition · un seul facteur modifié · génération greedy", fontsize=12)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        path = args.output_prefix.with_suffix("." + extension)
        fig.savefig(path, dpi=160)
        if extension == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
