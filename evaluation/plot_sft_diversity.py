"""Compare association diversity with matched updates and identical dev examples."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("variant", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in (args.baseline, args.variant)]
    if runs[0]["sft_manifest"]["contents"]["files"]["val"] != runs[1]["sft_manifest"]["contents"]["files"]["val"]:
        parser.error("the two runs must use identical diagnostic dev files")
    if any(run["updates"] != 400 or run["response_tokens"] != 7000 for run in runs):
        parser.error("expected matching 400-update and 7000-response-target budgets")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    categories = ("copy", "color", "name", None)
    x = np.arange(len(categories))
    for index, (run, label, color) in enumerate(zip(runs, ("32 exemples × 100 passes", "64 exemples × 50 passes"),
                                                   ("#2166ac", "#1b9e77"))):
        counts = []
        for category in categories:
            rows = [row for row in run["final"]["dev_generation"]["records"] if category is None or row["category"] == category]
            counts.append((sum(row["exact_match"] for row in rows), len(rows)))
        bars = axes[0].bar(x + (index - 0.5) * 0.36, [100 * correct / n for correct, n in counts],
                           width=0.36, color=color, label=label)
        axes[0].bar_label(bars, labels=[f"{correct}/{n}" for correct, n in counts], padding=4, fontsize=9)
        history = run["evaluation_history"]
        axes[1].plot([row["step"] for row in history],
                     [row["general_validation_sample"]["loss"] for row in history],
                     marker="o", color=color, label=label)
    axes[0].set_xticks(x, ("Copie", "Couleurs", "Prénoms", "Total"))
    axes[0].set(title="Mêmes 64 cas dev · génération libre", ylabel="Réponses exactes (%)", ylim=(0, 118))
    axes[0].set_yticks(range(0, 101, 20))
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].set_axisbelow(True)
    axes[1].set(title="Dégradation sur le dev général", xlabel="Mises à jour",
                ylabel="Loss sur les mêmes 32 768 cibles (nats)")
    axes[1].grid(alpha=0.2)
    axes[1].legend(loc="lower right")
    fig.suptitle("Diversité des associations SFT · mêmes poids initiaux et mêmes réglages\n"
                 "400 mises à jour · 3 200 présentations · 7 000 cibles réponse/EOS · seed 0", fontsize=12)
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
