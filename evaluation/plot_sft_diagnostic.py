"""Plot training memorization separately from novel-combination generalization."""

import argparse
import json
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
    if run["config"].get("diagnostic", {}).get("memorization_success_exact_matches") != 32:
        parser.error("expected the 32/64 memorization diagnostic, not the Dolly experiment")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
    rows = run["evaluation_history"]
    for key, label, color in (("train_response", "32 exemples appris", "#2166ac"),
                              ("dev_response", "64 combinaisons nouvelles", "#d6604d")):
        axes[0].plot([row["step"] for row in rows], [row[key]["loss"] for row in rows],
                     marker="o", label=label, color=color)
    axes[0].set(title="Loss avec les tokens précédents fournis",
                xlabel="Mises à jour", ylabel="Loss réponse et EOS (nats)")
    axes[0].set_ylim(bottom=0)
    axes[0].legend()
    axes[0].grid(alpha=0.2)
    labels, values, counts = [], [], []
    for key, split in (("probes", "Train"), ("dev_generation", "Dev")):
        for phase, title in (("initial", "avant"), ("final", "après")):
            metric = run[phase][key]
            labels.append(f"{split}\n{title}")
            values.append(100 * metric["exact_match_rate"])
            counts.append(f"{metric['exact_matches']}/{metric['examples']}")
    bars = axes[1].bar(labels, values, color=["#a6bddb", "#2166ac", "#f4b5a7", "#d6604d"])
    axes[1].bar_label(bars, labels=counts, padding=5)
    axes[1].set(title="Génération libre : prompt seul en entrée", ylabel="Réponses exactes (%)", ylim=(0, 112))
    axes[1].set_yticks(range(0, 101, 20))
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].set_axisbelow(True)
    fig.suptitle("Diagnostic SFT · 32 exemples train · 64 combinaisons nouvelles\n"
                 "Même vocabulaire et mêmes règles · seed 0 · budget fixé à 400 mises à jour", fontsize=12)
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
