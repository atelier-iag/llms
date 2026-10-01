"""Compare historical dev and final holdout perplexities on general text."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LABELS = {
    "general_94m_50m": "94 M · 50 M tokens (référence)",
    "general_94m_25m": "94 M · 25 M tokens",
    "general_59m_50m": "59 M · 50 M tokens",
    "python_94m": "94 M · adaptation Python",
    "dolly_94m": "94 M · SFT Dolly",
    "toy_sft_3e4": "94 M · SFT diagnostic 3e-4",
    "toy_sft_1e4": "94 M · SFT diagnostic 1e-4",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--config", type=Path, default=Path("experiments/final_holdout_config.json"))
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if set(metrics["models"]) != set(LABELS):
        parser.error("expected all seven frozen checkpoints")
    dev, holdout = [], []
    for key in LABELS:
        old = json.loads(Path(config["models"][key]["historical_metrics"]).read_text(encoding="utf-8"))
        if "final" in old:
            score = old["final"]["general_validation_full"]
        else:
            score = old.get("final_general_validation", old["final_full_validation"])
        final = metrics["models"][key]["general"]
        if score["scored_tokens"] != 999999 or final["scored_tokens"] != 999999:
            parser.error("expected full dev/holdout coverage")
        dev.append(score["perplexity"])
        holdout.append(final["perplexity"])
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(10, 5.6), layout="constrained")
    y = np.arange(len(LABELS))
    for i, (d, h) in enumerate(zip(dev, holdout)):
        ax.plot([h, d], [i, i], color="#c3c9ce", linewidth=2, zorder=1)
    ax.scatter(dev, y, label="Dev historique", color="#2166ac", s=45, zorder=2)
    ax.scatter(holdout, y, label="Holdout final", color="#1b9e77", marker="s", s=40, zorder=2)
    for i, (d, h) in enumerate(zip(dev, holdout)):
        ax.annotate(f"{d:,.2f}".replace(",", " ").replace(".", ","), (d, i),
                    xytext=(5, 6), textcoords="offset points", color="#2166ac", fontsize=9)
        ax.annotate(f"{h:,.2f}".replace(",", " ").replace(".", ","), (h, i),
                    xytext=(5, -12), textcoords="offset points", color="#13745a", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlim(min(holdout + dev) * 0.85, max(holdout + dev) * 1.5)
    ax.set_yticks(y, tuple(LABELS.values()))
    ax.invert_yaxis()
    ax.set_xlabel("Perplexité générale · échelle logarithmique · plus bas = meilleure prédiction")
    ax.grid(axis="x", which="both", alpha=0.15)
    ax.legend(loc="lower right")
    ax.set_title("Clôture du laboratoire LLM · sept checkpoints figés\n"
                 "999 999 cibles par split · aucun entraînement pendant l’évaluation", pad=16)
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
