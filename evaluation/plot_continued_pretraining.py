"""Plot domain adaptation and general-domain retention on their own fixed devs."""

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
    if not run.get("initialization") or not run.get("final_general_validation"):
        parser.error("requires an initialized run with before/after general validation")
    rows = run["evaluation_history"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
    for ax, (title, split, full, color) in zip(axes, (
        ("Dev Python", "validation", "full_validation", "#2166ac"),
        ("Dev général", "general_validation", "general_validation", "#d6604d"),
    )):
        initial, final = run[f"initial_{full}"], run[f"final_{full}"]
        counts = {row[split]["scored_tokens"] for row in rows}
        if len(counts) != 1 or initial["scored_tokens"] != final["scored_tokens"]:
            parser.error("before/after dev evaluation sizes differ")
        ax.plot([row["target_tokens"] / 1e6 for row in rows],
                [row[split]["loss"] for row in rows], color=color, marker="o", markersize=4)
        before, after = initial["perplexity"], final["perplexity"]
        ax.set_title(f"{title}\nPerplexité complète : {before:.2f} → {after:.2f}".replace(".", ","))
        ax.set_xlabel("Cibles Python supplémentaires (millions)")
        ax.set_ylabel(f"Loss sur {next(iter(counts)):,} cibles fixes (nats)".replace(",", " "))
        ax.set_xlim(left=0)
        ax.grid(alpha=0.2)
    fig.suptitle("Continued pretraining · 94 M paramètres · 5 M tokens Python\n"
                 "Même checkpoint initial · seed 0 · échelles verticales propres à chaque dev", fontsize=12)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        path = args.output_prefix.with_suffix(f".{extension}")
        fig.savefig(path, dpi=160)
        if extension == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n",
                            encoding="utf-8")
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
