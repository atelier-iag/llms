"""Plot the fixed-sample evaluations and training losses of a completed baseline."""

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
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    evaluations = metrics["evaluation_history"]
    training = metrics["training_history"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    x = [record["target_tokens"] / 1e6 for record in evaluations]
    for name, label, color in [
        ("train", "Train : échantillon fixe", "#2166ac"),
        ("validation", "Validation : échantillon fixe", "#d6604d"),
    ]:
        axes[0].plot(x, [record[name]["loss"] for record in evaluations],
                     marker="o", markersize=4, label=label, color=color)
    sample_size = evaluations[0]["validation"]["scored_tokens"]
    axes[0].set_title(f"Même modèle évalué sur {sample_size:,} cibles par split".replace(",", " "))
    axes[0].legend(frameon=False)
    axes[1].plot([record["target_tokens"] / 1e6 for record in training],
                 [record["mean_training_loss"] for record in training], color="#555555")
    axes[1].set_title("Loss pendant les mises à jour\nMoyennes pondérées par le nombre de cibles")
    for ax in axes:
        ax.set_xlabel("Cibles d’entraînement traitées (millions)")
        ax.set_ylabel("Cross-entropy moyenne (nats / cible)")
        ax.grid(alpha=0.2)
        ax.set_xlim(left=0, right=metrics["training_target_tokens"] / 1e6)
    config = metrics["config"]
    model = config["model"]
    fig.suptitle(
        f"Baseline simple · d={model['d_model']} · {model['num_layers']} blocs · "
        f"{metrics['parameter_count'] / 1e6:.1f} M paramètres · seed {config['seed']}\n"
        "Un passage sur le corpus · aucune position explicite · FP32",
        fontsize=13,
    )
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        path = args.output_prefix.with_suffix(f".{extension}")
        fig.savefig(path, dpi=160)
        if extension == "svg":
            # Matplotlib emits trailing spaces inside path attributes; newlines
            # already separate those coordinates, so keep the tracked SVG tidy.
            lines = path.read_text(encoding="utf-8").splitlines()
            path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
