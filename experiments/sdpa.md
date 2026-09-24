# Attention optimisée : SDPA

L’option `attention_backend="sdpa"` conserve notre modèle et remplace le calcul
manuel de l’attention par `torch.nn.functional.scaled_dot_product_attention`.
Elle s’applique à MHA et GQA, avec ou sans RoPE. Le défaut reste `manual` pour
les exemples pédagogiques et les anciens checkpoints.

Les projections de toutes les têtes Q sont regroupées dans une multiplication,
de même pour K et V. Les matrices apprises et leurs noms restent identiques ;
la concaténation des poids est différentiable. RoPE tourne ensuite Q et K,
puis SDPA calcule l’attention causale avec `is_causal=True`, `dropout_p=0.0`
et `enable_gqa=True` lorsque le nombre de têtes K/V est inférieur au nombre de Q.
Les sorties sont concaténées puis passent dans notre projection de sortie.

SDPA choisit un noyau compatible avec le matériel, le type et les dimensions.
Sur notre RTX 4060 Laptop, le profilage du vrai modèle en BF16 confirme
**FlashAttention en forward et backward**. Il évite de matérialiser la matrice
complète des scores et probabilités d’attention. L’évaluation demeure FP32 et
peut utiliser un autre noyau. Les modules renvoient `(sortie, None)` en mode
SDPA ; utiliser le mode manuel pour inspecter les poids d’attention.

La formule reste la même, mais l’ordre des opérations et les arrondis changent.
Les tests vérifient les sorties et tous les gradients en float64, la causalité,
le cas d’un token, les checkpoints et le parcours complet d’entraînement.
Un test CUDA BF16 force FlashAttention et compare ses sorties et gradients
au calcul manuel avec une tolérance relative adaptée.

```sh
python -m pytest tests/test_sdpa.py tests/test_baseline_training.py -q
python -m experiments.sdpa_benchmark
```

Le benchmark alterne manuel/SDPA/SDPA/manuel, avec les mêmes poids initiaux,
batches et précision BF16. Voir les [mesures](../results/sdpa-benchmark.md).
[sdpa_config.json](sdpa_config.json) ne change que le backend et le nom de
l’expérience par rapport à BF16. Pour une comparaison longue à données fixes :

```sh
python -m reimplementation.train_baseline --device cuda --config experiments/sdpa_config.json
```

Cette comparaison longue n’a pas été exécutée. Le prochain entraînement utilise
le [nouveau corpus de 50 M de tokens](pretraining_50m.md) ; il ouvre le jalon 4
et ne constitue donc pas une ablation contrôlée de SDPA sur la qualité.

Référence : [API SDPA, PyTorch 2.10](https://docs.pytorch.org/docs/2.10/generated/torch.nn.functional.scaled_dot_product_attention.html).
