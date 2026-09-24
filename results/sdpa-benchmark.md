# Benchmark court : attention manuelle et SDPA

Mesuré le 24 septembre 2026 sur **NVIDIA GeForce RTX 4060 Laptop GPU**, avec
PyTorch 2.10.0+cu128. [Mesures brutes](sdpa-benchmark.json).

Le modèle reste celui de GQA + RoPE : **94 124 928 paramètres**, 8 blocs de
dimension 384, 8 têtes Q, 2 groupes K/V, contexte 256, batch 8. Les deux variantes
s’entraînent en BF16 avec poids/AdamW FP32 et évaluation FP32.

| Mesure | Manuel | SDPA |
| --- | ---: | ---: |
| Débit médian, tokens cibles/s | 8 866 | 15 972 |
| Pic alloué maximal, octets | 4 900 682 752 | 4 564 052 480 |

Le débit médian augmente de **80,14 %** ; le pic alloué baisse de **6,87 %**.
Le profilage confirme les opérations `aten::_scaled_dot_product_flash_attention`
et `aten::_scaled_dot_product_flash_attention_backward` sur le vrai batch.
Le gain inclut le regroupement des projections et le noyau SDPA : il ne mesure
pas isolément l’effet de FlashAttention.

Quatre essais dans l’ordre manuel/SDPA/SDPA/manuel, chacun avec 3 mises à jour
de chauffe et 30 mesurées, soit 61 440 tokens cibles chronométrés. Les poids
initiaux ont le même SHA-256 ; les données et leur ordre sont identiques.
Le chronométrage inclut les transferts CPU→GPU, le forward, le backward et AdamW.
Il exclut l’initialisation, la préparation des batches CPU, la chauffe, les
évaluations et le profilage. Tous les poids et gradients vérifiés sont finis.

Les validations FP32 portent sur seulement 1 024 cibles : ce benchmark mesure
l’exécution, **pas la qualité après un entraînement complet**. Les tests séparés
comparent sorties et gradients entre les deux calculs. Deux essais par variante
et quelques secondes de mesure donnent un indicateur local, sensible notamment
à la température et à la charge du portable.

À ce débit, 50 M de tokens représenteraient environ **52 minutes** de boucle
d’entraînement, auxquelles s’ajoutent évaluations et sauvegardes. Le prochain
run mesurera le coût réel sur la durée.

Reproduction : `python -m experiments.sdpa_benchmark`.
Artefacts locaux : `runs/sdpa-benchmark-ufl68ho_/metrics.json`.
Voir le [fonctionnement](../experiments/sdpa.md).
