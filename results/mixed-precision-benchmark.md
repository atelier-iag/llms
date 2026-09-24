# Première mesure FP32 / BF16 sur notre modèle GQA

Mesure du **24 septembre 2026**, terminée sur la RTX 4060 Laptop GPU avec
PyTorch 2.10.0+cu128. Sur ce benchmark court, BF16 augmente le débit médian de
**10,64 %** et réduit le pic alloué maximal par PyTorch de **6,14 %**.
Ce résultat concerne le coût de quelques pas ; la comparaison de qualité après
un entraînement complet reste à réaliser.

[Mesures intégrales](mixed-precision-benchmark.json),
[script](../experiments/mixed_precision_benchmark.py),
[explications et protocole complet](../experiments/mixed_precision.md).

## Protocole

- Même modèle RoPE + GQA : **94 124 928 paramètres**, 8 têtes Q, 2 groupes K/V.
- Mêmes poids initiaux (SHA-256 identique pour les quatre essais), seed 0.
- Mêmes 23 premiers batches du corpus mélangé : contexte 256, batch 8.
- 3 pas de chauffe puis 20 pas mesurés, soit **40 960 cibles chronométrées** par essai.
- Même AdamW, clipping et calendrier d’apprentissage que le run GQA complet.
- Ordre des essais : FP32, BF16, BF16, FP32.
- Validation en FP32 sur les mêmes **1 024 cibles**, avant et après les 23 pas.

Les poids et les états AdamW restent en FP32. Les batches CPU sont préparés
avant le chronométrage ; leur transfert GPU est inclus. L’initialisation,
la chauffe, l’évaluation et les contrôles de gradients sont hors chronomètre.

## Résultats

| Mesure | FP32 | BF16 |
|---|---:|---:|
| Durées des 20 pas, deux essais | 4,932 s / 5,228 s | 4,625 s / 4,550 s |
| Débit médian | 8 070 cibles/s | 8 929 cibles/s |
| Maximum des pics alloués PyTorch | 5 220 002 816 octets | 4 899 520 512 octets |
| Loss de validation initiale, échantillon | 11,703400 | 11,703400 |
| Loss de validation après 23 pas, échantillon | 10,988699 | 10,988533 |

Les deux répétitions de chaque précision donnent ici les mêmes pertes.
Les pertes, poids et gradients sont finis ; les fichiers du corpus sont inchangés.
Les empreintes des données et des sources sont conservées dans le JSON.

Ces pertes presque identiques vérifient le début de l’apprentissage sur cet
échantillon ; elles ne démontrent pas une équivalence après le corpus entier.
Deux répétitions et vingt pas mesurés donnent une indication locale, sensible
à l’état du GPU. Aucun gain uniforme de temps, de mémoire ou de qualité n’est
revendiqué. L’évaluation finale restera en FP32 sur les 999 999 cibles habituelles.

Les **119 tests** du laboratoire passent, dont le calcul BF16 et les mises à jour
FP32 sur CPU/CUDA, les réductions en FP32, le rechargement des poids et l’égalité
exacte sur CPU entre entraînement BF16 continu et interrompu puis repris.

Run local de ce benchmark : `runs/precision-benchmark-88g4_kc1/`.
