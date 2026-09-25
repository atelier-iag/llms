# Benchmark court : modèles de 59 M et 94 M de paramètres

Mesuré le **24 septembre 2026**, avant le lancement du modèle 59 M, sur la
RTX 4060 Laptop GPU. [Métriques brutes](model-size-benchmark.json).

| Mesure | Largeur 256, 59 M paramètres | Largeur 384, 94 M paramètres |
| --- | ---: | ---: |
| Débit médian, cibles/s | 19 326,72 | 15 918,40 |
| Pic alloué maximal, octets | 3 947 675 136 | 4 564 052 480 |

Quatre essais dans l’ordre 384/256/256/384, chacun avec **5 mises à jour de
chauffe et 50 mesurées**. Les deux modèles traitent les mêmes batches réels
du corpus 50 M, avec BF16/SDPA, batch 8 et contexte 256. Le MLP garde une
largeur égale à quatre fois celle du modèle. Les poids et gradients vérifiés
sont finis dans tous les essais.

Le chronométrage utilise `trial` dans
[mixed_precision_benchmark.py](../experiments/mixed_precision_benchmark.py).
Il inclut les transferts CPU→GPU, le forward, le backward et AdamW ; il exclut
la préparation des batches CPU, l’initialisation, la chauffe et la validation
FP32. Les évaluations courtes portent sur 1 024 cibles et ne mesurent pas la
qualité après entraînement complet.

Le débit médian du modèle 59 M est supérieur d’environ **21,4 %**. L’extrapolation
directe donne **43,1 minutes** pour la boucle sur 50 M de tokens. En appliquant
le ratio de débits à la durée réelle du run 94 M, on obtient **47,4 minutes**.
La référence a passé environ 3,1 minutes supplémentaires dans les évaluations,
sauvegardes et générations selon le chronomètre interne.

L’estimation retenue est donc **45 à 55 minutes au total**. Il s’agit d’une
extrapolation de courts essais, sensible à la charge et à la température du
portable ; le prochain run mesurera le coût soutenu réel.

Artefacts locaux : `runs/model-size-benchmark-5wjjgkw0/`, contenant
`metrics.json` et le script autonome `benchmark.py`. Le JSON publié est une
copie exacte des mesures originales. Voir le [protocole long](../experiments/model_scaling.md).
