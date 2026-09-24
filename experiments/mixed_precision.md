# Mixed precision BF16 — suite du jalon 3

Après RoPE et GQA, on modifie la précision des calculs d’entraînement. Le modèle
et le budget de données restent les mêmes. Le but est de mesurer le compromis
entre temps, mémoire et stabilité numérique ; la précision mixte ne fournit
pas à elle seule davantage de connaissances au modèle.

## Ce qui change

La configuration [bf16_config.json](bf16_config.json) ajoute `precision: "bf16"`
à la [référence GQA](gqa_config.json), en changeant uniquement le nom de l’expérience.
Sans cette option, l’entraînement conserve son comportement FP32.

Dans [train.py](../reimplementation/train.py), le calcul avant et la loss sont
encadrés par `torch.autocast` :

```python
with training_autocast(tokens.device, precision):
    logits = model(inputs)
    loss = cross_entropy_loss(logits, targets)
loss.backward()
optimizer.step()
```

Les projections linéaires et produits matriciels éligibles utilisent BF16.
Les poids, leurs gradients et les moments AdamW restent FP32. Le softmax de
l’attention, les réductions RMSNorm et le log-softmax de la loss utilisent
explicitement FP32 lorsque leurs entrées sont en basse précision. RoPE conserve
ses angles en FP32, puis restitue les vecteurs dans leur type d’entrée.
La rétropropagation se fait hors du contexte autocast. Cette variante BF16
n’utilise pas de GradScaler et ne propose pas de variante FP16.

Voir la [documentation AMP de PyTorch 2.10](https://docs.pytorch.org/docs/2.10/amp.html)
pour la sélection des opérations et le périmètre du contexte autocast.

L’évaluation et la génération restent **FP32** afin de comparer les poids
entraînés avec le même protocole que la référence. Dans `metrics.json`, `dtype`
décrit les poids FP32, `training_precision` décrit l’entraînement et
`evaluation_precision` explicite l’évaluation. Les checkpoints d’inférence
gardent le même format. Les checkpoints de reprise conservent la configuration
de précision et refusent sa modification lors d’une reprise.

## Première mesure courte

```sh
python -m experiments.mixed_precision_benchmark
```

Le script utilise le vrai modèle GQA de **94 124 928 paramètres**, les vrais tokens,
le contexte 256 et le batch 8. Chaque essai repart des mêmes poids (empreinte
vérifiée), traite les mêmes 3 batches de chauffe puis les mêmes 20 batches
chronométrés, avec le calendrier d’apprentissage de l’expérience complète.
Deux répétitions inversent l’ordre : FP32/BF16, puis BF16/FP32.

Les métriques restent dans un nouveau dossier local `runs/precision-benchmark-*/`.
Le chronomètre inclut le transfert des batches CPU vers le GPU, le calcul avant,
la rétropropagation et AdamW ; les batches CPU sont préparés au préalable.
L’initialisation, la chauffe et les évaluations FP32 sont hors du chronomètre.
Le pic mémoire est le pic alloué par PyTorch pendant les pas mesurés.

La vérification de stabilité inclut des pertes finies, des gradients et poids
finis, ainsi qu’une évaluation FP32 sur 1 024 cibles de validation avant et après.
Ce petit échantillon n’est ni la validation complète ni un holdout : cette mesure
ne permet pas de conclure sur la qualité finale du modèle.

**Mesure du 24 septembre 2026 :** sur deux essais par précision, le débit médian
passe de 8 070 à 8 929 cibles/s (+10,64 %), et le pic alloué maximal baisse de
6,14 %. Les pertes et gradients restent finis. Voir les
[résultats détaillés et leurs limites](../results/mixed-precision-benchmark.md).

## Comparaison sur le corpus complet

```sh
python -m reimplementation.train_baseline --device cuda --config experiments/bf16_config.json
```

Le protocole conserve **18 999 999 cibles / 9 279 mises à jour**, la seed 0 et
les **999 999 cibles** de validation complète. Il repart de poids aléatoires,
comme l’expérience GQA FP32. Comparer sa loss et ses générations aux
[mesures GQA](../results/gqa-baseline.md). Les coûts globaux du run GQA interrompu
restent indisponibles ; utiliser le benchmark court pour la comparaison directe
de débit et de mémoire.

Pour reprendre un run BF16 interrompu :

```sh
python -m reimplementation.train_baseline --device cuda --resume runs/REPLACE/recovery.pt
```

Le runner vérifie la prise en charge native BF16 sur CUDA avant de charger les
données. Les tests CPU utilisent aussi BF16 pour vérifier le pipeline et la
reprise ; ils ne servent pas à prédire les performances GPU.
