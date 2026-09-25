# Jalon 5 : comparer 59 M et 94 M de paramètres

Question : à budget de données fixé, quel compromis qualité/temps/mémoire offre
une réduction de largeur du modèle ? Le run
[94 M sur 50 M de tokens](../results/pretraining-50m.md) est déjà terminé ; un seul
nouveau modèle, de 59 M de paramètres, est entraîné depuis zéro.
La [comparaison est terminée le 25 septembre 2026](../results/model-scaling-59m-94m.md) :
perplexité dev complète **211,08 pour 59 M** contre **184,21 pour 94 M** ;
le modèle réduit économise **25,40 % de temps de boucle** et **13,56 % de mémoire
allouée**, avec des générations encore répétitives.

## Configuration

| Paramètre | Modèle 59 M | Référence 94 M |
| --- | ---: | ---: |
| Paramètres totaux | 58 948 864 | 94 124 928 |
| Largeur `d_model` | 256 | 384 |
| Largeur du MLP | 1 024 | 1 536 |
| Blocs | 8 | 8 |
| Têtes Q / groupes K/V | 8 / 2 | 8 / 2 |
| Dimension d’une tête | 32 | 48 |
| Cibles d’entraînement | 49 999 999 | 49 999 999 |
| Mises à jour | 24 415 | 24 415 |

La [configuration 59 M](pretraining_59m_params_50m_tokens_config.json) ne change
que le nom, `d_model` et `hidden_size` par rapport à
[celle de référence](pretraining_50m_config.json). Le ratio MLP/largeur reste 4.
RoPE, GQA, BF16/SDPA, contexte 256, batch 8, tokenizer et seed 0 sont conservés.
AdamW garde le pic `3e-4`, le plancher `3e-5`, 250 mises à jour de warmup,
la décroissance cosinus sur 24 415 mises à jour et le clipping à 1.

Les deux modèles partent de poids aléatoires propres à leur taille. La seed
identique n’implique pas des tenseurs initiaux identiques ni la même loss initiale
entre architectures de dimensions différentes.

## Données et exécution

Le run réutilise directement `data/pretraining-50m-v1/`. Les fichiers, le
manifeste, les fenêtres train et leur permutation, les 128 fenêtres fixes
d’évaluation et le dev complet sont les mêmes que dans la référence 94 M.
Les hashes et offsets ont été vérifiés avant lancement. Le holdout reste réservé
et le runner ne le lit pas.

```sh
python -m reimplementation.train_baseline --device cuda \
  --config experiments/pretraining_59m_params_50m_tokens_config.json \
  --data-dir data/pretraining-50m-v1
```

Le runner crée un nouveau dossier `runs/simple-baseline-*/`. Il sauvegarde la
reprise toutes les 1 000 mises à jour et à la fin, puis le modèle final, les
mesures et les trois générations greedy habituelles. Le checkpoint final est
rechargé et ses logits comparés à ceux du modèle entraîné.

Pour reprendre après interruption, utiliser le même corpus :

```sh
python -m reimplementation.train_baseline --device cuda \
  --resume runs/simple-baseline-REPLACE/recovery.pt \
  --data-dir data/pretraining-50m-v1
```

## Mesures et portée

Comparer la loss et la perplexité sur les mêmes **999 999 cibles dev**, les
courbes d’évaluation, le débit, la durée et la mémoire. Les trois prompts, le
décodage greedy et la limite de 64 nouveaux tokens restent identiques.
Le [benchmark court](../results/model-size-benchmark.md) mesure environ
19 327 cibles/s pour 59 M contre 15 918 pour 94 M. L’estimation avant lancement
était de **45 à 55 minutes**, évaluations et sauvegardes comprises. Le run a
pris **49,82 minutes entre les horodatages du lanceur** ; les durées internes
et les courbes sont détaillées dans le rapport.

La quantité de données et le nombre de mises à jour sont fixés ; le calcul et
la durée varient avec la largeur. Il ne s’agit pas d’une comparaison à compute
égal. La recette est commune et n’est pas optimisée séparément pour chaque
taille. Une seule seed et deux tailles ne suffisent pas à ajuster une loi de
scaling ni à conclure à une taille optimale générale.
