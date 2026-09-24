# Jalon 5 : comparer 25 M et 50 M de tokens

Question : à modèle fixé, que gagne-t-on en qualité de validation et de génération
en doublant le budget de données et le calcul associé ? Le run
[50 M terminé](../results/pretraining-50m.md) sert de référence. Un nouveau run
25 M repart de poids aléatoires avec la même seed et le même modèle.

## Données imbriquées et dev fixe

```sh
python -m reimplementation.subset_corpus \
  --parent-dir data/pretraining-50m-v1 \
  --output-dir data/pretraining-25m-v1 --train-tokens 25000000
```

Le script extrait **6 250 000 tokens train par domaine** directement des fichiers
locaux de la référence. Il retient les premiers documents dans l’ordre du corpus
parent. Le dernier document de chaque domaine est tronqué au quota et conserve
son EOS : quatre positions de coupure reçoivent donc cet EOS. Les autres tokens
sont copiés tels quels. C’est un sous-ensemble de documents et de préfixes de
documents ; il n’y a ni nouveau téléchargement ni nouvelle tokenisation.

**Dev et holdout sont copiés à l’identique, octet pour octet**, avec les mêmes
documents, offsets et SHA-256. Ils ne sont pas reconstruits par une nouvelle
exécution de la préparation Dolma. Les empreintes du manifeste et de l’index
parents sont enregistrées, ainsi que l’offset parent de chaque document train
retenu. Le script exige un nouveau dossier et vérifie l’intégrité du parent,
la couverture de l’index, les quotas et les copies réservées.

Voir l’[audit du sous-ensemble](../results/pretraining-25m-data.md) et le
[manifeste exact](../results/pretraining-25m-data.json). Le holdout reste réservé :
le copier et vérifier son hash ne calcule aucun score de modèle.

## Recette et budget

```sh
python -m reimplementation.train_baseline --device cuda \
  --config experiments/pretraining_25m_config.json \
  --data-dir data/pretraining-25m-v1
```

| Paramètre | 25 M | 50 M |
| --- | ---: | ---: |
| Tokens train, EOS compris | 25 000 000 | 50 000 000 |
| Cibles d’entraînement, un passage | 24 999 999 | 49 999 999 |
| Mises à jour | 12 208 | 24 415 |
| Warmup, mises à jour | 125 | 250 |
| Cibles du dev complet | 999 999 | 999 999 |

Les deux configurations conservent l’architecture de 94 124 928 paramètres,
RoPE, GQA 8Q/2KV, BF16/SDPA, le tokenizer, la seed 0, le contexte 256, le batch 8,
AdamW et son pic de learning rate `3e-4`, le plancher `3e-5` et le clipping à 1.
Le warmup garde une proportion d’environ 1 % du parcours ; le cosinus est
recalculé sur la durée de chaque run et atteint son plancher au dernier pas.
Les checkpoints intermédiaires du run 50 M n’ont donc pas le même calendrier
qu’un entraînement complet de 25 M et ne le remplacent pas.

Le mélange des fenêtres train est recalculé sur le sous-ensemble avec la même
seed ; ce n’est pas le préfixe de la séquence des batches du run 50 M. Les
fenêtres peuvent traverser un EOS comme dans la référence. Les 128 fenêtres
dev et le dev complet sont identiques dans les deux runs ; les échantillons
train couvrent des fichiers différents et ne doivent pas être présentés comme
une évaluation sur les mêmes tokens.

## Mesures prévues et portée

Comparer les métriques finales sur le dev complet, les courbes dev en fonction
des tokens traités, la durée, le débit et le pic mémoire. Garder les trois mêmes
prompts, le décodage greedy et la limite de 64 nouveaux tokens pour examiner les
générations. Vérifier le checkpoint rechargé et les hashes des données.
Le run 25 M devrait prendre environ **30 à 35 minutes**, à confirmer par mesure.

Cette première comparaison utilise une seule seed et deux budgets. Elle mesure
le résultat de deux budgets d’entraînement à forme de calendrier comparable,
pas une loi de scaling ajustée ni l’effet isolé de la diversité des données
indépendamment du nombre de mises à jour. Le mélange de domaines reste celui du
corpus OCR initial, avec ses limites de représentativité et de quasi-doublons.
L’étude de la taille du modèle viendra ensuite ; aucune nouvelle architecture
n’est introduite dans cette comparaison.
