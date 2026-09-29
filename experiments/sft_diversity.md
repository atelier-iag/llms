# SFT : diversité des associations à budget égal

Le [diagnostic précédent](../results/sft-diagnostic-32-64.md) obtient 32/32
réponses sur train et 46/64 sur dev. Sur les questions demandant qui possède
la carte, le modèle réutilise un partenaire vu dans le train malgré un
contexte nouveau. L’hypothèse testée est qu’une **deuxième association par
valeur dans le train** réduit ce raccourci.

Ce protocole est fixé avant l’exécution de la variante. Le dev est déjà
connu et a servi à formuler l’hypothèse ; ce n’est pas un test aveugle ni un
holdout. Aucun holdout réservé du laboratoire n’est évalué.

## Comparaison

| Élément | Référence | Variante diversité |
| --- | ---: | ---: |
| Exemples train distincts | 32 | 64 |
| Associations par valeur | 1 | 2 |
| Passes | 100 | 50 |
| Batch | 8 | 8 |
| Mises à jour | 400 | 400 |
| Présentations d’exemples | 3 200 | 3 200 |
| Cibles réponse/EOS | 7 000 | 7 000 |
| Tokens non paddés | 101 600 | 101 600 |
| Exemples dev | Les mêmes 64 | Les mêmes 64 |

On conserve le checkpoint généraliste 94 M / 50 M tokens, la seed 0,
l’architecture, le tokenizer, le format, le contexte, les tâches et leurs
proportions, la distribution des réponses, la précision et tous les réglages
d’AdamW. Même calendrier par mise à jour : pic `3e-4`, warmup 20, cosinus
jusqu’à `3e-5`, weight decay nul, clipping à 1.

Seules les associations d’entraînement sont enrichies : le double
d’exemples uniques, chacun vu moitié moins souvent. Le nombre de passes
change pour garder le budget constant. Les permutations et la composition
des batches changent avec la taille du train ; une seule seed ne sépare pas
cet effet de celui de la diversité.

## Données

Le [générateur](sft_diagnostic_data.py) garde la recette originale par défaut.
Avec `--diverse`, il ajoute le décalage 4 au décalage 1 du train. Les
décalages 2 et 3 restent réservés au dev. Le décalage 4 est fixé avant le run,
sans recherche de la meilleure variante sur les scores.

Les [128 exemples complets](sft_diversity_examples.json) contiennent :

- 64 train : 16 copies de mots, 24 extractions de couleur, 24 extractions de prénom ;
- 64 dev : exactement le fichier utilisé dans le premier diagnostic.

Les 32 exemples train originaux sont conservés. Pour chaque valeur, une
association supplémentaire est ajoutée. Par exemple, le train contient
désormais Alice avec Bruno **et** Alice avec Emma. Les associations Alice
avec Clara et Alice avec David restent au dev. Les deux questions opposées
sur la clé et la carte sont présentes pour chaque contexte train.

Ni les gabarits, ni les mots disponibles, ni l’ordre des phrases ne changent.
Les prompts et contextes sont disjoints entre train et dev. Les réponses
sont vérifiées par l’oracle textuel du diagnostic. Les séquences font au
plus 33 tokens, et les réponses avec EOS au plus 3 tokens.

L’[audit](../results/sft-diversity-64-64-data-audit.json) vérifie les 128
exemples, le réencodage, le même fichier dev et les budgets. La préparation
de la recette originale reproduit son manifeste et tous ses fichiers
**octet pour octet**. Le [nouveau manifeste](../results/sft-diversity-64-64-manifest.json)
conserve les hashes.

## Mesures fixées avant l’essai

Le point final reste la mise à jour 400, sans arrêt anticipé ni sélection
de checkpoint selon le dev. Les métriques sont :

- génération greedy sur les **64 cas dev fixes**, scores globaux et par tâche ;
- réussite sur les **12 questions dev demandant qui possède la carte** ;
- paires de questions correctement résolues dans le même contexte ;
- génération sur les 64 exemples train, séparée du score de généralisation ;
- loss réponse/EOS sur train et dev, toutes les 100 mises à jour ;
- loss/perplexité générale, même échantillon fixe pendant le run et mêmes
  999 999 cibles avant/après.

Les générations ont le même budget de 8 tokens et la même normalisation
que la référence. Le score `probes` du runner désigne les **64 cas train**,
et `dev_generation` désigne les **64 cas dev**. L’augmentation de la taille
du train rend ses scores de mémorisation non directement comparables en
nombre absolu ; les scores dev portent, eux, sur les mêmes cas.

Une progression appuierait l’intérêt de diversifier ces associations dans
ce cadre. Elle ne démontrerait ni une capacité générale à suivre les
consignes, ni une résolution de l’échec Dolly. La dégradation générale
reste mesurée et le checkpoint source reste intact.

## Exécution

```sh
python -m experiments.sft_diagnostic_data --diverse \
  --output-dir data/sft-diversity-64-64-v1

python -m reimplementation.train_sft --device cuda \
  --config experiments/sft_diversity_config.json \
  --init-from runs/simple-baseline-2fsdgx0q/model.pt \
  --data-dir data/sft-diversity-64-64-v1 \
  --probes data/sft-diversity-64-64-v1/train-probes.json \
  --general-data-dir data/pretraining-50m-v1
```

La [configuration](sft_diversity_config.json) réutilise le runner SFT et ses
sauvegardes atomiques. Le modèle final est sauvegardé avant les évaluations
et la mémoire d’AdamW est libérée avant cette dernière phase.
