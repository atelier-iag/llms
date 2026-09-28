# Diagnostic SFT : mémoriser 32 exemples, tester 64 combinaisons nouvelles

Le [premier SFT Dolly](../results/sft-dolly-3k.md) échoue aux 12 consignes
diagnostiques. Cela ne suffit pas à déterminer si le problème vient du
préentraînement, des données, des réglages ou de l’implémentation. Ce petit
essai contrôle d’abord la capacité à **mémoriser des réponses connues**, puis
mesure la généralisation à de nouvelles combinaisons.

**Essai terminé** : [32/32 exemples appris et 46/64 combinaisons nouvelles](../results/sft-diagnostic-32-64.md),
avec erreurs d’association sur les prénoms et forte dégradation générale.

L’[entraînement sur un minuscule sous-ensemble](https://cs231n.github.io/neural-networks-3/)
est un contrôle classique du pipeline. Une réussite ne certifie pas à elle
seule toute l’implémentation, ni une capacité générale à suivre des consignes.

## Données fixées avant l’expérience

Les [96 exemples complets](sft_diagnostic_examples.json) sont produits par
un [générateur déterministe](sft_diagnostic_data.py), sans source externe ni
réponse inventée par un autre modèle. Un oracle séparé relit les consignes
et contextes textuels et recalcule chaque réponse.

| Tâche | Train | Nouveaux exemples dev | Réponses possibles |
| --- | ---: | ---: | --- |
| Copier le mot indiqué par Target word | 8 | 16 | 8 mots |
| Extraire la couleur de la lanterne ou du panier | 12 | 24 | 6 couleurs |
| Extraire le prénom de la personne possédant la clé ou la carte | 12 | 24 | 6 prénoms |
| Total | **32** | **64** | |

Exemple train :

- consigne : `Which color is the lantern? Reply with one color.`
- contexte : `The lantern is red. The basket is blue.`
- réponse : `red`

Le même contexte est également associé à une question sur le panier, dont
la réponse est `blue`. Les questions opposées empêchent qu’une simple
lecture du premier objet suffise à tous les exemples d’extraction.
Le dev recombine les mêmes valeurs : par exemple, lanterne rouge avec
panier vert. Les réponses ont exactement les mêmes proportions par tâche
dans les deux splits. Répéter la réponse la plus fréquente de chaque tâche
donnerait **5/32 sur train et 10/64 sur dev, soit 15,625 %**.

Les paires utilisent un décalage de 1 dans les listes de valeurs pour train,
puis de 2 et 3 pour dev. Les prompts complets et les contextes sont
disjoints entre splits ; les gabarits, questions et mots possibles sont
**volontairement partagés**. On teste donc des combinaisons nouvelles de
valeurs connues, pas des consignes, formulations ou mots entièrement nouveaux.
Ce découpage contrôlé est distinct du regroupement anti-doublons de Dolly.

Le format instruction/contexte/réponse, le tokenizer et le masque de loss
restent ceux du premier SFT. Les séquences font au plus **33 tokens**,
réponse et EOS compris, dans le contexte maximal de 256. Aucune troncature.
Les 32 exemples train contiennent **1 016 tokens**, dont **70 cibles
réponse/EOS** ; le dev contient 2 032 tokens, dont 140 cibles.

L’[audit](../results/sft-diagnostic-32-64-data-audit.json) vérifie les 96
réponses par l’oracle, leur réencodage exact et les identités des splits.
Les anciennes sondes et les holdouts ne servent pas à entraîner ce modèle.

## Budget fixe et réutilisation du pipeline

On repart du **checkpoint généraliste 94 M / 50 M tokens**, avec un optimiseur
neuf. Tous les poids sont entraînés, BF16 pour les opérations éligibles,
poids/AdamW FP32 et évaluation FP32. Architecture, tokenizer et contexte
restent inchangés. La [configuration](sft_diagnostic_config.json) fixe :

- **100 passes, 400 mises à jour**, batch 8, seed 0 ;
- 3 200 présentations d’exemples, **7 000 cibles réponse/EOS**, 101 600 tokens non paddés ;
- AdamW, taux maximal `3e-4`, warmup 20 mises à jour, cosinus jusqu’à `3e-5` ;
- weight decay nul et clipping du gradient à 1 ;
- sauvegarde et mesures toutes les 100 mises à jour ;
- point final fixé à 400, sans arrêt anticipé ni sélection sur le dev.

Le taux d’apprentissage et la répétition sont volontairement plus élevés
que dans l’essai Dolly pour contrôler la mémorisation. Ce n’est **pas une
ablation à une seule variable**, ni une recette choisie pour préserver les
capacités générales.

Le runner SFT existant est réutilisé : mêmes batches, même loss, mêmes
générations et mêmes sauvegardes de reprise. Seule la sélection d’exemples
d’évaluation accepte désormais un split complet sans sous-échantillonnage
par catégories Dolly.

**Convention des métriques du runner pour ce diagnostic :**

- `probes` = génération sur les **32 exemples TRAIN**, pour mesurer la mémorisation ;
- `dev_generation` = génération sur les **64 exemples nouveaux** ;
- `train_response` = loss sur les 32 exemples train complets ;
- `dev_response` = loss sur les 64 exemples dev complets.

Le fichier `train-probes.json` contient exactement les prompts et réponses
du train. Son score ne doit jamais être présenté comme une validation
indépendante. Les 64 exemples dev ne participent à aucune mise à jour.

## Évaluation et interprétation prévues

Génération greedy avant/après sur les 32 + 64 prompts, limitée à 8 nouveaux
tokens, EOS compris. Toutes les réponses de référence tiennent dans
3 tokens avec EOS. Le prompt est fourni seul, sans réponse attendue.
La correspondance exacte est normalisée comme dans le premier SFT ;
la terminaison à EOS est comptée séparément.

Le critère de mémorisation fixé avant le run est **32/32 réponses exactes**.
Les scores sur les 64 cas nouveaux seront publiés globalement et par tâche,
sans choisir ensuite un seuil favorable. Un succès train suivi d’échecs dev
pointerait vers une limite de généralisation sur ce test ; un échec train
motiverait une inspection des réglages et du pipeline, sans prouver un bug.

Le même dev général est mesuré avant/après sur ses **999 999 cibles** ;
l’échantillon fixe de 32 768 cibles est suivi en cours d’entraînement.
Cela mesure les dommages éventuels du test de mémorisation. Les devs
diagnostiques ne remplacent pas les holdouts réservés, qui ne sont pas évalués.

## Commandes

Depuis la racine du dépôt, dans l’environnement Python existant :

```sh
python -m experiments.sft_diagnostic_data \
  --output-dir data/sft-diagnostic-32-64-v1

python -m reimplementation.train_sft --device cuda \
  --config experiments/sft_diagnostic_config.json \
  --init-from runs/simple-baseline-2fsdgx0q/model.pt \
  --data-dir data/sft-diagnostic-32-64-v1 \
  --probes data/sft-diagnostic-32-64-v1/train-probes.json \
  --general-data-dir data/pretraining-50m-v1
```

Le modèle généraliste reste intact. La reprise utilise `--resume` à la place
de `--init-from`, avec le même fichier `--probes`. Le modèle final est
sauvegardé avant les évaluations ; AdamW et les gradients sont libérés avant
l’évaluation finale.
