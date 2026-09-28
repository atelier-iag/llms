# Jalon 6 : premier SFT sur des instructions courtes

On reprend le checkpoint **94 M généraliste préentraîné sur 50 M de tokens**,
avant l’adaptation Python. L’objectif est d’apprendre le mécanisme du SFT et
de mesurer si le modèle suit mieux des consignes, tout en surveillant le dev
général. La perte de généralité observée dans le
[continued pretraining Python](../results/continued-python-5m.md) motive ce choix.

## Une consigne, un contexte éventuel, une réponse

Chaque exemple est présenté avec les séparateurs textuels suivants, sans
ajouter de tokens au vocabulaire :

```text
### Instruction:
What color is the box? Answer with one word.

### Context:
The box is blue. The bag is red.

### Response:
blue<|endoftext|>
```

Cet exemple est une sonde écrite pour ce laboratoire, réservée au diagnostic.
Il ne figure pas dans les exemples d’entraînement. La section Context est
omise quand elle est vide. Le prompt complet et la réponse sont tokenisés
séparément puis concaténés : cela fixe sans ambiguïté la frontière de réponse,
avec exactement les mêmes tokens de prompt à l’entraînement et à l’inférence.

Le décalage causal reste celui de la prédiction du prochain token. **Seuls les
tokens de réponse et son EOS sont supervisés.** La dernière position du prompt
prédit le premier token de réponse. Les cibles de prompt et de padding portent
`-100` et sont exclues du calcul de la loss. Cela n’empêche pas les gradients
de traverser les représentations du prompt qui servent à prédire la réponse.
La [documentation SFT de Hugging Face](https://huggingface.co/docs/trl/sft_trainer#train-on-completion-only)
décrit cette variante ; ici, elle est réimplémentée dans notre boucle PyTorch.

Les exemples restent dans des lignes de batch distinctes : aucun packing.
Le padding est ajouté **à droite**, jusqu’au plus long exemple du batch. Il
réutilise l’ID EOS comme valeur d’entrée, mais son masque dépend des positions,
pas de cette valeur : le véritable EOS final reste supervisé. Avec notre
attention causale, les tokens utiles ne peuvent pas voir les positions de
padding futures. Ce comportement est testé avec attention manuelle et SDPA.

## Source, sélection et limites de qualité

Source : [Databricks Dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k),
révision `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a`, fichier
`databricks-dolly-15k.jsonl`, 15 011 exemples. La carte décrit des contributions
humaines en anglais. Licence **CC BY-SA 3.0**, attribution Databricks et aux
contributeurs Wikipédia pour les passages concernés. Le fichier source reste
dans le cache local ; sa révision et son SHA-256 sont conservés.

Le premier filtre de longueur laissait 7 734 exemples parmi six catégories.
Une revue qualitative de **60 exemples, dix par catégorie**, a révélé des
erreurs factuelles, des ambiguïtés et des réponses hors sujet dans certaines
questions sans contexte. Les [IDs et décisions](sft_quality_review.json) sont
archivés. Cette revue a été effectuée par Codex, sans vérification factuelle
indépendante de tout le corpus.

Le premier essai retient donc **classification, closed QA, extraction et résumé**.
Les catégories open/general QA sont écartées, ainsi que brainstorming et
création littéraire. Onze IDs signalés dans la revue sont exclus explicitement,
dont trois appartiennent aux catégories finalement retenues. Les labels des
3 000 exemples conservés ne sont pas tous certifiés corrects.

Contraintes : instruction et réponse non vides ; réponse de 1 à 80 tokens ;
**prompt + réponse + EOS ≤ 256 tokens**. Les exemples trop longs sont rejetés,
sans tronquer leurs réponses. Les doublons normalisés exacts sont supprimés.
Des groupes réunissent les mêmes consignes ou contextes normalisés, ainsi que
les prompts dont les ensembles de trigrammes de mots ont un Jaccard ≥ 0,8,
avec fermeture transitive. Cette heuristique ne détecte pas tous les doublons
sémantiques ; des patrons de classification restent très répétitifs.

Le hash du groupe avec seed 0 détermine le split, selon des buckets 80/10/10.
La sélection parcourt ensuite les catégories à tour de rôle, avec un ordre
déterministe. Les catégories les moins fournies s’épuisent : le résultat
reste dominé par la classification, et n’est pas uniformément équilibré.

| Split | Exemples | Groupes | Tokens avec prompt et EOS | Cibles réponse, EOS compris |
| --- | ---: | ---: | ---: | ---: |
| Train | 3 000 | 2 876 | 361 590 | 82 266 |
| Dev | 300 | 287 | 39 740 | 7 518 |
| Holdout réservé | 300 | 277 | 41 278 | 7 699 |

Le train comprend 1 399 classifications, 770 closed QA, 541 extractions et
290 résumés. Le [manifeste](../results/sft-dolly-3k-manifest.json) et
l’[audit des fichiers](../results/sft-dolly-3k-data-audit.json) donnent les hashes,
la séparation des groupes et les comptes. Les 3 600 séquences ont été
retokenisées et comparées à leur source épinglée. Les 12 prompts diagnostiques
n’apparaissent pas exactement dans ces données. Le holdout SFT est préparé et
audité pour son intégrité, sans calculer de score dessus.

```sh
python -m reimplementation.prepare_sft \
  --output-dir data/sft-dolly-3k-v1 \
  --quality-review experiments/sft_quality_review.json
```

## Entraînement fixé avant les mesures

La [configuration](sft_dolly_config.json) garde l’architecture, le tokenizer,
le contexte 256, batch 8, RoPE/GQA/SDPA et BF16. Tous les paramètres sont
entraînés, avec poids/AdamW FP32 et évaluations FP32. AdamW repart de zéro,
pic `5e-5`, warmup 50 mises à jour, cosinus jusqu’à `5e-6`, weight decay 0,01
et clipping à 1. La seed est 0.

**Trois passes** donnent **1 125 mises à jour**, **9 000 présentations d’exemples**,
**1 084 770 tokens non paddés** et **246 798 cibles supervisées de réponse/EOS**.
Les 3 000 exemples uniques sont réutilisés ; il ne s’agit pas de 9 000 exemples
différents. Le point final est fixé à l’avance, sans arrêt anticipé ni sélection
du meilleur checkpoint sur le dev. La nouvelle recette et le nouvel objectif
ne constituent pas une ablation contrôlée du précédent préentraînement.

```sh
python -m reimplementation.train_sft --device cuda \
  --config experiments/sft_dolly_config.json \
  --init-from runs/simple-baseline-2fsdgx0q/model.pt \
  --data-dir data/sft-dolly-3k-v1 \
  --general-data-dir data/pretraining-50m-v1
```

Le runner refuse de remplacer un run existant. Il vérifie les métadonnées du
checkpoint, le manifeste SFT, ses fichiers train/dev et le dev général. Les
sauvegardes de reprise après chaque époque gardent les poids, AdamW, RNG,
compteurs de réponses et exemples, identités des données et historiques.

```sh
python -m reimplementation.train_sft --device cuda \
  --resume runs/sft-REPLACE/recovery.pt \
  --data-dir data/sft-dolly-3k-v1 \
  --general-data-dir data/pretraining-50m-v1
```

## Mesures avant/après

- Loss et perplexité sur les **7 518 cibles réponse/EOS du dev complet**.
  L’accuracy par token est calculée avec les tokens précédents de la réponse
  de référence fournis au modèle : elle ne mesure pas la réussite en génération libre.
- Génération greedy sur **24 exemples dev fixes**, six par catégorie,
  avec uniquement le prompt en entrée, au plus 81 nouveaux tokens, sans
  dépasser le contexte total de 256. Correspondance exacte normalisée,
  F1 lexical, arrêt à EOS et répétition de 4-grammes sont enregistrés.
  Le F1 mesure le recouvrement des mots, pas la correction sémantique.
- **12 sondes écrites pour le laboratoire**, fixées avant entraînement :
  extraction, classification par règle fournie et format simple. Les
  [consignes et réponses attendues](sft_probes.json) sont publiques. Elles
  constituent un petit diagnostic dev, pas un benchmark représentatif ou un holdout.
- Loss/perplexité sur les **999 999 cibles du même dev général**. Le score
  initial doit reproduire celui du checkpoint de référence.

La normalisation de la correspondance exacte applique NFKC, casse minuscule,
espaces réduits et suppression de la ponctuation terminale `. ! ?`. Elle ne
reconnaît ni paraphrases ni synonymes. Les 24 exemples générés sont trop peu
nombreux pour une estimation robuste de capacité générale.
Après chaque époque : dev réponse complet, échantillon train fixe de 128
exemples et 32 768 cibles générales fixes. **Aucun holdout n’est évalué**.

Le checkpoint final est rechargé et doit reproduire exactement les logits sur
la sonde de contrôle. Les données et le checkpoint initial doivent rester
inchangés. Les réponses Dolly détaillées restent dans les fichiers locaux de
génération ; les scores agrégés, IDs, sondes propres au laboratoire et mesures
sont publiés dans Git. Le SFT ne garantit pas de combler les lacunes du modèle.
