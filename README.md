# Laboratoire LLM

Ce laboratoire étudie les modèles de langage comme voie vers l’AGI : faire
fonctionner une référence OLMo, en réimplémenter les mécanismes centraux, puis
mesurer les améliorations et leurs limites.

Ce README regroupe les objectifs, les [jalons du parcours](#jalons) et les critères de maîtrise.
Le [contrat commun des laboratoires](https://github.com/atelier-iag/.github/blob/main/LAB_CONTRACT.md), centralisé dans le dépôt `.github` de l’organisation, fixe la progression :

**baseline → reimplementation → modern improvement → ablation → holdout.**

## Objectif

S’approprier le fonctionnement pratique d’un petit modèle de fondation de type **OLMo-like**, de l’entraînement initial jusqu’au post-entraînement, et apprendre à utiliser un LLM avec des documents externes grâce au **RAG**.

## Jalons

Les huit jalons (*milestones*) structurent le parcours « Modèles de fondation / scaling », complété par les bases du RAG.
Chaque jalon est un objectif important, atteint grâce aux réalisations ci-dessous.

Le **[jalon 4 — RAG minimal](#jalon-4--comprendre-et-construire-un-rag-minimal)**
peut être commencé dès maintenant, en parallèle du jalon 3, sans attendre les
expériences de préentraînement, de scaling ou d’adaptation.

### Jalon 1 — Faire tourner une baseline existante

- entraîner un très petit modèle ;
- comprendre le pipeline `données → tokens → batches → modèle → loss → optimisation`.

### Jalon 2 — Réimplémenter le cœur du modèle

- embeddings ;
- self-attention causale ;
- MLP ;
- résidus + normalisation ;
- Transformer decoder ;
- boucle d’entraînement.

**Réimplémentation fonctionnelle.** Le pipeline
entraîne notre modèle sur des tokens réels, mesure une validation séparée,
sauvegarde et recharge les poids, puis génère du texte.
Voir les [commandes](reimplementation/README.md#complete-pipeline-on-real-tokens)
et les [résultats mesurés](results/reimplementation-corpus.md).
Une **baseline de notre propre modèle simple est aussi mesurée** : dimension 384,
8 blocs, 95,9 millions de paramètres et un passage complet sur le corpus existant.
La loss sur la validation complète passe de **11,68 à 5,42** ; le texte généré reste
fortement répétitif. Voir le [protocole, les courbes et les mesures](results/simple-baseline.md).
Le jalon 2 dispose ainsi d’un code vérifié et d’une référence expérimentale pour
comparer les mécanismes du jalon 3 à taille, données et budget identiques.

### Jalon 3 — Ajouter les mécanismes modernes

- RoPE ;
- GQA ;
- mixed precision ;
- optimisations d’entraînement utiles.

**RoPE est implémenté et mesuré.** À taille, données et budget identiques, la
perplexité de validation passe de **225,76 à 199,10** (−11,81 %) sur cette seed.
Le temps total passe de 38,44 à 55,44 minutes et les générations restent répétitives.
Voir la [comparaison complète](results/rope-baseline.md) et la
[démo RoPE](experiments/README.md).

**GQA est implémenté et mesuré** : 8 têtes Q partagent 2 groupes K/V, avec RoPE conservé.
La variante compte 94,12 millions de paramètres, soit 1,85 % de moins. Au même
budget de tokens, sa perplexité est de **199,41**, contre **199,10** avec RoPE seul :
une qualité très proche sur cette seed. Voir la [comparaison complète](results/gqa-baseline.md)
et le [code et le protocole GQA](experiments/gqa.md).

### Jalon 4 — Comprendre et construire un RAG minimal

Le **RAG** (*Retrieval-Augmented Generation*, génération augmentée par recherche)
consiste à rechercher des passages dans des documents, puis à les fournir au LLM
dans son contexte pour l’aider à répondre. Dans ce premier exercice, les poids
du modèle restent fixes.

- comprendre le pipeline `documents → passages → index → recherche → contexte → réponse avec sources` ;
- distinguer RAG, fine-tuning et simple ajout de documents au prompt ;
- préparer un petit corpus, découper les documents et conserver leurs références ;
- construire une recherche lexicale simple, puis une recherche par embeddings de passages et similarité cosinus ;
- assembler soi-même le contexte et le prompt, avec une réponse sourcée ou une abstention si les documents ne suffisent pas ;
- comparer le même LLM sans recherche, avec RAG et avec les passages de référence, pour distinguer erreurs de recherche et de génération ;
- mesurer la recherche et la qualité des réponses sur des questions de développement, puis sur un holdout réservé.

**Exercice minimal :** une dizaine de documents, une vingtaine de questions et
un modèle déjà entraîné capable de suivre une consigne. Notre modèle pédagogique
reste encore trop répétitif pour servir de générateur principal à cet exercice.
Voir le [parcours pratique, les livrables et les critères de maîtrise](experiments/rag.md).

**Première démo disponible :** `python -m experiments.rag_demo` recherche des
passages dans nos comptes rendus et affiche le prompt avec ses sources.
Elle s’arrête avant l’appel au LLM ; la génération et l’évaluation restent à faire.

Ce jalon couvre les bases du RAG dans le laboratoire LLM ; les recherches
itératives pilotées par un agent et la mémoire persistante seront approfondies
dans la voie « Agents et outils ».

### Jalon 5 — Faire un mini-préentraînement propre

- corpus préparé ;
- splits train/dev/holdout ;
- suivi de la loss, perplexité, temps et coût.

### Jalon 6 — Étudier le scaling

- faire varier taille du modèle, quantité de données et compute ;
- comparer les courbes obtenues.

### Jalon 7 — Pratiquer l’adaptation

- continued pretraining ;
- SFT ;
- éventuellement une méthode simple de post-training.

### Jalon 8 — Faire des ablations

- retirer ou modifier certains mécanismes ;
- mesurer leur effet ;
- tester sur le holdout ;
- documenter les échecs et différences observées.

## Critère de maîtrise

**Baseline → réimplémentation → amélioration moderne → ablation → holdout.**

Le code produit servira ensuite de premier laboratoire à intégrer au **workbench commun**.

## Structure

```text
llms/
├── README.md                 # Point d’entrée, objectifs, jalons et critères de maîtrise
├── baseline/                 # Implémentations de référence et mesures
│   └── olmo3/
│       ├── OLMo-core/        # Submodule Git de la référence AllenAI
│       └── mini/             # Préparation des données et entraînement réduit
├── reimplementation/         # Réimplémentation des mécanismes centraux
├── experiments/              # Améliorations modernes, ablations et expériences
├── evaluation/               # Protocoles et scripts, notamment holdout
├── results/                  # Résultats légers et conclusions
└── tests/                    # Tests du code du laboratoire
```

Les dossiers encore vides contiennent un `.gitkeep`.

## Baseline OLMo 3

Cloner le dépôt avec son submodule :

```sh
git clone --recurse-submodules https://github.com/atelier-iag/llms.git
```

Dans un clone existant, depuis la racine du dépôt :

```sh
git submodule update --init --recursive
```

La référence reste dans [baseline/olmo3/OLMo-core](baseline/olmo3/OLMo-core).
Les scripts [prepare_data.py](baseline/olmo3/mini/prepare_data.py) et
[train.py](baseline/olmo3/mini/train.py) utilisent le dossier
[mini/data/](baseline/olmo3/mini/data). Les fichiers générés `train.npy` et
`val.npy` y restent locaux et ignorés par Git.

### Entraînement sur un GPU

Depuis la racine du dépôt, dans le terminal WSL et avec l’environnement installé :

```sh
source .venv/bin/activate
python baseline/olmo3/mini/train.py \
  --save-folder runs/olmo3-60m-valtest \
  --train-single
```

`--train-single` utilise directement le GPU sans initialiser de groupe distribué
ni NCCL. Le lanceur local corrige ce comportement pour la version épinglée
d’OLMo-core ; les lancements distribués utilisent toujours le lanceur amont.
La configuration actuelle entraîne pendant 100 pas d’optimisation, puis évalue sur la
validation. Un checkpoint présent dans `--save-folder` est repris automatiquement ;
choisir un nouveau dossier pour démarrer un nouvel entraînement.

Vérifier la configuration sans lancer l’entraînement en ajoutant `--dry-run`.
Les tests du lanceur se lancent avec `python -m pytest tests/test_mini_launcher.py`.
