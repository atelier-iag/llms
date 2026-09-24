# Laboratoire LLM

Ce laboratoire étudie les modèles de langage comme voie vers l’AGI : faire
fonctionner une référence OLMo, en réimplémenter les mécanismes centraux, puis
mesurer les améliorations et leurs limites.

Ce README regroupe les objectifs, les [jalons du parcours](#jalons) et les critères de maîtrise.
Le [contrat commun des laboratoires](https://github.com/atelier-iag/.github/blob/main/LAB_CONTRACT.md), centralisé dans le dépôt `.github` de l’organisation, fixe la progression :

**baseline → reimplementation → modern improvement → ablation → holdout.**

## Objectif

S’approprier le fonctionnement pratique d’un petit modèle de fondation de type **OLMo-like**, de l’entraînement initial jusqu’au post-entraînement.

## Jalons

Les sept jalons (*milestones*) structurent le parcours « Modèles de fondation / scaling ».
Chaque jalon est un objectif important, atteint grâce aux réalisations ci-dessous.

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

**La mixed precision BF16 est intégrée**, avec poids et optimiseur FP32, calculs
matriciels éligibles en BF16 et évaluation FP32. La configuration conserve le
modèle GQA et son budget de tokens. Voir le [fonctionnement et le protocole](experiments/mixed_precision.md).
Le [premier benchmark court](results/mixed-precision-benchmark.md) mesure un débit
médian supérieur de 10,64 % et un pic alloué inférieur de 6,14 % sur le GPU local.
La qualité après entraînement complet reste à comparer.

### Jalon 4 — Faire un mini-préentraînement propre

- corpus préparé ;
- splits train/dev/holdout ;
- suivi de la loss, perplexité, temps et coût.

### Jalon 5 — Étudier le scaling

- faire varier taille du modèle, quantité de données et compute ;
- comparer les courbes obtenues.

### Jalon 6 — Pratiquer l’adaptation

- continued pretraining ;
- SFT ;
- éventuellement une méthode simple de post-training.

### Jalon 7 — Faire des ablations

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
