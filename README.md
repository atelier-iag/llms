# Laboratoire LLM

Ce laboratoire étudie les modèles de langage comme voie vers l’AGI : faire
fonctionner une référence OLMo, en réimplémenter les mécanismes centraux, puis
mesurer les améliorations et leurs limites.

Les objectifs et critères de maîtrise sont définis dans [PLAN.md](PLAN.md).
Le [contrat commun des laboratoires](https://github.com/atelier-iag/.github/blob/main/LAB_CONTRACT.md), centralisé dans le dépôt `.github` de l’organisation, fixe la progression :

**baseline → reimplementation → modern improvement → ablation → holdout.**

## Structure

```text
llms/
├── README.md                 # Point d’entrée du laboratoire
├── PLAN.md                   # Objectifs et critères de maîtrise
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
La configuration actuelle entraîne pendant 100 étapes, puis évalue sur la
validation. Un checkpoint présent dans `--save-folder` est repris automatiquement ;
choisir un nouveau dossier pour démarrer un nouvel entraînement.

Vérifier la configuration sans lancer l’entraînement en ajoutant `--dry-run`.
Les tests du lanceur se lancent avec `python -m pytest tests/test_mini_launcher.py`.
