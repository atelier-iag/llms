# Validation du lancement OLMo 3 sur un GPU

Date : 2026-09-10.

Environnement testé : RTX 4060 Laptop 8 Go, WSL, pilote Windows 536.90,
PyTorch 2.10.0+cu128, NCCL 2.27.5, OLMo-core épinglé à
`92870a33c3fee060d57c3faec52b0b369ece85a6`.

Le lanceur amont initialise un groupe distribué même avec `--train-single`.
Dans cet environnement, le lancement normal plante dans NCCL à l’initialisation
du modèle. Avec `--train-single`, le calcul avant/arrière passe, mais la réduction
de loss de SkipStepAdamW déclenche ensuite le même plantage NCCL.

Le lanceur du laboratoire utilise désormais `backend=None` avec `--train-single` :
le calcul reste sur GPU et les réductions sont locales. Le pilote, les paquets,
le modèle, l’optimiseur, les données et le submodule n’ont pas été modifiés.

Vérifications :

- 7 tests du lanceur réussis ; vérification Ruff réussie.
- Entraînement réel avec les paramètres par défaut : 100 étapes, 102 400 tokens,
  code de sortie 0 et message `Training complete`.
- Loss d’entraînement affichée : 11,55 à l’étape 1 ; 5,511 à l’étape 100.
- Validation sur 50 batches : CE loss 7,744 ; perplexité 2 307.
- Checkpoints des étapes 0 et 100 sauvegardés.

Le journal local est dans `runs/diagnostic-python-e253k41r/python-fixed.log`.
Ces mesures vérifient le fonctionnement du lancement ; ce court entraînement
et cette validation ne constituent pas une évaluation holdout.
