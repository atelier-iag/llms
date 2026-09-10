# Contrat commun des laboratoires AGI

Chaque dépôt explore une voie vers l’AGI (`llms`, `reasoning`, `agents`,
`world-models`, etc.) selon la même structure :

- `baseline/` : faire fonctionner et mesurer une implémentation de référence.
- `reimplementation/` : réimplémenter les mécanismes centraux.
- `experiments/` : améliorations modernes, ablations et autres expériences.
- `evaluation/` : protocoles et scripts d’évaluation, notamment holdout.
- `results/` : résultats légers et conclusions.
- `tests/` : tests du code du laboratoire.
- `PLAN.md` : objectifs et critères de maîtrise du laboratoire.
- `README.md` : point d’entrée du dépôt.

Progression commune :

**baseline → reimplementation → modern improvement → ablation → holdout.**

Comparer les étapes avec des protocoles cohérents, isoler l’effet des changements
par des ablations et réserver le holdout à l’évaluation finale. Documenter les
mesures, les conclusions et les limites dans `results/`.
