# Bilan final du laboratoire LLM

**Cycle pédagogique clôturé le 1er octobre 2026.** Les réalisations techniques
et expérimentales des sept jalons sont terminées : référence OLMo,
réimplémentation, mécanismes modernes, préentraînement, scaling, adaptation,
ablations et évaluation finale.

L’objectif était de pratiquer la chaîne d’un petit modèle de fondation,
de l’entraînement initial au post-entraînement, en mesurant ses limites.
Le résultat est un laboratoire reproductible et un ensemble de modèles
expérimentaux. Leur génération reste répétitive ou fragile ; le projet
ne fournit pas un assistant généraliste fiable.

## Les sept jalons et leurs preuves

| Jalon | Réalisation | Preuves principales |
| --- | --- | --- |
| 1. Baseline existante | OLMo entraîné sur un GPU, validation et checkpoints | [Lancement réel](single-gpu-launch.md) |
| 2. Réimplémentation | Embeddings, attention causale, SwiGLU, normalisation, decoder, loss, entraînement, sauvegarde et génération | [Code expliqué](../reimplementation/README.md), [baseline mesurée](simple-baseline.md) |
| 3. Mécanismes modernes | RoPE, GQA, BF16 et SDPA intégrés, testés et comparés | [RoPE](rope-baseline.md), [GQA](gqa-baseline.md), [BF16](bf16-baseline.md), [SDPA](sdpa-benchmark.md) |
| 4. Mini-préentraînement | Corpus traçable, splits séparés, 50 M tokens, dev et holdout | [Préentraînement](pretraining-50m.md), [évaluation finale](final-holdout-v1.md) |
| 5. Scaling | Comparaisons 25/50 M tokens et 59/94 M paramètres, avec temps et mémoire | [Données](data-scaling-25m-50m.md), [taille](model-scaling-59m-94m.md) |
| 6. Adaptation | Continued pretraining Python, SFT Dolly et diagnostics de mémorisation/transfert | [Python](continued-python-5m.md), [Dolly](sft-dolly-3k.md), [transfert](sft-transfer-v1.md) |
| 7. Ablations et clôture | Changements contrôlés, effets et échecs documentés, checkpoints figés et holdouts évalués | [Taux SFT](sft-learning-rate-1e-4.md), [diversité](sft-diversity-64-64.md), [holdouts finaux](final-holdout-v1.md) |

Le statut décrit les livrables et les expériences. Les rapports, démonstrations
et tests permettent de reprendre chaque mécanisme pour approfondir sa maîtrise.

## Résultats à retenir

**Les mécanismes et le protocole comptent.** Sur le premier corpus, RoPE
réduit la perplexité dev de 225,76 à 199,10. GQA réduit légèrement le nombre
de paramètres avec une qualité proche sur cette seed. BF16 conserve une
qualité proche ; le benchmark SDPA mesure un gain de débit de 80,14 %.
Ces mesures ont des périmètres différents, explicités dans leurs rapports.

**Davantage de données et de capacité aident dans nos comparaisons.**
Sur le holdout général final, la perplexité passe de **226,23 à 141,06**
avec 25 M puis 50 M tokens à modèle fixé, et de **160,20 à 141,06**
avec 59 M puis 94 M paramètres à données fixées. Deux points par axe et
une seule seed constituent une exploration du scaling, pas une loi générale.

**L’adaptation a un coût hors domaine.** Après 5 M tokens Python, la
perplexité du holdout Python passe de **950,89 à 64,80**, tandis que celle
du holdout général monte de **141,06 à 303,52**. Prédire mieux les tokens
Python ne garantit pas de produire du code correct.

**Une baisse de loss SFT ne suffit pas à suivre les consignes.**
Sur le holdout Dolly, la perplexité des réponses passe de **431,11 à 171,83**.
Le modèle s’arrête plus souvent à EOS, mais reste à **0/24 réponses exactes**
sur l’échantillon de génération fixé.

**La mémorisation peut masquer une généralisation fragile.**
Le petit SFT à taux réduit réussit 64/64 exemples train et 61/64
combinaisons dev. Les diagnostics de transfert donnent ensuite
**59/64** avec des reformulations, **6/64** avec l’ordre des phrases inversé
et **0/64** avec de nouvelles valeurs. Ce sont des mesures de développement,
distinctes des holdouts finaux.

**Une ablation peut conclure à un échec ou à un compromis.**
Doubler les exemples distincts à budget égal ne donne pas de gain global
sur le diagnostic. Réduire le taux d’apprentissage améliore ensuite le
compromis : le holdout général passe de 1 020,78 à 201,63, mais reste
moins bien prédit que par le modèle généraliste à 141,06.

## Décision de clôture

La référence généraliste conservée est le modèle **94 124 928 paramètres,
préentraîné sur 50 M tokens**, avec RoPE, GQA, BF16 à l’entraînement,
attention SDPA et contexte de 256 tokens. Ce choix précédait les scores holdout.

Les autres checkpoints restent disponibles comme traces des comparaisons.
Leurs chemins, hashes et architectures sont figés dans la
[configuration finale](../experiments/final_holdout_config.json).
Aucun modèle n’a été réentraîné, prolongé ou sélectionné après consultation
des scores finaux.

Les nouvelles données SFT, les optimisations supplémentaires et les études
plus vastes sont des approfondissements possibles, **hors du périmètre
nécessaire pour terminer ce cycle**. Aucune de ces suites n’est lancée.
L’intégration au workbench commun appartient à une étape ultérieure de
l’atelier ; elle n’est pas implémentée par cette clôture.

Les trois holdouts ont désormais servi à l’évaluation finale. Une future
optimisation informée par ces résultats devra disposer d’une nouvelle
évaluation indépendante pour soutenir une nouvelle conclusion.

## Vérifier et reprendre les résultats

La suite complète contient **225 tests réussis**. Le dernier run évalue
sept checkpoints et onze couples modèle/jeu ; il vérifie les identités
des fichiers, les comptes de cibles, la reproduction exacte des contrôles
dev et l’absence de modification des poids.

Depuis la racine du dépôt, avec l’environnement local existant :

```sh
source .venv/bin/activate
python -m pytest tests -q
```

Consulter les [mesures finales](final-holdout-v1.md) et leur
[provenance](final-holdout-v1-execution.json), puis les protocoles liés
dans le tableau des jalons. Ils enregistrent les recettes, budgets,
sources épinglées et limites de chaque expérience.

Avec les checkpoints et corpus locaux conservés, reproduire l’évaluation
déjà publiée, sans lui attribuer le statut d’un nouveau test inconnu :

```sh
python -m evaluation.final_holdout --device cuda \
  --config experiments/final_holdout_config.json
```

Essayer la génération du checkpoint généraliste :

```sh
python -m reimplementation.generate runs/simple-baseline-2fsdgx0q/model.pt \
  --device cuda --prompt "The purpose of science is" --max-new-tokens 32
```

Les poids, données externes et générations Dolly détaillées restent locaux
dans les dossiers ignorés par Git. Le dépôt contient le code, les tests,
les configurations, les identités des artefacts, les mesures et les
conclusions nécessaires pour examiner le travail.
