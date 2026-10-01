# Clôture du laboratoire : protocole holdout final

Protocole fixé le **1er octobre 2026**, avant toute mesure sur les holdouts.
Le parcours s’arrête après cette évaluation et sa synthèse : **aucun nouvel
entraînement, réglage ou choix de checkpoint selon ces scores**.

Les acquis et limites du SFT sont déjà documentés. La construction d’un
autre jeu SFT et les optimisations supplémentaires sortent du périmètre
de cette clôture.

**Évaluation terminée le 1er octobre 2026 :** [résultats des holdouts](../results/final-holdout-v1.md)
et [bilan final du laboratoire](../results/bilan-final.md). Ces liens ont été
ajoutés après l’audit ; le protocole était figé au commit `09589cc` avant
l’ouverture des jeux réservés.

## Modèles retenus avant d’ouvrir les holdouts

La [configuration gelée](final_holdout_config.json) épingle les sept fichiers
de poids par SHA-256, leurs architectures, étapes et résultats historiques.
Le checkpoint **94 M / 50 M tokens** reste la référence généraliste,
retenue sur les expériences dev précédentes.

| Identifiant | Rôle dans le bilan | Holdouts évalués |
| --- | --- | --- |
| general_94m_50m | Référence généraliste | Général, Python, Dolly |
| general_94m_25m | Effet du budget de données | Général |
| general_59m_50m | Effet de la taille | Général |
| python_94m | Adaptation sur 5 M de tokens Python | Général, Python |
| dolly_94m | SFT sur 3 000 exemples Dolly | Général, Dolly |
| toy_sft_3e4 | Référence de l’ablation du taux SFT | Général |
| toy_sft_1e4 | Taux SFT divisé par trois | Général |

Les deux modèles du petit diagnostic SFT n’ont pas de holdout de consignes
distinct : leurs résultats de transfert restent des diagnostics de
développement. Leur évaluation générale finale mesure la dégradation du
modèle de langue, pas leur généralisation aux consignes.

Les essais NoPE/RoPE/GQA/BF16 du corpus ancien restent comparés sur leur
protocole historique. Ils ne sont pas évalués sur le holdout récent :
l’absence de chevauchement avec leur ancien train n’est pas garantie.
Le corpus général récent a précisément été associé à un redémarrage depuis
des poids aléatoires pour éviter cette contamination.

## Jeux réservés déjà préparés

| Jeu | Taille du holdout | Mesure principale |
| --- | ---: | --- |
| Général, Dolma filtré | 1 000 000 tokens | Loss et perplexité sur 999 999 cibles |
| Python, CodeParrot filtré | 250 000 tokens | Loss et perplexité sur 249 999 cibles |
| Dolly, réponses courtes | 300 exemples | Loss/perplexité sur 7 699 cibles réponse/EOS |

Les manifestes existaient avant les entraînements. Leurs SHA-256 et ceux
des fichiers holdout sont repris tels quels dans la configuration.
Aucun quota, filtre ou split n’est modifié. La copie du holdout général
dans le corpus 25 M est identique ; elle ne constitue pas un autre test.

La séparation repose sur les contrôles déjà documentés :
[corpus général](../results/pretraining-50m-data.md),
[Python](../results/continued-python-5m-data.md),
[Dolly](../results/sft-dolly-3k-data-audit.json).
La déduplication exacte, les groupes Dolly et la séparation par dépôt
Python ne garantissent pas l’absence de tous les quasi-doublons.

## Calcul et contrôles

Évaluation FP32, précision matricielle `highest`, contexte 256, batch 8,
un modèle à la fois, sans optimiseur. Les évaluateurs et le masque de
réponse SFT existants sont réutilisés.

Avant de lire un holdout, chaque checkpoint doit reproduire sa loss sur les
mêmes 128 fenêtres du dev général, soit 32 768 cibles. Tolérance absolue
fixée à `1e-6` pour la loss ; couverture exacte requise. Ces contrôles
valident le chargement et le chemin numérique, sans orienter le choix des
modèles. Les sept contrôles doivent passer avant la première mesure finale.

Les fichiers de texte sont parcourus complètement : chaque token sauf
le premier est prédit exactement une fois, y compris la dernière fenêtre
courte. Les fenêtres peuvent traverser EOS ; l’attention ne se réinitialise
pas aux frontières documentaires. Les conventions sont celles du dev.

Sur Dolly, le masque exclut prompt et padding, et inclut EOS.
La génération utilise **24 exemples holdout fixes, six par catégorie**,
choisis par `balanced_select` avec seed 0 et tri par hash des empreintes,
sans examiner les réponses générées. Même sélection pour les deux modèles,
greedy, au plus 81 nouveaux tokens, contexte total limité à 256.
Cette règle de sélection est figée avant la lecture du holdout.

Mesures de génération : correspondance exacte normalisée, F1 lexical,
arrêt à EOS, répétition de 4-grammes, scores par catégorie. Les 24 générations
ne représentent pas les 300 exemples du score de loss. Le score exact
n’accepte pas toutes les réponses sémantiquement valides ; le F1 n’est pas
un jugement de correction.

## Politique de clôture et publication

Le code et ce protocole sont commités avant le run. Le runner exige un
dépôt propre et vérifie les fichiers avant/après. Chaque résultat est
enregistré immédiatement ; un incident technique sera documenté, avec
conservation des résultats déjà produits et sans réglage des modèles.

Les résultats et leur provenance sont publiés, y compris les échecs.
Les textes externes et les réponses Dolly détaillées restent locaux ;
seuls scores, IDs et empreintes entrent dans Git.

L’évaluation consomme les holdouts pour ce cycle : ils ne devront plus
être présentés comme des jeux restés inconnus si le travail reprend.
Une future optimisation fondée sur ce bilan demandera une nouvelle
évaluation indépendante pour soutenir une nouvelle conclusion.

## Exécution

Avec les checkpoints et les trois corpus locaux déjà préparés :

```sh
python -m evaluation.final_holdout --device cuda \
  --config experiments/final_holdout_config.json
```

Les résultats sont écrits dans un nouveau dossier `runs/final-holdout-*/`.
Aucune commande de cette clôture ne relance les entraînements.
