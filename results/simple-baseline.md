# Baseline de la réimplémentation simple

Expérience du 18 septembre 2026, avant l’ajout des mécanismes du jalon 3.
**En cours de mesure.** Le test de 200 mises à jour précédent validait le pipeline ;
cette expérience établit une référence avec un modèle plus large et un passage
complet sur le corpus existant.

## Recette fixée avant l’entraînement

[Configuration versionnée](../reimplementation/baseline_config.json) et
[script d’entraînement](../reimplementation/train_baseline.py).

| Élément | Valeur |
|---|---:|
| Vocabulaire | 100 278 |
| Dimension d’embedding / du modèle | 384 |
| Matrice d’embedding | 100 278 × 384 = 38 506 752 paramètres |
| Blocs Transformer | 8 |
| Têtes par bloc | 8 × 48 dimensions |
| Dimension intermédiaire du MLP | 1 536 |
| Paramètres dans les blocs | 18 880 512 |
| Paramètres totaux | 95 894 400 |
| Longueur du contexte | 256 |
| Séquences par batch complet | 8 |
| Passages sur le corpus | 1 |
| Cibles d’entraînement uniques | 18 999 999 |
| Mises à jour | 9 279 |

Les matrices d’embedding et de sortie sont indépendantes. Pas de RoPE, de
position explicite, de Q/K normalization, de GQA ou de mixed precision. Modèle
initialisé aléatoirement, calcul FP32, seed 0. Le tokenizer Dolma 2 reste identique
à celui des données, épinglé dans le code.

AdamW avec learning rate maximal de `3e-4`, betas `(0.9, 0.95)`, weight decay `0.1`
sur tous les paramètres, warmup linéaire de 100 mises à jour, décroissance cosinus
jusqu’à `3e-5`, clipping de la norme du gradient à 1.0. La recette sera identique
pour une comparaison future avec RoPE.

## Données et protocole

Les fichiers locaux de la baseline contiennent 19 millions de tokens de train et
1 million de validation. Ils restent en lecture seule ; leurs empreintes SHA-256
sont vérifiées avant et après. Aucun nouvel accès au holdout final.

Les fenêtres de train sont mélangées une fois, sans remise. Elles partagent un
token de contexte aux frontières, mais chaque cible est utilisée exactement une
fois. Le tout premier token n’a pas de contexte précédent ; les 18 999 999 autres
sont tous prédits, y compris les 191 cibles de la dernière fenêtre partielle.
La mémoire du modèle ne se prolonge pas d’une fenêtre à la suivante. Les documents
sont séparés par EOS ; une fenêtre peut traverser cette frontière.

Le fichier de validation complet est évalué avant et après : 999 999 cibles,
chacune comptée une fois, avec remise à zéro du contexte entre fenêtres. Pour
suivre les courbes, les mêmes 128 fenêtres fixes par split (32 768 cibles) sont
évaluées au départ, toutes les 1 000 mises à jour, et à la fin. Cette validation
sert au développement, pas à une évaluation finale indépendante.

Le script de préparation évite de partager un document entre train et validation,
mais cette expérience ne constitue pas un audit de doublons du corpus.

## Calibration sur le GPU local

RTX 4060 Laptop, 8 Go de VRAM, contexte de 256 tokens. Mesures courtes après trois
mises à jour de chauffe, avec le modèle complet et l’optimiseur ; ces poids de
calibration ne sont pas utilisés dans le véritable entraînement.

| Batch | Débit mesuré, cibles/s | Pic alloué par PyTorch | Durée estimée du passage |
|---|---:|---:|---:|
| 2 | 3 517 | 2,20 Go | 90 min |
| 4 | 7 126 | 3,23 Go | 44 min |
| 8, retenu | 9 583 | 5,28 Go | 33 min |

Débits mesurés sur 10 mises à jour pour les batches 2 et 4, sur 20 pour le batch 8.
Ce sont des estimations, hors évaluations et sauvegardes ; elles peuvent varier
avec l’état et la température du GPU. Le pic alloué n’est pas la mémoire GPU totale.

## Reproduire et retrouver le run

Depuis la racine du dépôt, avec l’environnement existant activé :

```sh
python -m reimplementation.train_baseline --device cuda
python -m pytest tests/ -q --disable-warnings
```

Run local : `runs/simple-baseline-34yq5mdv/`. Les checkpoints restent ignorés par
Git. Les métriques et courbes finales seront conservées ici sous forme légère.

Les 45 tests du laboratoire passent, dont les vérifications de couverture complète
des cibles, validation sans mise à jour, clipping, calendrier de learning rate,
sauvegarde et restitution des scores après rechargement.
