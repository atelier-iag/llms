# Scaling de la taille : 59 M contre 94 M de paramètres

**État au lancement du 25 septembre 2026 à 13 h 49 (Paris) : run 59 M en cours.**
La référence 94 M est terminée. Les deux modèles sont entraînés sur les mêmes
50 M de tokens ; les résultats de qualité du nouveau modèle restent à mesurer.

[Protocole](../experiments/model_scaling.md),
[configuration 59 M](../experiments/pretraining_59m_params_50m_tokens_config.json),
[référence 94 M](pretraining-50m.md),
[benchmark préalable](model-size-benchmark.md).

| Mesure | Modèle 59 M | Modèle 94 M, mesuré |
| --- | ---: | ---: |
| Paramètres | 58 948 864 | 94 124 928 |
| Largeur / MLP | 256 / 1 024 | 384 / 1 536 |
| Cibles d’entraînement prévues | 49 999 999 | 49 999 999 |
| Mises à jour prévues | 24 415 | 24 415 |
| Loss finale, dev complet | À mesurer | 5,216053 |
| Perplexité finale, dev complet | À mesurer | 184,205643 |
| Boucle d’entraînement, minutes | À mesurer | 57,60 |

Le fichier train, le dev, les fenêtres et leur ordre sont identiques. Les hashes
du manifeste et des fichiers ainsi que les offsets d’évaluation ont été comparés
à la référence avant lancement. Le holdout reste réservé et n’est pas évalué.

La configuration ne change que le nom, la largeur du modèle et celle du MLP.
Les huit blocs, RoPE, GQA 8Q/2KV, BF16/SDPA, le tokenizer, le contexte, le batch,
la seed et l’optimiseur sont conservés. Le budget de tokens identique permet de
garder aussi exactement le même warmup et le même calendrier cosinus. Les poids
sont initialisés aléatoirement pour la nouvelle taille ; aucune reprise des
poids du modèle 94 M n’est utilisée.

Le benchmark court préalable a exercé le vrai modèle 59 M en BF16/SDPA avec
des poids et gradients finis. Les contrôles avant lancement confirment les
58 948 864 paramètres, les 24 415 mises à jour et l’identité des données avec
la référence. L’estimation reste **45 à 55 minutes au total** ; elle sera
remplacée par les coûts mesurés à la fin.

Contrôle du démarrage : validation initiale complète terminée sur 999 999 cibles,
avec une loss finie de 11,671055. Les 200 premières mises à jour ont traité
409 600 cibles sans erreur ; la loss moyenne des pas 101 à 200 vaut 8,387663.
Ces mesures vérifient le départ du run et ne constituent pas ses résultats finaux.

Commit du code lancé : `efbc7fbe63b2c25ff6201e25d54d4cd981fe5e26`, publié sur
`main`. Le dossier `runs/model-59m-launch-gszxu42i/` conserve la commande, les
horodatages, le manifeste, la copie des métriques de référence, le snapshot des
sources et leurs SHA-256. Les checkpoints et journaux complets restent locaux.
Le run écrit dans `runs/simple-baseline-ng4b6qq_/`, avec une sauvegarde de reprise
atomique toutes les 1 000 mises à jour et à la fin. Les 28 sources Python communes
au snapshot du run de référence 94 M sont inchangées.

La comparaison finale portera sur les mêmes 999 999 cibles dev, les courbes, les
trois générations greedy, le temps et la mémoire. Une seule seed et une recette
partagée entre deux tailles donnent une première mesure à budget de tokens fixé,
sans recherche d’hyperparamètres propres à chaque taille ni conclusion générale
sur une taille optimale.
