# Scaling des données : 25 M contre 50 M

**État au lancement du 24 septembre 2026 à 15 h 55 (Paris) : run 25 M en cours.**
Le run 50 M est terminé. Aucune conclusion comparative de qualité n’est encore
tirée ; les métriques finales du run 25 M restent à mesurer.

[Protocole](../experiments/data_scaling.md),
[configuration 25 M](../experiments/pretraining_25m_config.json),
[audit du sous-ensemble](pretraining-25m-data.md),
[résultats 50 M](pretraining-50m.md).

| Mesure | 25 M | 50 M, mesuré |
| --- | ---: | ---: |
| Paramètres | 94 124 928 | 94 124 928 |
| Cibles d’entraînement prévues | 24 999 999 | 49 999 999 |
| Mises à jour prévues | 12 208 | 24 415 |
| Loss finale, dev complet | À mesurer | 5,216053 |
| Perplexité finale, dev complet | À mesurer | 184,205643 |
| Boucle d’entraînement, minutes | À mesurer | 57,60 |

Les deux runs partent de poids aléatoires avec seed 0. Le modèle, les modules
d’entraînement, le tokenizer et la recette d’optimisation sont communs. Le
warmup et le cosinus suivent la durée de chaque budget : warmup 125/250 mises
à jour, puis décroissance vers le même plancher au dernier pas. Les fenêtres
train sont recalculées sur les fichiers respectifs.

Le sous-ensemble 25 M contient 6,25 M de tokens par domaine. Le dev est
**strictement identique** à celui de la référence 50 M, avec 999 999 cibles
pour l’évaluation complète. Le holdout est aussi conservé à l’identique et
reste réservé, sans score de modèle. Les 162 tests et l’audit passent avant
le lancement du run 25 M.

Le contrôle au démarrage retrouve **exactement les mêmes métriques dev
initiales** que la référence 50 M, sur le dev complet (loss 11,687100346301015)
et sur l’échantillon fixe. L’échantillon train diffère, conformément au changement
de fichier train.
Les 200 premières mises à jour passent sans erreur (409 600 cibles) ; la loss
moyenne des pas 101 à 200 est finie, à 7,820782. C’est un contrôle de démarrage,
pas une évaluation de qualité à modèle fixe.

Le code du lancement est celui du commit
`fb5f98080e17df4217cea964104fc7958d35c243`, publié sur `main`. Les 28 sources
Python communes au snapshot 50 M sont inchangées. Le lanceur conserve les
sources exécutées, leurs SHA-256, la commande, le manifeste et une copie exacte
des métriques de référence : `runs/pretraining-25m-launch-ar5qxr1r/`.
Le run écrit dans `runs/simple-baseline-korcrdq1/` et sauvegarde une reprise
atomique toutes les 1 000 mises à jour.
Le checkpoint final et les journaux complets resteront locaux, ignorés par Git.

Estimation avant départ : environ **30 à 35 minutes**, selon le débit soutenu.
Après achèvement, comparer le dev complet, les courbes dev, le coût réel et les
trois générations greedy communes. Deux budgets et une seule seed fournissent
une première comparaison ; ils ne suffisent pas à ajuster une loi de scaling.
