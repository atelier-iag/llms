# Préentraînement 50 M — premier lancement

**État au lancement le 24 septembre 2026 à 14 h 03 (Paris) : en cours.**
Les métriques finales ne sont pas encore disponibles. Le code exécuté est celui
du commit `eb5e971c5047d20c03e51121901c300c0a0ea7ff`, publié sur `main`.

Le modèle de 94 124 928 paramètres repart de poids aléatoires, avec RoPE, GQA
8Q/2KV, BF16 et SDPA. Le parcours prévu couvre **49 999 999 tokens cibles en
24 415 mises à jour**, avec 1 M de tokens dev et 1 M de tokens holdout réservé.
Voir le [protocole](../experiments/pretraining_50m.md), la
[configuration](../experiments/pretraining_50m_config.json) et l’[audit des données](pretraining-50m-data.md).

L’estimation avant lancement est d’environ 52 minutes de boucle d’entraînement,
hors évaluations et sauvegardes ; le temps réel sera mesuré à la fin. Les
152 tests et l’audit du corpus passent avant lancement.

Contrôle du départ : validation initiale complète réussie sur 999 999 cibles
(loss 11,687100). Les 100 premières mises à jour ont traité 204 800 cibles sans
erreur, avec une loss moyenne d’entraînement finie de 10,424740. Cette moyenne
porte sur des poids changeants ; ce n’est pas une mesure finale de qualité.

Artefacts locaux, ignorés par Git :

- Run : `runs/simple-baseline-2fsdgx0q/`, avec `config.json` et `progress.jsonl`.
- Lanceur : `runs/pretraining-50m-launch-5355zduc/`, avec `console.log`, commande,
  heure de départ, copie du manifeste, snapshot des sources et empreintes SHA-256.
- `recovery.pt` sera sauvegardé toutes les 1 000 mises à jour et à la fin.
- En fin de parcours : `model.pt` et `metrics.json` dans le run ; `exit.json`
  dans le dossier du lanceur pour le code de sortie.

Le dev servira au suivi ; aucun score du modèle n’est calculé sur le holdout.
Le nouveau corpus empêche une comparaison directe de perplexité avec les
anciens essais du jalon 3. Après la fin, consigner les mesures réelles, les
générations et les limites avant de décider du prochain changement.
