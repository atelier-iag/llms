# Préentraînement 50 M — résultats

**Terminé sans erreur le 24 septembre 2026 à 15 h 09 (Paris).** Le modèle a traité
les **49 999 999 tokens cibles prévus en 24 415 mises à jour**. La validation
complète atteint une loss de **5,216053** et une perplexité de **184,205643**.
Les générations restent fortement répétitives et le code produit est incorrect.

[Mesures intégrales et générations](pretraining-50m.json),
[protocole](../experiments/pretraining_50m.md),
[configuration](../experiments/pretraining_50m_config.json),
[audit des données](pretraining-50m-data.md).

## Protocole et qualité

Le modèle de **94 124 928 paramètres** part de poids aléatoires, avec RoPE, GQA
8Q/2KV, BF16 et SDPA. Contexte 256, batch 8, seed 0, un passage sur les 50 M
de tokens train. Poids et AdamW FP32, évaluation et génération FP32.
Le dev contient 1 M de tokens ; le holdout de 1 M reste réservé, sans score de modèle.

| Mesure | Avant entraînement | Après entraînement |
| --- | ---: | ---: |
| Loss, dev complet (999 999 cibles) | 11,687100 | 5,216053 |
| Perplexité, dev complet | 119 026,37 | 184,21 |
| Loss, échantillon fixe train (32 768 cibles) | 11,694481 | 4,750711 |
| Loss, échantillon fixe dev (32 768 cibles) | 11,687591 | 5,252076 |

La dernière perplexité sur l’échantillon dev est **190,96** ; elle diffère de
**184,21** sur le dev complet car les tokens évalués diffèrent. Ce nouveau corpus
et son budget ne permettent pas une comparaison directe avec les anciennes
perplexités 198,77/199,41 du jalon 3, ni d’attribuer un gain de qualité à SDPA.
Ce run établit une référence pour les prochaines expériences à données fixes.

## Coût mesuré

Sur la RTX 4060 Laptop GPU, la boucle d’entraînement a pris **57,60 minutes**,
soit **14 467 cibles/s**. Le chronomètre interne du runner indique **60,67 minutes**
avec évaluations, sauvegardes et générations. Le pic mémoire alloué mesuré par
PyTorch est de **4 566 484 480 octets** (4,25 Gio).

Les horodatages civils du lanceur donnent **66,41 minutes** entre le départ
(14 h 03 min 20 s) et la sortie (15 h 09 min 44 s). Cette durée diffère du
chronomètre interne ; les journaux disponibles ne permettent pas d’en attribuer
précisément l’écart. Les deux mesures sont conservées distinctement.
Le débit soutenu est inférieur aux 15 972 cibles/s du benchmark court ;
l’estimation de 52 minutes de boucle était donc optimiste pour ce run complet.

## Générations et vérifications

Les trois prompts sont décodés en greedy, jusqu’à 64 nouveaux tokens. Le texte
sur la science répète « comprehensive and comprehensive » puis « research » ;
celui sur l’histoire répète la même phrase. Le prompt Python produit
`class kivy.uix.button import`, puis une suite de `@`. La baisse de loss ne
suffit donc pas à rendre les générations utilisables. Les sorties intégrales
sont conservées dans le JSON, sans correction ni sélection supplémentaire.

- Le journal se termine par `complete` et le processus sort avec le code 0.
- Le budget est couvert exactement, y compris la dernière fenêtre partielle.
- Le checkpoint rechargé produit des logits identiques : écart maximal 0.
- Les hashes train/dev et du manifeste correspondent au lancement.
- Les sources Python et la configuration exécutée correspondent au snapshot.
- Les **152 tests** avaient passé avant lancement ; cette publication archive
  les métriques à l’identique et met à jour la documentation, sans changer le code.

Run local : `runs/simple-baseline-2fsdgx0q/` (`model.pt`, `recovery.pt`,
`metrics.json`, `config.json`, `progress.jsonl`). Lanceur et sources archivées :
`runs/pretraining-50m-launch-5355zduc/`. Les checkpoints restent locaux et ignorés
par Git. Commit du code entraîné : `eb5e971c5047d20c03e51121901c300c0a0ea7ff`.
