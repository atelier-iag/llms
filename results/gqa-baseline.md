# GQA + RoPE contre la référence RoPE

Expérience terminée le **23 septembre 2026**. Notre modèle GQA utilise **8 têtes Q
et 2 groupes K/V**, avec RoPE. Sa perplexité de validation est de **199,410936**,
contre **199,103016** avec RoPE seul : +0,15 % sur cette seed, avec **1,85 % de
paramètres en moins**. Les générations restent répétitives.

[Mesures complètes et générations](gqa-baseline.json),
[référence RoPE](rope-baseline.md), [configuration](../experiments/gqa_config.json),
[fonctionnement et protocole](../experiments/gqa.md).

## Comparaison contrôlée

La configuration change uniquement de nom et ajoute `model.num_kv_heads = 2`.
Le tokenizer, les fichiers de données (SHA-256 vérifiés), les fenêtres de
validation, la seed 0, le mélange des fenêtres, AdamW et son calendrier de taux
d’apprentissage sont identiques. Les modèles sont entraînés depuis zéro.
La même seed ne donne pas les mêmes poids initiaux dans ces architectures
différentes ; ce résultat porte sur une seule initialisation par variante.

Dimension 384, 8 blocs, contexte 256, batch 8, calcul FP32 : un passage complet
sur le corpus, soit **18 999 999 cibles et 9 279 mises à jour**. La validation
complète compte **999 999 cibles**. Aucun résultat de holdout final n’est revendiqué.

## Mesures

| Mesure | RoPE seul | RoPE + GQA |
|---|---:|---:|
| Paramètres | 95 894 400 | 94 124 928 |
| Loss initiale, validation complète | 11,676672 | 11,675590 |
| Loss finale, validation complète | 5,293822 | 5,295368 |
| Perplexité finale, validation complète | 199,103016 | 199,410936 |
| Loss finale, échantillon fixe de train | 4,772334 | 4,771347 |
| Loss finale, échantillon fixe de validation | 5,379222 | 5,388184 |

Les échantillons fixes contiennent 32 768 cibles chacun et ne remplacent pas
la validation complète. Les métriques légères sont copiées à l’identique depuis
le run final dans `results/gqa-baseline.json`.

## Interruption et reprise

Après un premier essai interrompu par un crash, le run
`runs/simple-baseline-vp_qv7vi/` a été arrêté à la demande de l’utilisateur,
avec un checkpoint à **7 000 mises à jour / 14 336 000 cibles**.
Il a repris dans `runs/simple-baseline-hm0xav7w/` en restaurant les poids, AdamW,
les états aléatoires, le calendrier et la position dans l’ordre des données.
Les **2 279 mises à jour restantes** ont traité **4 663 999 cibles**.

Les coûts de l’ensemble du run ne sont pas récupérables depuis l’ancien
checkpoint : `training_seconds`, `wall_seconds`, le débit et le pic mémoire
globaux restent donc `null`. Le champ `session` décrit seulement la continuation :
582,42 secondes d’entraînement, 672,04 secondes au total et un pic alloué par
PyTorch de 5 224 083 456 octets. Ces valeurs ne permettent pas de conclure à un
gain de temps ou de mémoire global par rapport au run RoPE continu.

## Interprétation et vérifications

Le partage K/V préserve ici une perplexité proche de la référence avec moins
de paramètres. Ce petit écart ne démontre ni une équivalence générale ni un
gain de qualité. Les trois générations gloutonnes restent peu utilisables :
répétitions de « the study of », de « and the Congo », et code Python incorrect.

Le générateur actuel n’utilise pas de KV cache. Passer de 8 à 2 groupes K/V
diviserait par quatre la quantité de vecteurs à stocker dans un tel cache,
à contexte et type numérique identiques ; ce n’est pas une mesure de mémoire
ou de vitesse d’inférence réalisée dans cette expérience.

- Le journal se termine par `complete`, avec le budget intégral prévu.
- Les empreintes des données correspondent à la référence RoPE et sont inchangées.
- Le checkpoint final rechargé restitue des logits identiques : écart maximal 0.
- Les 105 tests du laboratoire passent, dont la comparaison exacte sur CPU
  entre entraînement continu et entraînement interrompu puis repris.

Le journal, les métadonnées et la copie des sources de la continuation sont
dans `runs/gqa-resume-_nsars2l/`. Le checkpoint final est
`runs/simple-baseline-hm0xav7w/model.pt`. Ces artefacts volumineux restent locaux.
