# Entraînement complet GQA en BF16 contre FP32

Expérience du **24 septembre 2026**, terminée : **9 279 mises à jour**, soit
**18 999 999 cibles d’entraînement**. La validation complète, calculée en FP32,
donne une loss de **5,292143** et une perplexité de **198,768992**, contre
**199,410936** pour la référence GQA entraînée en FP32 (−0,32 %).
La qualité mesurée reste proche sur cette seed ; les générations restent répétitives.

[Mesures intégrales et générations](bf16-baseline.json),
[référence GQA FP32](gqa-baseline.md),
[configuration BF16](../experiments/bf16_config.json),
[fonctionnement](../experiments/mixed_precision.md).

## Protocole et comparaison

La configuration diffère de la référence GQA uniquement par son nom et
`precision: "bf16"`. Architecture identique : **94 124 928 paramètres**, dimension
384, 8 blocs, 8 têtes Q, 2 groupes K/V et RoPE. Le corpus, ses SHA-256, les fenêtres
d’évaluation, le tokenizer, la seed 0, l’ordre des batches et la recette AdamW
restent identiques. Contexte 256, batch 8, un passage sur le corpus.

Les poids, gradients et états AdamW sont FP32 ; les opérations éligibles du
calcul avant utilisent BF16. L’évaluation et la génération restent FP32 dans
les deux expériences. La validation complète contient **999 999 cibles**.
Aucun holdout final n’a été utilisé.

| Mesure | GQA FP32 | GQA BF16 |
|---|---:|---:|
| Loss initiale, validation complète | 11,675590 | 11,675590 |
| Loss finale, validation complète | 5,295368 | 5,292143 |
| Perplexité finale, validation complète | 199,410936 | 198,768992 |
| Loss finale, échantillon fixe de train | 4,771347 | 4,762368 |
| Loss finale, échantillon fixe de validation | 5,388184 | 5,375575 |

Les échantillons fixes comptent 32 768 cibles chacun. Les deux runs partent de
la même initialisation et traitent les mêmes cibles ; leurs trajectoires peuvent
diverger avec les arrondis numériques. La petite différence finale observée ne
démontre pas un gain de qualité général de BF16.

## Coût et stabilité

Le run BF16 a duré **36,97 minutes d’entraînement** et **39,60 minutes au total**,
avec un débit moyen de **8 566 cibles/s** dans la boucle d’entraînement. Le pic
alloué mesuré par PyTorch est de **4 905 411 072 octets**. Ces mesures concernent
la RTX 4060 Laptop GPU et notre implémentation pédagogique.

Les coûts globaux de la référence GQA FP32 sont indisponibles à cause de sa reprise
depuis un ancien checkpoint ; ils ne doivent pas être comparés à la seule durée
de sa continuation. Le [benchmark court contrôlé](mixed-precision-benchmark.md)
mesurait +10,64 % de débit médian et −6,14 % de pic alloué avec BF16, dans son
propre protocole. Le run complet confirme l’exécution au budget prévu, sans
erreur numérique détectée, avec une qualité de validation proche de la référence.

## Générations et vérifications

Les mêmes trois prompts sont décodés de façon gloutonne, jusqu’à 64 nouveaux
tokens. Les sorties restent peu utilisables : répétitions autour de « the study
of the research », de « the » entre guillemets et code Python incorrect autour
de `kivy`. La précision mixte n’a pas résolu ces problèmes.

- Le journal se termine par `complete` et couvre tout le budget de tokens prévu.
- Les données sont inchangées et leurs empreintes correspondent à GQA FP32.
- Le checkpoint final rechargé restitue des logits identiques : écart maximal 0.
- Les sources actuelles correspondent aux empreintes conservées au lancement.
- Les 119 tests avaient passé avant le lancement ; cette publication ne change
  que la documentation et archive le fichier de métriques à l’identique.

Run local : `runs/simple-baseline-6aay3x5v/`. Journal, métadonnées et copie des
sources : `runs/bf16-launch-zqr283f8/`. Commit du code entraîné :
`b0c883c8ed5cb9798b26cbe001b0c1509b0b3ebd`. Le checkpoint `model.pt` reste local.
