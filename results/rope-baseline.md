# RoPE contre la baseline sans position explicite

Expérience du 19 septembre 2026, **terminée**. RoPE améliore la prédiction sur
notre validation : loss **5,419455 → 5,293822**, perplexité **225,756 → 199,103**
soit une baisse de **11,81 %**. Les trois générations gloutonnes restent fortement
répétitives. Ce résultat porte sur une seule seed et ce budget précis.

[Mesures complètes et générations](rope-baseline.json),
[baseline NoPE](simple-baseline.md), [configuration RoPE](../experiments/rope_config.json),
[démo et fonctionnement](../experiments/README.md).

## Comparaison contrôlée

La seule différence de configuration, hors nom de l'expérience, est
`model.rope_theta = 10000.0`. Les deux modèles repartent de poids aléatoires avec
la même seed 0. Ils ont la même architecture de base et **95 894 400 paramètres** :
dimension 384, 8 blocs, 8 têtes de 48 dimensions, MLP 1 536, vocabulaire 100 278.
RoPE fait tourner les 24 paires adjacentes de Q et K dans chaque tête ; V ne tourne pas.

Le tokenizer, les SHA-256 des données, l'ordre de mélange, les fenêtres de
validation, AdamW, le calendrier de learning rate, le clipping et le calcul FP32
sont identiques. Contexte 256, batch 8, un passage sur 19 millions de tokens :
**18 999 999 cibles et 9 279 mises à jour**, dernière fenêtre partielle incluse.
La validation complète compte **999 999 cibles** et ne met pas à jour les poids.
Le holdout final n'a pas été utilisé.

## Mesures

| Mesure | Sans RoPE | Avec RoPE |
|---|---:|---:|
| Loss initiale, validation complète | 11,677568 | 11,676672 |
| Loss finale, validation complète | 5,419455 | **5,293822** |
| Perplexité finale, validation complète | 225,756 | **199,103** |
| Loss finale, échantillon fixe de train | 4,911379 | 4,772334 |
| Loss finale, échantillon fixe de validation | 5,501702 | 5,379222 |
| Entraînement, hors évaluations et sauvegardes | 36,09 min | 51,81 min |
| Temps total enregistré | 38,44 min | 55,44 min |
| Débit de la boucle | 8 774 cibles/s | 6 112 cibles/s |
| Pic alloué par PyTorch | 5 277 036 544 octets | 5 283 328 000 octets |

Les échantillons fixes contiennent 32 768 cibles chacun ; leurs losses ne doivent
pas être confondues avec la mesure sur toute la validation. Les historiques
complets sont conservés dans les deux JSON.

Ces durées ont été observées sur la même RTX 4060 Laptop, avec l'implémentation
pédagogique actuelle. Elles incluent son surcoût de rotations et peuvent aussi
varier avec l'état du GPU ; elles ne mesurent pas la vitesse d'un noyau RoPE optimisé.
La mémoire allouée n'est pas la mémoire totale réservée ou occupée sur le GPU.

## Générations et interprétation

Même protocole que la baseline : trois prompts fixés, décodage glouton,
64 nouveaux tokens au maximum. Avec RoPE :

- `The purpose of science is` continue par « the case of the research of the study
  of the study of the study », puis boucle autour de « the study » et « first-time ».
- `The history of the world` finit par répéter « and the Congo ».
- `To write a Python function,` finit par répéter « and `k` is a `k` ».

La meilleure perplexité signifie une meilleure probabilité attribuée aux vrais
tokens de validation sur ce run. Elle ne suffit pas à obtenir un texte utile.
RoPE n'a pas résolu les répétitions observées. Une seule seed et trois prompts
ne permettent pas de généraliser ce gain ou d'isoler toutes les causes des échecs.

## Vérifications et traçabilité

- Les compteurs couvrent exactement le budget prévu ; les données sont inchangées.
- Le checkpoint final rechargé restitue des logits identiques : écart maximal 0.
- La première génération a été reproduite à l'identique dans un processus séparé.
- Les sources Python et configurations actuelles ont été comparées à la copie
  conservée au lancement. Les empreintes des sources et du checkpoint figurent
  dans le JSON. Le code était alors non commité au-dessus de `403f7ce`.
- Les 70 tests vérifient notamment les rotations, gradients, positions relatives,
  causalité, sauvegarde/rechargement et conservation du comportement sans RoPE.

Run local : `runs/simple-baseline-r5x189so/` ; journal, métadonnées de lancement et
copie des sources : `runs/rope-launch-ga4x7dhg/`. Les checkpoints et corpus restent
locaux, ignorés par Git. Les métriques légères sont archivées dans `results/`.

```sh
python -m reimplementation.train_baseline --device cuda --config experiments/rope_config.json
python -m reimplementation.generate runs/simple-baseline-r5x189so/model.pt \
  --device cuda --prompt "The purpose of science is" --max-new-tokens 64
```

Cette variante RoPE sert désormais de référence pour étudier le partage K/V de GQA.
