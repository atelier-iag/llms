# Jalon 4 : préentraînement de 50 M de tokens

Budget choisi : **50 000 000 tokens train**, **1 000 000 tokens dev** et
**1 000 000 tokens holdout**. Le fichier `val.npy` désigne le dev pour rester
compatible avec notre lecteur. Le holdout est réservé à l’évaluation finale
du protocole figé ; le runner d’entraînement ne l’ouvre pas.

La préparation est terminée : voir l’[audit du corpus](../results/pretraining-50m-data.md)
et le [manifeste exact](../results/pretraining-50m-data.json).
Le [premier run est terminé](../results/pretraining-50m.md) le 24 septembre 2026 :
perplexité dev complète **184,21**, générations encore répétitives et holdout réservé.

## Préparation des données

```sh
python -m reimplementation.prepare_corpus --output-dir data/pretraining-50m-v1
```

La commande exige un nouveau dossier et conserve l’ancien corpus. Elle lit en
streaming `allenai/dolma3_pool`, révision
`6462556697df1a8f5c953727e9c686629ad98b68`, et le tokenizer Dolma2 épinglé dans
[tokenizer.py](../reimplementation/tokenizer.py). Les quatre sources OCR sont
science/math/technologie, éducation/emploi, histoire/géographie et développement
logiciel. Chaque source fournit 25 % des tokens de chaque split.

Les shards sont parcourus dans l’ordre lexical et les documents dans l’ordre
des lignes jusqu’à remplissage des quotas. Il s’agit de préfixes déterministes
par domaine, pas d’un échantillon aléatoire représentatif de tout Dolma.

- Filtrage minimal : 100 à 2 000 000 caractères et au moins 20 % de caractères
  alphabétiques parmi les caractères non blancs.
- Empreinte SHA-256 du texte normalisé NFKC et des espaces regroupés : suppression
  des doublons exacts normalisés, commune à toutes les sources.
- Affectation au split par hash de cette empreinte avec seed 0 : buckets 96/2/2,
  indépendante de la source. Un même document normalisé ne peut changer de split.
- Tokenisation du texte original, ajout d’EOS. Seul le dernier document retenu
  par source/split peut être tronqué pour atteindre exactement le quota ; il
  conserve un EOS final.

Ces contrôles ne retirent pas les quasi-doublons, les passages communs à des
documents distincts ni toutes les erreurs OCR. Ce corpus constitue une première
base traçable pour le laboratoire, pas un nettoyage exhaustif de production.

Les trois fichiers sont des uint32 little-endian bruts, sans en-tête NumPy
malgré le suffixe `.npy`. `documents.jsonl` conserve source, shard/ligne, empreinte,
split, offset et longueur de chaque document retenu. `manifest.json` enregistre
versions, politiques, compteurs, tailles et SHA-256. Il ne passe à `complete`
qu’après fermeture et vérification des trois fichiers. Une erreur conserve le
dossier partiel marqué `incomplete` ; choisir un nouveau dossier pour réessayer.
Les gros fichiers restent locaux, ignorés par Git.

## Entraînement et mesures

```sh
python -m reimplementation.train_baseline --device cuda \
  --config experiments/pretraining_50m_config.json \
  --data-dir data/pretraining-50m-v1
```

La [configuration](pretraining_50m_config.json) exige un manifeste complet,
les bonnes tailles et le tokenizer attendu. Le runner contrôle les hashes
train/dev avant de créer le run, les revérifie à la fin et conserve l’empreinte
du manifeste dans les résultats et les checkpoints de reprise.

Le modèle repart **de poids aléatoires**, avec RoPE, GQA 8Q/2KV, BF16 et SDPA.
Reprendre les anciens poids exposerait potentiellement le nouveau holdout à des
données déjà vues. Même taille, contexte 256 et batch 8 ; un passage complet
donne **49 999 999 tokens cibles et 24 415 mises à jour**. Le premier token n’a
pas de contexte et n’est pas une cible. AdamW conserve le pic `3e-4` et le
plancher `3e-5` ; le warmup passe à 250 mises à jour, environ 1 % du parcours.

Les fenêtres train sont mélangées avec seed 0, sans répétition des cibles.
Elles peuvent traverser un EOS ; le masque causal ne se réinitialise pas à cette
frontière. Dev complet avant/après : 999 999 cibles. Toutes les 1 000 mises à jour,
évaluation des mêmes 128 fenêtres train/dev et sauvegarde de reprise atomique.
Logs de loss toutes les 100 mises à jour, puis perplexité, temps, débit, mémoire,
rechargement du checkpoint et trois générations greedy en fin de run.

Les données et le budget changent par rapport aux essais du jalon 3 : les
perplexités ne sont **pas directement comparables** aux anciens 198,77/199,41.
Ce run établit une nouvelle référence sur laquelle mener des comparaisons à
données fixes. Le coût est mesuré en durée et mémoire GPU ; aucun coût monétaire
n’est inventé pour le GPU local. L’estimation issue du benchmark court est
d’environ 52 minutes de calcul, hors évaluations et sauvegardes.

Reprise après interruption, toujours avec le même corpus :

```sh
python -m reimplementation.train_baseline --device cuda \
  --resume runs/simple-baseline-REPLACE/recovery.pt \
  --data-dir data/pretraining-50m-v1
```
