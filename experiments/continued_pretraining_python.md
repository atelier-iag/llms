# Jalon 6 : continued pretraining sur Python

Question : que gagne notre modèle sur du code Python après un petit budget
d’adaptation, et que perd-il sur son domaine initial ? On repart du
[94 M préentraîné sur 50 M de tokens](../results/pretraining-50m.md), puis on
apprend **5 M de tokens supplémentaires** par prédiction du prochain token.
Le tokenizer et l’architecture restent identiques. Ce n’est pas encore du SFT :
les exemples sont des fichiers Python, sans couples instruction/réponse.

Le principe de poursuite du préentraînement sur un domaine est étudié dans
[Gururangan et al., 2020](https://aclanthology.org/2020.acl-main.740/).
Notre essai vérifie ce mécanisme à petite échelle ; il ne reproduit pas leurs
résultats et ne garantit pas du code correct ou la disparition des répétitions.

## Corpus et séparation

Source : [CodeParrot clean](https://huggingface.co/datasets/codeparrot/codeparrot-clean),
révision `35a59fb025bc0a102f7d96eac09d145b896d487b`, shards lus dans l’ordre lexical.
La carte décrit des fichiers Python issus de GitHub avec un premier nettoyage.
Notre préparation conserve les fichiers `.py` de 100 à 100 000 caractères,
étiquetés non générés, dont l’analyse syntaxique Python 3 réussit, avec une
licence déclarée MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause ou ISC. Le code est
uniquement analysé et tokenisé, jamais exécuté.

Une fonction de hash de l’identité du dépôt, insensible à la casse et salée
avec la seed 0, affecte chaque dépôt à train/dev/holdout avec des probabilités
90/5/5. Les quotas sont de **5 M / 250 k / 250 k tokens**. Un dépôt ne peut
apparaître dans deux splits. Les doublons exacts après NFKC et réduction des
espaces sont supprimés globalement ; les 5 274 empreintes documentaires de
l’ancien corpus, tous splits confondus, sont aussi exclues. Cette exclusion
n’utilise ni les scores ni les tokens du holdout général.

Le texte original garde son indentation lors de la tokenisation, avec le même
tokenizer Dolma épinglé et EOS entre fichiers. Le dernier fichier de chaque
quota est tronqué et terminé par EOS. Les fenêtres de contexte peuvent franchir
ces séparateurs, comme dans le préentraînement précédent. `documents.jsonl`
conserve dépôt, chemin, licence, shard/ligne, empreintes et intervalle de tokens.
Manifeste, fichiers et index ont chacun leur SHA-256. Les tokens et l’index
restent locaux ; le manifeste léger est publié avec les résultats.

Les identités de forks ne sont pas regroupées : la séparation par dépôt et la
déduplication exacte ne garantissent pas l’absence de quasi-doublons ou de
fragments communs. Ce petit préfixe filtré d’un seul shard n’est pas un
échantillon représentatif de tout Python. La réussite de `ast.parse` ne prouve
ni la qualité ni la correction du code.

```sh
python -m reimplementation.prepare_python_corpus \
  --output-dir data/continued-python-5m-v1 \
  --exclude-corpus data/pretraining-50m-v1
```

La préparation refuse d’écraser un dossier existant et marque un corpus
interrompu/incomplet comme tel ; le runner exige un manifeste complet.

## Adaptation et évaluation

La [configuration](continued_pretraining_python_config.json) conserve les
94 124 928 paramètres, RoPE, GQA 8 Q / 2 KV, SDPA, BF16, contexte 256 et batch 8.
Le checkpoint initial local est `runs/simple-baseline-2fsdgx0q/model.pt`.
`--init-from` charge uniquement ses poids : **AdamW neuf**, pic `1e-4`,
warmup 50 mises à jour, puis cosinus jusqu’à `1e-5`, une passe, clipping à 1.
Cela représente **4 999 999 cibles et 2 443 mises à jour**. Le compteur de
l’adaptation repart de zéro ; le checkpoint initial provenait de 24 415 mises
à jour sur 50 M de tokens. Son chemin et son SHA-256 sont enregistrés.

```sh
python -m reimplementation.train_baseline --device cuda \
  --config experiments/continued_pretraining_python_config.json \
  --init-from runs/simple-baseline-2fsdgx0q/model.pt \
  --data-dir data/continued-python-5m-v1 \
  --general-data-dir data/pretraining-50m-v1
```

Avant et après adaptation, on évalue en FP32 les **249 999 cibles du dev Python**
et les **999 999 cibles du même dev général**. Les perplexités sont comparées
avant/après sur chaque corpus séparément. Le score général initial doit retrouver
celui du checkpoint de référence. Les 128 fenêtres fixes de train, dev Python et
dev général sont aussi évaluées toutes les 500 mises à jour, avec une sauvegarde
de reprise. Seuls les batches Python train entrent dans l’optimiseur.

Les trois prompts habituels et deux débuts de fonctions Python sont générés
avant/après, avec décodage greedy et au plus 64 nouveaux tokens. Ce sont des
sondes qualitatives, pas une mesure de réussite fonctionnelle. Le modèle final
est rechargé et doit redonner exactement les mêmes logits sur la sonde de contrôle.
**Aucun des deux holdouts n’est évalué.** Le checkpoint final est le point fixé
par le budget ; les dev servent au diagnostic et ne déclenchent pas d’arrêt anticipé.

Pour reprendre cette même adaptation après interruption, `--resume` restaure
poids, optimiseur, calendrier, RNG et historique. Il est exclusif de `--init-from` :

```sh
python -m reimplementation.train_baseline --device cuda \
  --resume runs/simple-baseline-REPLACE/recovery.pt \
  --data-dir data/continued-python-5m-v1 \
  --general-data-dir data/pretraining-50m-v1
```

Les hashes des données et des deux manifestes sont contrôlés à la reprise et
en fin d’exécution. Cet essai n’a qu’une seed, un domaine, un budget et une recette :
il mesure une adaptation et son coût hors domaine, pas un optimum. Le SFT reste
l’étape suivante du jalon 6.
