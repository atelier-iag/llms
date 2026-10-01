# Diagnostic de transfert SFT : consignes, ordre et vocabulaire

Protocole fixé le **1er octobre 2026**, avant de générer les nouvelles
réponses. Le meilleur modèle du petit diagnostic obtient
[61/64 à taux réduit](../results/sft-learning-rate-1e-4.md), avec des
gabarits et un vocabulaire partagés entre train et dev. Nous mesurons ici
sa sensibilité à trois changements séparés.

## Plan fixé avant la mesure

Deux checkpoints gelés : le généraliste 94 M préentraîné sur 50 M tokens
et le SFT à pic `1e-4`, après 400 mises à jour. **Aucun nouvel entraînement,
aucune sélection de checkpoint, aucun holdout réservé consulté.**
Les identités SHA-256 figurent dans la [configuration](sft_transfer_config.json).

Chaque condition contient les mêmes 64 cas appariés du dev connu :
16 copies de mot, 24 couleurs, 24 prénoms. Les nouvelles conditions
conservent les associations et les questions de référence.

| Condition | Modification unique | Inchangé |
| --- | --- | --- |
| Contrôle | Aucune | Les 64 exemples dev précédents |
| Reformulation | Instruction | Contexte, réponse, ordre des phrases, valeurs |
| Ordre inversé | Ordre des deux phrases/lignes du contexte | Instruction, faits, réponse |
| Nouvelles valeurs | Substitution bijective des mots, couleurs et prénoms | Gabarits, instruction, ordre et relations |

Total : **256 réponses par modèle, 512 générations**. Les nouveaux prompts
sont absents du petit train SFT. Les prompts du contrôle sont volontairement
ceux du dev déjà utilisé. Chaque question demeure liée à son ID dev original.

Les reformulations sont fixées, avec la contrainte de réponse inchangée :

- `Copy the target word.` → `Return the target word.` ;
- `Which color is the lantern/basket?` → `What is the color of the lantern/basket?` ;
- `Who has the key/map?` → `Which person has the key/map?`.

Cette dernière formulation conserve explicitement la relation « has ».
Une seule reformulation par gabarit est testée ; elle ne représente pas
toutes les manières de poser la question.

Les nouvelles valeurs sont fixées dans le
[générateur](sft_transfer_data.py) et le [manifeste](sft_transfer_manifest.json) :
mots `table, window, garden, forest, water, train, flower, planet`,
couleurs `purple, orange, pink, brown, gray, cyan`,
prénoms `Grace, Henry, Irene, Jack, Karen, Louis`.
Elles sont absentes des réponses du petit jeu SFT, mais peuvent avoir
été vues pendant le préentraînement. Leur fréquence et leur segmentation
en tokens peuvent différer des anciennes valeurs.

## Mesures

Le même [évaluateur de génération](../evaluation/sft.py) est réutilisé :
génération greedy, FP32, contexte de 256 tokens, budget de huit nouveaux
tokens, arrêt à EOS. Les réponses de référence ne sont jamais fournies
à la génération. Le score exact normalise casse, espaces et ponctuation
terminale, comme lors du diagnostic précédent.

Rapporter pour chaque modèle et chaque condition :

- réponses exactes globales et par tâche ;
- arrêt à EOS ;
- cas corrects conservés, perdus ou gagnés par rapport au contrôle apparié ;
- contextes où les deux questions opposées sont réussies, pour couleurs et prénoms.

Les différences sont descriptives. Les conditions sont corrélées par
construction ; on ne les agrégera pas en un prétendu échantillon indépendant
de 256 cas. Le diagnostic est destiné au développement et devient connu
dès sa première évaluation.

Les générations du contrôle doivent reproduire exactement les sorties
archivées du modèle initial et du SFT, notamment leurs scores 0/64 et 61/64.
Un désaccord interrompt le run pour examen du protocole.

Un oracle textuel vérifie les labels indépendamment des réponses enregistrées.
La préparation contrôle les 256 prompts distincts, leur absence du train,
les paires de questions, les facteurs modifiés et le budget réel du tokenizer.
Les exemples, le manifeste, le protocole et le code sont commités avant
l’évaluation ; les empreintes sont vérifiées à nouveau en fin de run.

## Reproduction

La préparation écrit de nouveaux fichiers et refuse de les écraser :

```sh
python -m experiments.sft_transfer_data
```

Les fichiers [d’exemples](sft_transfer_examples.json) et de manifeste étant
versionnés, on peut directement les évaluer avec les deux checkpoints locaux :

```sh
python -m evaluation.sft_transfer --device cuda \
  --config experiments/sft_transfer_config.json
```

Le runner exige un dépôt propre et crée un nouveau dossier sous `runs/`.
Il conserve les sorties après chaque condition, puis les métriques,
l’analyse appariée et les identités du code, des données et des checkpoints.
Les fichiers publiés ne contiennent que nos exemples et leurs générations.
