# Corpus Python pour l’adaptation de 5 M tokens

Préparation terminée le 25 septembre 2026. Le [protocole](../experiments/continued_pretraining_python.md)
décrit les filtres, la source épinglée, la séparation par dépôt et la déduplication.
Le [manifeste archivé](continued-python-5m-manifest.json) contient les identités
exactes des données et du tokenizer.

| Split | Tokens | Fichiers Python | Dépôts distincts |
| --- | ---: | ---: | ---: |
| Train | 5 000 000 | 2 390 | 2 164 |
| Dev | 250 000 | 122 | 107 |
| Holdout réservé | 250 000 | 113 | 99 |

La préparation a visité 5 370 entrées du premier shard
`file-000000000001.json.gz` : 2 722 entrées filtrées, 2 doublons normalisés
éliminés, aucun match avec les 5 274 empreintes de l’ancien corpus. Les quotas
déjà remplis expliquent les 21 autres fichiers non retenus. Les trois derniers
documents, un par split, sont tronqués pour atteindre exactement les quotas.

Les 2 625 fichiers conservés portent les labels de licence suivants :
979 Apache-2.0, 790 MIT, 780 BSD-3-Clause, 72 BSD-2-Clause et 4 ISC.
Les chemins et attributions restent dans l’index local `documents.jsonl`.

Un audit après préparation a recalculé les SHA-256 des trois fichiers de tokens,
vérifié les sommes des intervalles de l’index, l’unicité des empreintes,
l’absence d’intersection avec l’ancien index et la disjonction des identités
de dépôt entre splits. Les résultats sont conformes au manifeste.
Cet audit d’intégrité ne calcule aucune loss sur le holdout.

SHA-256 du manifeste :
`1b25b6e25b165154f188f659d8692e495cebd7219ffbcd431212ef289f0ed7f3`.
SHA-256 de l’index :
`9c95a94bd125134c21df4102236d9051bc427d370a878d09a945f92be6c921a4`.

La séparation par dépôt ne regroupe pas les forks ; des quasi-doublons ou
fragments communs peuvent subsister. Le corpus général initial comprend des
documents sur le développement logiciel : « nouveau » désigne ici une source
distincte et l’absence de doublons documentaires normalisés détectés, pas la
preuve d’une absence totale de code déjà rencontré.
