# Corpus du préentraînement 50 M — version 1

Préparation terminée le 24 septembre 2026. Le [manifeste publié](pretraining-50m-data.json)
est la copie exacte du manifeste local `data/pretraining-50m-v1/manifest.json`.
Voir le [protocole reproductible](../experiments/pretraining_50m.md).

| Split | Tokens, EOS compris | Documents retenus |
| --- | ---: | ---: |
| Train | 50 000 000 | 5 062 |
| Dev (`val`) | 1 000 000 | 100 |
| Holdout réservé | 1 000 000 | 112 |

Chaque source fournit 12,5 M / 250 k / 250 k tokens train/dev/holdout.
7 074 documents ont été parcourus ; 25 ont été filtrés et 1 doublon normalisé
a été écarté. Les autres documents non retenus appartenaient à des quotas déjà
remplis. Douze documents ont été tronqués pour terminer les douze quotas.

Un audit indépendant de l’index vérifie **5 274 empreintes uniques**, l’affectation
par hash au bon split, la continuité des offsets, les tailles exactes, les bornes
du vocabulaire et chaque EOS de fin de document. Les SHA-256 des trois fichiers
correspondent au manifeste. Aucun score de modèle n’a été calculé sur le holdout.

Cette séparation concerne les documents identiques après normalisation ; elle
ne garantit pas l’absence de passages communs ou de quasi-doublons. Les documents
OCR sont longs : le dev compte notamment seulement 6 documents du domaine
logiciel, malgré ses 250 k tokens. Les scores par domaine auront donc une
représentativité limitée et doivent être interprétés avec prudence.

SHA-256 du manifeste :
`74bd9a402fef117d425843859755da36db91a6b3445ded6c8aea4081e0d7d65f`.
SHA-256 du script de préparation exécuté :
`0c7b8ae7cb2698c757b0381cdb1e7a1d5120105a392a8c82b6bb9f2e85a10e1c`.

Validation du code avant lancement : **152 tests réussis**, comprenant la
comparaison des sorties/gradients SDPA, la causalité, les checkpoints, la reprise
BF16/SDPA, les quotas, la déduplication, la reproductibilité et le rejet d’un
manifeste ou de données incompatibles. Les 20 avertissements de dépréciation
proviennent des dépendances existantes.
