# Sous-ensemble train 25 M — audit

Préparation terminée le 24 septembre 2026 à partir du
[corpus 50 M](pretraining-50m-data.md), sans retokenisation. Le
[manifeste publié](pretraining-25m-data.json) est identique au manifeste local
`data/pretraining-25m-v1/manifest.json`.

| Domaine | Tokens train | Documents train retenus |
| --- | ---: | ---: |
| Science, mathématiques et technologie | 6 250 000 | 641 |
| Éducation et emploi | 6 250 000 | 958 |
| Histoire et géographie | 6 250 000 | 450 |
| Développement logiciel | 6 250 000 | 420 |
| **Total** | **25 000 000** | **2 469** |

Dev et holdout restent chacun à 1 M de tokens, respectivement 100 et 112
documents. Leurs fichiers et lignes d’index sont identiques à ceux du parent.
Un audit indépendant compare chaque plage train retenue à sa plage parent :
tous les tokens sont identiques, sauf l’EOS conservé au nouveau point de coupure
des quatre derniers documents de domaine. Les 2 681 empreintes de documents
retenus sont uniques dans l’ensemble des trois splits.

Les quotas, offsets, hashes et budgets ont été vérifiés. Le lecteur de corpus
confirme **24 999 999 cibles et 12 208 mises à jour** pour un passage avec
contexte 256 et batch 8. L’extraction conserve tous les fichiers parents.

SHA-256 du manifeste :
`137a8c87acb44bef5919d481abc6a376552f25bdbf2cffffeb977c061294f174`.
SHA-256 du script d’extraction :
`a3b43a932d9aeb5e798a7a94db355efa7d4686b72953b4ccec4ee3c0db0ac9b2`.

**162 tests réussis** avant lancement, dont dix nouveaux contrôles couvrant
l’extraction, les quotas par domaine, les copies dev/holdout, la reproductibilité,
la conservation du parent et le rejet des entrées incompatibles. Les 20
avertissements de dépréciation viennent des dépendances existantes.
Voir le [protocole de comparaison](../experiments/data_scaling.md).
