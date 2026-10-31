# Nettoyage du stockage — 31 octobre 2026

Nettoyage autorisé après inventaire, afin de poursuivre le laboratoire
`agents-and-tools` sur le même ordinateur.

## Suppressions

- Onze fichiers `recovery.pt`, chacun accompagné d’un `model.pt` final :
  **11,22 Gio**. Avant suppression, les étapes et tous les tenseurs du modèle
  ont été comparés et trouvés identiques. L’état de l’optimiseur et des
  générateurs aléatoires de ces reprises n’est plus disponible.
- Les deux répertoires `step0` d’OLMo, dans `runs/olmo3-60m-valtest/` et
  `runs/diagnostic-python-e253k41r/python-fixed/` : **2,15 Gio**.
  Les deux checkpoints entraînés `step100` sont conservés.

Au total, **45 fichiers, soit 13,368 Gio alloués**, ont été supprimés.
Les chemins et empreintes exacts figurent dans le [manifeste](storage-cleanup-2026-10-31.json).
Les trois autres `recovery.pt`, sans modèle final voisin, n’ont pas été supprimés.

## Conservation et vérifications

Les 47 fichiers de checkpoints conservés ont été hachés avant et après la
suppression, puis après le compactage : aucune empreinte n’a changé. Les sept
modèles de l’évaluation finale correspondent aux SHA-256 de son protocole.
Le code, les résultats, les données et les journaux expérimentaux sont conservés.

L’environnement `llms/.venv` est conservé : le laboratoire
`deliberative-reasoning` l’utilise via son fichier `_llms_runtime.pth`.
Aucun artefact de ce laboratoire n’a été supprimé.

## Espace rendu à Windows

Après `fstrim`, WSL a été arrêté et son disque dynamique compacté avec
`Optimize-VHD -Mode Quick`, sous les droits administrateur Windows.
Le fichier VHDX est passé de **91 980 038 144 à 75 889 639 424 octets**,
soit environ **16,09 Go rendus**. Le gain inclut de l’espace précédemment
libéré dans Linux ; il ne correspond pas seulement aux fichiers de ce nettoyage.

L’espace libre sur C: est passé d’environ **13,80 à 29,86 Go** pendant cette
opération. Cette valeur varie avec l’activité de Windows. WSL a redémarré ;
les trois dépôts ont passé `git fsck --full` sans erreur d’intégrité.

Les sauvegardes supprimées ne sont pas dans Git : une restauration demanderait
une autre copie ou un nouvel entraînement. Les modèles finaux restent utilisables
pour l’inférence et une nouvelle phase d’entraînement avec optimiseur neuf.
