# Jalon 2 — Premier apprentissage de la réimplémentation

Date : 2026-09-18.

La boucle d’entraînement de [train.py](../reimplementation/train.py) effectue
une remise à zéro des gradients, un calcul de loss, une backpropagation et une
mise à jour des poids à chaque itération.

## Protocole

- Modèle : 388 paramètres, vocabulaire de 6 tokens, dimension 4, 2 blocs,
  2 têtes d’attention et dimension intermédiaire du feed-forward de 8.
- Exécution : CPU, un thread, FP32, PyTorch 2.10.0+cu128, seed 0.
- Optimiseur : AdamW, learning rate 0,01, betas `(0.9, 0.95)`, weight decay 0.
- Données : un seul batch fixe, répété pendant 200 mises à jour.
- Extrait `[2, 5, 2, 4]` : entrées `[2, 5, 2]`, cibles `[5, 2, 4]`.

Depuis la racine du dépôt, avec l’environnement existant activé :

```sh
python -m reimplementation.train --steps 200 --lr 0.01 --seed 0
```

## Résultats

| Mesure | Valeur |
|---|---:|
| Loss avant entraînement | 1,740844 |
| Loss après 200 mises à jour | 0,000521 |
| Cibles correctement prédites après entraînement | 3 / 3 |

Les prédictions finales sont `[5, 2, 4]`. Les probabilités attribuées aux trois
cibles sont respectivement 0,999490, 0,999522 et 0,999425.

Les métriques détaillées de cette exécution sont conservées localement dans
`runs/reimplementation-tiny-batch-nwck2fml/metrics.json`. Chaque nouvelle exécution
crée son propre dossier ; les métriques brutes restent ignorées par Git.

Les 27 tests du dépôt passent, dont les vérifications de remise à zéro des
gradients et d’apprentissage du batch fixe.

## Portée

Cet essai confirme que notre boucle peut entraîner le modèle sur les trois
prédictions du batch. Il ne mesure ni la généralisation ni la qualité sur du texte
nouveau. Les prédictions sont calculées avec les préfixes corrects en entrée ;
aucune génération autonome n’est évaluée ici.

Pour poursuivre le jalon 2 : entraînement avec validation séparée, sauvegarde et
rechargement des poids, puis génération. Les mécanismes modernes prévus au jalon 3
restent à ajouter.
