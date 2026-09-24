# Jalon 2 — Réimplémentation entraînée sur de vrais tokens

Mesure du 18 septembre 2026. Le pipeline du jalon 2 fonctionne de bout en bout :
lecture des tokens, entraînement, validation séparée, sauvegarde, rechargement
et génération. Les [mesures détaillées](reimplementation-corpus.json) sont versionnées ;
le checkpoint reste local dans `runs/`.

## Reproduire

Depuis la racine du dépôt, avec l’environnement existant activé :

```sh
python -m reimplementation.train_corpus --device cuda --steps 200 --seed 0
python -m pytest tests/ -q --disable-warnings
```

La commande crée un nouveau dossier de résultats à chaque exécution. Cette mesure
se trouve localement dans `runs/reimplementation-corpus-xosapju_/`.

## Protocole

- Modèle initialisé aléatoirement : 12 966 976 paramètres, 64 features,
  4 têtes, 2 blocs, MLP de dimension 256, vocabulaire de 100 278 tokens.
- Composants réimplémentés en PyTorch ; aucun poids ni composant d’OLMo-core
  utilisé dans ce modèle. Pas encore de RoPE, GQA ou mixed precision.
- FP32, PyTorch 2.10.0+cu128, NumPy 2.4.6, RTX 4060 Laptop GPU, seed 0.
- AdamW : learning rate 0,001, betas `(0.9, 0.95)`, weight decay 0,01.
- 200 mises à jour, 2 séquences de 128 positions par batch : **51 200 cibles
  d’entraînement traitées**, tirées aléatoirement dans le fichier de train.
- Données réutilisées : 19 millions de tokens de train et 1 million de validation,
  préparés par la baseline avec le tokenizer Dolma 2. Les fichiers sont lus en
  mémoire mappée, en lecture seule. Leurs SHA-256 sont inchangés avant/après.
- Avant et après : mêmes 32 fenêtres fixes réparties dans chaque fichier,
  soit **4 096 cibles par split**. La validation utilise un fichier séparé.
  Les fenêtres peuvent traverser une frontière de document séparée par EOS.
- Aucun holdout final utilisé. Le script de préparation sépare train et validation
  aux frontières des documents, mais cette expérience ne vérifie pas les doublons
  éventuels entre documents du corpus.

## Mesures

| Mesure | Avant entraînement | Après 200 mises à jour |
|---|---:|---:|
| Loss sur l’échantillon de train | 11,733945 | 7,616480 |
| Loss sur l’échantillon de validation | 11,704372 | 7,785092 |
| Perplexité de validation | 121 099,97 | 2 404,49 |

La boucle des 200 mises à jour a pris **4,81 s** sur cette machine, hors chargement,
évaluations, sauvegarde et génération. Pic de mémoire allouée mesuré par PyTorch
durant cette boucle : **588 557 312 octets**, distinct de la mémoire GPU totale.
Les durées et les derniers chiffres peuvent varier selon le matériel et l’environnement.

Le checkpoint est rechargé dans un nouveau modèle avec sa configuration :
**écart maximal de logits = 0** sur une séquence de validation.
La génération a également été relancée dans un processus Python séparé :

```sh
python -m reimplementation.generate runs/reimplementation-corpus-xosapju_/model.pt \
  --device cuda --prompt "The purpose of science is" --max-new-tokens 32
```

Résultat : le prompt est suivi de **32 répétitions de « the »**. Le mécanisme de
génération fonctionne, mais cette sortie n’est pas un texte de qualité.

**38 tests passent** : comparaisons des composants et gradients, causalité,
mémorisation du petit batch, lecture des tokens sans modification, validation
sans mise à jour, restitution exacte du checkpoint, génération et pipeline complet.
Les 20 avertissements proviennent des dépendances existantes.

## Conclusion

Le cœur pédagogique et le pipeline exécutable du jalon 2 sont validés.
Nous savons désormais reconstruire et entraîner notre petit modèle causal,
mesurer sa loss, récupérer ses poids et les utiliser pour générer.

La baisse sur ce petit échantillon de validation dépasse la simple mémorisation
du batch jouet ; elle ne prouve pas une bonne qualité linguistique ou une maîtrise
du corpus. Aucun objectif de qualité n’est atteint ici. Les scores ne sont pas
directement comparables à ceux de la baseline, dont la taille et le protocole diffèrent.

Une [baseline du modèle simple](simple-baseline.md) a depuis été mesurée avec
un modèle plus large et un passage complet sur le corpus, avant le jalon 3.
Ce run de 200 mises à jour reste un test de fonctionnement. Le jalon 3 comparera
ensuite les mécanismes modernes à cette référence ; le mini-préentraînement avec
protocole train/dev/holdout complet demeure au jalon 5.
