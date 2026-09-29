# Ablation SFT : réduire le taux d’apprentissage

Le [test de diversité](../results/sft-diversity-64-64.md) obtient 44/64
réponses dev exactes et une perplexité générale de 1 476,86, contre
184,21 avant adaptation. On teste maintenant si un taux d’apprentissage
plus faible améliore le compromis entre apprentissage et conservation
du comportement général.

**Résultat du 29 septembre 2026 :** [61/64 sur dev et perplexité générale
272,98](../results/sft-learning-rate-1e-4.md), contre 44/64 et 1 476,86.
Le protocole ci-dessous a été fixé avant le run ; ce lien de bilan a été
ajouté après l’audit final.

## Un seul paramètre d’entraînement modifié

La [configuration](sft_learning_rate_config.json) reprend celle du test de
diversité, avec **`optimizer.lr = 1e-4` au lieu de `3e-4`**. Seuls le
nom et la description de l’expérience changent également.

Le warmup de 20 mises à jour, la forme du cosinus et son plancher relatif
de 0,1 restent identiques. L’ensemble du calendrier est donc divisé par
trois : le taux final passe de `3e-5` à `1e-5`. Il ne s’agit pas de
changer séparément le warmup, la durée ou la forme de la décroissance.

Les deux modèles repartent du **même checkpoint généraliste 94 M / 50 M
tokens**, avec AdamW neuf. Mêmes données train/dev, mêmes probes train,
même seed et donc mêmes permutations et batches. Même architecture,
tokenizer, contexte, précision BF16/FP32, clipping, betas et weight decay.

Budget inchangé : **64 exemples train × 50 passes**, batch 8,
**400 mises à jour**, **3 200 présentations**, **7 000 cibles réponse/EOS**
et **101 600 tokens non paddés**. Les données ne sont pas régénérées :
on réutilise directement `data/sft-diversity-64-64-v1/`.

## Mesures et règles de comparaison

Le point final est fixé à la mise à jour 400, sans arrêt anticipé, sans
prolongation pour compenser le taux réduit et sans sélection de checkpoint
selon les scores. La référence à `3e-4` est le run déjà archivé.

- Génération greedy sur les **mêmes 64 cas dev**, budget de 8 nouveaux
  tokens, score exact normalisé global et par tâche.
- Génération sur les **mêmes 64 exemples train**, comptée séparément
  comme mémorisation.
- Questions sur la carte et paires de questions dans un même contexte.
- Loss/perplexité générale sur les **mêmes 999 999 cibles** avant/après ;
  même échantillon fixe de 32 768 cibles aux mises à jour 0/100/200/300/400.
- Identité exacte des données, prompts, références et budgets, et rapport
  de 3 entre les taux d’apprentissage journalisés.

Le dev est connu et guide le développement ; **aucun holdout réservé
n’est évalué**. Une baisse de perplexité générale ne suffirait pas à
annoncer une amélioration globale si la réussite aux consignes reculait.
Le résultat sera présenté comme un compromis, avec ses deux dimensions.
Une seule seed ne permet pas d’établir la robustesse de l’effet.

Le rôle du taux d’apprentissage dans l’oubli est étudié notamment dans
[Fine-Tuning Without Forgetting via Loss-Adaptive Learning Rates](https://arxiv.org/abs/2605.20005).
Notre essai est une simple ablation de l’amplitude d’un calendrier fixe :
il n’implémente pas la méthode adaptative de cet article et n’en suppose
pas les gains pour notre modèle.

## Exécution

```sh
python -m reimplementation.train_sft --device cuda \
  --config experiments/sft_learning_rate_config.json \
  --init-from runs/simple-baseline-2fsdgx0q/model.pt \
  --data-dir data/sft-diversity-64-64-v1 \
  --probes data/sft-diversity-64-64-v1/train-probes.json \
  --general-data-dir data/pretraining-50m-v1
```

Le runner, le masque SFT, les évaluateurs et les sauvegardes sont inchangés.
Le checkpoint généraliste est conservé. Le score `probes` mesure les
64 exemples train ; `dev_generation` mesure les 64 cas de généralisation.
