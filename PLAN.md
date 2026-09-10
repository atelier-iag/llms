# Laboratoire 1 — Modèles de fondation / scaling

## Objectif

S’approprier le fonctionnement pratique d’un petit modèle de fondation de type **OLMo-like**, de l’entraînement initial jusqu’au post-entraînement.

## Étapes

1. **Faire tourner une baseline existante**
   - entraîner un très petit modèle ;
   - comprendre le pipeline `données → tokens → batches → modèle → loss → optimisation`.

2. **Réimplémenter le cœur du modèle**
   - embeddings ;
   - self-attention causale ;
   - MLP ;
   - résidus + normalisation ;
   - Transformer decoder ;
   - boucle d’entraînement.

3. **Ajouter les mécanismes modernes**
   - RoPE ;
   - GQA ;
   - mixed precision ;
   - optimisations d’entraînement utiles.

4. **Faire un mini-préentraînement propre**
   - corpus préparé ;
   - splits train/dev/holdout ;
   - suivi de la loss, perplexité, temps et coût.

5. **Étudier le scaling**
   - faire varier taille du modèle, quantité de données et compute ;
   - comparer les courbes obtenues.

6. **Pratiquer l’adaptation**
   - continued pretraining ;
   - SFT ;
   - éventuellement une méthode simple de post-training.

7. **Faire des ablations**
   - retirer ou modifier certains mécanismes ;
   - mesurer leur effet ;
   - tester sur le holdout ;
   - documenter les échecs et différences observées.

## Critère de maîtrise

**Baseline → réimplémentation → amélioration moderne → ablation → holdout.**

Le code produit servira ensuite de premier laboratoire à intégrer au **workbench commun**.
