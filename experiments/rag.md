# Jalon 4 — Les bases du RAG

**Statut : première démo de recherche et de construction du prompt disponible.**
La génération par un LLM, la recherche par embeddings et l’évaluation du RAG
complet restent à réaliser. Ce jalon peut être suivi en parallèle du jalon 3 ;
il ne dépend pas de la fin du préentraînement de notre modèle.

## Commencer avec les notes du laboratoire

Depuis la racine du dépôt, avec Python (aucune dépendance supplémentaire) :

```sh
python -m experiments.rag_demo
python -m experiments.rag_demo --question "Quelle perplexité obtient GQA ?" --k 1
```

La [démo](rag_demo.py) lit trois comptes rendus réels dans `results/` : la baseline
simple, RoPE et GQA. Elle les découpe en passages de 80 mots avec 16 mots de
chevauchement, puis affiche le classement et le prompt contenant les sources.
Ces mots sont séparés par les espaces : ce ne sont pas les tokens du LLM.

Le code central est dans [retrieval.py](../reimplementation/retrieval.py) :
`chunk_document → TfidfIndex → search → build_prompt`. L’index est ici lexical :
chaque dimension correspond à un terme, pondéré par sa fréquence et son IDF,
puis les vecteurs sont normalisés pour une comparaison cosinus. Aucun encodeur
neuronal n’est encore utilisé. Le vocabulaire et les IDF sont calculés sur les
passages, avant de transformer la question dans le même espace.

**Observer d’abord la sélection de l’information.** Pour la question par défaut,
repérer le passage qui indique les 8 têtes Q et les 2 groupes K/V. Le LLM recevrait
ce passage dans son prompt ; ses poids n’auraient pas besoin d’être modifiés pour
exploiter ces nombres. Cette démo s’arrête au prompt et n’invente pas une sortie
de modèle : l’appel au générateur sera l’étape suivante.

Essayer ensuite une question contenant `GQA` mais demandant une information absente,
par exemple son score sur un benchmark non mesuré. Des passages peuvent être
retrouvés sans contenir la réponse : **pertinence lexicale et réponse justifiée
sont deux vérifications différentes**. De même, cette recherche simple peut
manquer un passage pertinent si la question utilise des synonymes.

Ces exemples servent à comprendre le pipeline ; ils ne constituent pas le
benchmark de vingt questions ni son holdout. À cette étape, le contexte contient
au plus `k × 80` mots de passages ; un budget en tokens du générateur sera ajouté
avec l’appel au LLM.

## Ce qu’il faut comprendre

Un LLM répond à partir de ses poids et du contexte qu’on lui donne. Le RAG ajoute
une recherche dans un corpus externe pour sélectionner les informations à placer
dans ce contexte. Pour cet exercice, on met à jour les documents et leur index,
sans réentraîner le générateur.

```text
Préparation : documents → passages identifiés → index
Question → recherche des k passages les plus proches → contexte + consigne
         → LLM → réponse avec références aux passages
```

- **Découpage (chunking)** : transformer un document en passages assez courts
  pour tenir dans le contexte, en conservant assez d’information pour répondre.
- **Embedding de passage** : un vecteur représentant un texte pour la recherche.
  Le modèle qui le calcule peut être distinct du LLM qui génère la réponse ;
  ce n’est pas simplement récupérer un vecteur dans notre table d’embeddings de tokens.
- **Index et recherche** : stocker les passages et leurs représentations, puis
  classer les candidats pour une question. Une matrice de vecteurs en mémoire
  suffit pour ce petit corpus.
- **Contexte et sources** : transmettre les passages retenus avec leurs
  identifiants et demander au modèle d’appuyer sa réponse sur ces passages.
  Une citation produite par le modèle doit être vérifiée : elle ne garantit pas
  à elle seule que la source soutient l’affirmation.

Le fine-tuning modifie les poids du modèle. Le RAG minimal étudié ici sélectionne
des documents au moment de répondre. Fournir manuellement un document dans le
prompt permet déjà de répondre sur ce document ; le RAG ajoute la sélection
automatique des passages dans un corpus.

## Petit exercice de bout en bout

### 1. Préparer les documents et les questions

Choisir environ dix documents courts, par exemple des notes techniques de
l’atelier. Conserver un identifiant de document et un identifiant par passage.
Commencer avec une taille de passage fixe et un petit chevauchement, à ajuster
uniquement sur les questions de développement.

Écrire environ vingt questions avec leur réponse attendue et les passages qui
la justifient. Inclure des questions dont la réponse est absente des documents.
Séparer dès le départ douze questions de développement et huit de holdout, avec
des questions sans réponse dans chaque groupe. Ne pas indexer les questions,
les réponses attendues ni leurs annotations.

Le corpus documentaire reste consultable lors du holdout : c’est la situation
normale d’une question sur des documents. Ce sont les questions d’évaluation et
leurs annotations qui restent réservées, sans guider les réglages.

### 2. Faire fonctionner une baseline

Choisir un modèle déjà entraîné capable de suivre des consignes, et conserver
le même modèle et les mêmes paramètres de génération dans les comparaisons.
Mesurer ses réponses sans documents, puis avec une recherche lexicale simple,
par exemple TF-IDF. Inspecter les passages retenus avant de regarder les réponses.

### 3. Réimplémenter le cœur du pipeline

Écrire explicitement le découpage, l’association passages/sources, la sélection
des `top-k` passages, l’assemblage du contexte et l’appel au générateur. Utiliser
des bibliothèques pour le modèle et les calculs, tout en gardant chaque étape
du pipeline visible et inspectable.

Fixer un budget de contexte et commencer avec `k = 3`. Donner une consigne courte :
répondre à partir des passages, citer leurs identifiants et signaler lorsque
les informations sont insuffisantes. Vérifier si le modèle respecte effectivement
cette consigne sur les questions sans réponse.

### 4. Ajouter la recherche par embeddings

Encoder les passages et la question avec un encodeur adapté à la recherche.
Normaliser les vecteurs, calculer leurs similarités cosinus et prendre les
`top-k` passages. Comparer cette variante à la recherche lexicale sur les mêmes
questions, avec le même générateur et le même budget de contexte.

### 5. Comparer, faire une ablation et évaluer

Comparer sur les questions de développement :

| Variante | Ce qu’elle permet d’examiner |
|---|---|
| LLM sans documents | Référence sans apport documentaire |
| RAG lexical | Apport d’une première recherche simple |
| RAG par embeddings | Effet du changement de méthode de recherche |
| LLM avec passages de référence fournis manuellement | Capacité à répondre quand les bonnes informations sont présentes |

La dernière variante est un diagnostic sur les questions ayant une réponse dans
le corpus ; ses passages sont choisis à partir des annotations de référence et
ne représentent pas une recherche automatique.

Faire une petite ablation, par exemple `k = 1` contre `k = 3`, en conservant le
reste du pipeline. Choisir la configuration sur le développement, puis la figer
et exécuter l’évaluation finale sur le holdout.

Mesurer séparément :

- **Recherche** : part des questions répondables pour lesquelles au moins un
  passage justificatif est retrouvé dans les `top-k` ; pour une réponse exigeant
  plusieurs passages, vérifier aussi que tous les éléments nécessaires sont présents.
- **Réponse** : exactitude par rapport à la réponse attendue, affirmations
  effectivement soutenues par les passages cités, et abstention sur les questions
  sans réponse. Une grille manuelle suffit pour vingt questions.
- **Coût** : temps de réponse et nombre de tokens de contexte, en distinguant
  la préparation de l’index du traitement d’une question.

Documenter au moins une erreur de recherche et une erreur de génération si elles
apparaissent. Une amélioration n’est pas garantie : un mauvais passage peut aussi
dégrader la réponse.

## Livrables et critère de maîtrise

À produire en suivant la structure du laboratoire :

- un pipeline minimal dans `reimplementation/`, avec passages et scores consultables ;
- un protocole d’évaluation dans `evaluation/` ;
- une note dans `results/` contenant configuration, comparaison, ablation, résultats
  du holdout et exemples d’échecs.

Le jalon est maîtrisé lorsque tu peux expliquer chaque étape, remplacer la méthode
de recherche, montrer quelles sources soutiennent une réponse, et diagnostiquer
si un échec vient du découpage, de la recherche, du contexte ou de la génération.

Les recherches itératives, le reranking et la mémoire persistante constituent des
prolongements ; les mécanismes pilotés par un agent seront approfondis dans la
voie « Agents et outils ».
