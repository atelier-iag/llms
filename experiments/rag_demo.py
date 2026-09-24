"""First RAG step: inspect retrieved passages and the prompt, without a LLM call."""

import argparse
from pathlib import Path

from reimplementation.retrieval import TfidfIndex, build_prompt, chunk_document


ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = (
    "results/simple-baseline.md",
    "results/rope-baseline.md",
    "results/gqa-baseline.md",
)
DEFAULT_QUESTION = "Combien de têtes Q et de groupes K/V utilise notre modèle GQA ?"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()
    if not args.question.strip() or args.k < 1:
        parser.error("provide a nonempty question and k >= 1")

    passages = [passage for name in DOCUMENTS
                for passage in chunk_document(name, (ROOT / name).read_text(encoding="utf-8"))]
    index = TfidfIndex(passages)
    hits = index.search(args.question, k=args.k)
    print(f"Corpus : {len(DOCUMENTS)} documents, {len(passages)} passages.")
    print(f"Index TF-IDF : {len(index.idf)} termes ; recherche cosinus.")
    print(f"Question : {args.question}\n")
    for rank, (score, passage) in enumerate(hits, 1):
        print(f"P{rank} : score {score:.4f} | {passage.source} | "
              f"mots {passage.start_word}:{passage.end_word}")
    if not hits:
        print("Aucun recouvrement lexical avec le corpus.")
    print("\nPROMPT CONSTRUIT POUR LE LLM\n")
    print(build_prompt(args.question, hits))
    print("Aucun LLM appelé : la démo s'arrête au prompt, sans réponse générée.")
    print("Un score de recherche positif ne garantit pas que le passage contient la réponse.")


if __name__ == "__main__":
    main()
