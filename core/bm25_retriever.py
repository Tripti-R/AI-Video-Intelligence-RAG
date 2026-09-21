from rank_bm25 import BM25Okapi
import re


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def build_bm25_index(chunks):
    tokenized_chunks = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    return BM25Okapi(tokenized_chunks)


def search_bm25(bm25, chunks, query, k=10):
    query_tokens = tokenize(query)

    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    results = []

    for i in ranked_indices[:k]:
        results.append({
            **chunks[i],
            "chunk_index": i,
            "bm25_score": float(scores[i])
        })

    return results