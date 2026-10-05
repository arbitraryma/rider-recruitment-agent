from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import numpy as np
import jieba


# =========================
# 1. Load FAQ + Policy
# =========================

documents = []

# FAQ
with open("faq.txt", "r", encoding="utf-8") as f:
    faq_text = f.read()

faq_chunks = [
    x.strip()
    for x in faq_text.split("\n\n")
    if x.strip()
]

for chunk in faq_chunks:
    documents.append({
        "text": chunk,
        "source": "faq"
    })


# Policy / instructions
with open("instructions.txt", "r", encoding="utf-8") as f:
    policy_text = f.read()

# 你的 instructions.txt 基本是一行一个政策，所以按行切最合理
policy_chunks = [
    line.strip()
    for line in policy_text.splitlines()
    if line.strip()
]

for chunk in policy_chunks:
    documents.append({
        "text": chunk,
        "source": "policy"
    })


print(f"Loaded {len(faq_chunks)} FAQ chunks")
print(f"Loaded {len(policy_chunks)} policy chunks")
print(f"Total documents: {len(documents)}")


# =========================
# 2. Prepare texts
# =========================

chunks = [
    doc["text"]
    for doc in documents
]


# =========================
# 3. Semantic embedding model
# =========================

embedding_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

chunk_embeddings = embedding_model.encode(
    chunks,
    normalize_embeddings=True
)


# =========================
# 4. BM25 tokenizer
# =========================

def tokenize(text):
    return [
        word.strip()
        for word in jieba.lcut(text.lower())
        if word.strip()
    ]


tokenized_chunks = [
    tokenize(chunk)
    for chunk in chunks
]

bm25 = BM25Okapi(
    tokenized_chunks
)


# =========================
# 5. Reranker
# =========================

reranker = CrossEncoder(
    "BAAI/bge-reranker-base"
)


# =========================
# 6. Semantic search
# =========================

def semantic_search(
    query,
    top_k=10
):
    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )[0]

    scores = (
        chunk_embeddings
        @ query_embedding
    )

    indices = np.argsort(
        scores
    )[::-1][:top_k]

    return [
        {
            "index": int(i),
            "text": chunks[i],
            "source": documents[i]["source"],
            "semantic_score": float(scores[i])
        }
        for i in indices
    ]


# =========================
# 7. BM25 search
# =========================

def bm25_search(
    query,
    top_k=10
):
    query_tokens = tokenize(
        query
    )

    scores = bm25.get_scores(
        query_tokens
    )

    indices = np.argsort(
        scores
    )[::-1][:top_k]

    return [
        {
            "index": int(i),
            "text": chunks[i],
            "source": documents[i]["source"],
            "bm25_score": float(scores[i])
        }
        for i in indices
    ]


# =========================
# 8. Hybrid merge
# =========================

def hybrid_candidates(
    query,
    semantic_k=10,
    bm25_k=10
):
    semantic_results = semantic_search(
        query,
        top_k=semantic_k
    )

    bm25_results = bm25_search(
        query,
        top_k=bm25_k
    )

    merged = {}

    for r in semantic_results:
        idx = r["index"]

        merged[idx] = {
            "index": idx,
            "text": r["text"],
            "source": r["source"],
            "semantic_score": r["semantic_score"],
            "bm25_score": 0.0
        }

    for r in bm25_results:
        idx = r["index"]

        if idx not in merged:
            merged[idx] = {
                "index": idx,
                "text": r["text"],
                "source": r["source"],
                "semantic_score": 0.0,
                "bm25_score": r["bm25_score"]
            }

        else:
            merged[idx]["bm25_score"] = (
                r["bm25_score"]
            )

    return list(
        merged.values()
    )


# =========================
# 9. Reranking
# =========================

def rerank(
    query,
    candidates,
    top_k=5
):

    pairs = [
        [query, item["text"]]
        for item in candidates
    ]

    rerank_scores = reranker.predict(
        pairs
    )

    for item, score in zip(
        candidates,
        rerank_scores
    ):
        item["rerank_score"] = float(score)

        # Policy lines are the official rules, so give them
        # a small boost when reporting the combined score.
        if item["source"] == "policy":
            item["final_score"] = float(score) + 0.1
        else:
            item["final_score"] = float(score)


    # Separate policy and FAQ
    policy_results = [
        x for x in candidates
        if x["source"] == "policy"
    ]

    faq_results = [
        x for x in candidates
        if x["source"] == "faq"
    ]


    # Sort separately
    policy_results = sorted(
        policy_results,
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    faq_results = sorted(
        faq_results,
        key=lambda x: x["rerank_score"],
        reverse=True
    )


    final_results = []

    # If a policy result is reasonably relevant,
    # always keep the best policy first
    if (
        policy_results
        and policy_results[0]["rerank_score"] > 0.25
    ):
        final_results.append(
            policy_results[0]
        )


    # Fill remaining slots with strongest FAQ results
    for item in faq_results:

        if len(final_results) >= top_k:
            break

        final_results.append(item)


    return final_results[:top_k]


# =========================
# 10. Final retrieval
# =========================

def retrieve_faq(
    query,
    top_k=5
):
    candidates = hybrid_candidates(
        query,
        semantic_k=10,
        bm25_k=10
    )

    results = rerank(
        query,
        candidates,
        top_k=top_k
    )

    return results


# =========================
# 11. Test
# =========================

if __name__ == "__main__":

    query = (
        "这个工作可以兼职吗？"
    )

    results = retrieve_faq(
        query,
        top_k=5
    )

    for i, r in enumerate(
        results,
        1
    ):
        print(
            f"\nRank {i}"
        )

        print(
            "Source:",
            r["source"]
        )

        print(
            "Semantic:",
            round(
                r["semantic_score"],
                3
            )
        )

        print(
            "BM25:",
            round(
                r["bm25_score"],
                3
            )
        )

        print(
            "Rerank:",
            round(
                r["rerank_score"],
                3
            )
        )

        print(
            "Final:",
            round(
                r["final_score"],
                3
            )
        )

        print(
            r["text"]
        )