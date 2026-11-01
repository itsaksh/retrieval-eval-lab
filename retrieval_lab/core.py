from __future__ import annotations

from collections import Counter
import math
import re


def tokens(text: str) -> list[str]:
    """Normalize prose, snake_case and simple camelCase into word tokens."""
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return re.findall(r"[a-z0-9]+", text.casefold())


class Retriever:
    def __init__(self, documents: list[dict]):
        if not documents:
            raise ValueError("documents cannot be empty")
        self.documents = {}
        for doc in documents:
            if not isinstance(doc, dict) or not isinstance(doc.get("id"), str) or not doc["id"] or not isinstance(doc.get("text"), str):
                raise ValueError("each document requires a nonempty string id and string text")
            if doc["id"] in self.documents:
                raise ValueError("document ids must be unique")
            self.documents[doc["id"]] = {"id": doc["id"], "text": doc["text"],
                                          **({"source": doc["source"]} if "source" in doc else {})}
        self.counts = {key: Counter(tokens(doc["text"])) for key, doc in self.documents.items()}
        self.lengths = {key: sum(count.values()) for key, count in self.counts.items()}
        self.avg_length = sum(self.lengths.values()) / len(self.lengths)
        self.df = Counter(term for count in self.counts.values() for term in count)

    def search(self, query: str, k=3, method="bm25") -> list[dict]:
        if not isinstance(query, str) or type(k) is not int or k < 1:
            raise ValueError("query must be text and k must be a positive integer")
        if method not in ("overlap", "bm25"):
            raise ValueError("method must be overlap or bm25")
        # Stable summation order makes saved floating-point scores reproducible
        # across processes with different Python hash seeds.
        query_terms = sorted(set(tokens(query)))
        n, k1, b = len(self.documents), 1.5, 0.75
        hits = []
        for key, counts in self.counts.items():
            score = 0.0
            for term in query_terms:
                tf = counts[term]
                if tf == 0:
                    continue
                if method == "overlap":
                    score += 1
                else:
                    idf = math.log(1 + (n - self.df[term] + 0.5) / (self.df[term] + 0.5))
                    norm = (1-b + b*self.lengths[key]/self.avg_length) if self.avg_length else 1
                    score += idf * tf * (k1+1) / (tf+k1*norm)
            if score > 0:
                hits.append({**self.documents[key], "score": score})
        return sorted(hits, key=lambda h: (-h["score"], h["id"]))[:k]


def metrics(ranked_ids: list[str], relevant_ids: set[str], k: int) -> dict:
    if not relevant_ids or type(k) is not int or k < 1:
        raise ValueError("nonempty relevance labels and a positive integer k are required")
    if len(ranked_ids) != len(set(ranked_ids)):
        raise ValueError("ranked ids must be unique")
    top = ranked_ids[:k]
    found = set(top) & relevant_ids
    reciprocal = next((1.0/i for i, key in enumerate(top, 1) if key in relevant_ids), 0.0)
    return {"recall_at_k": len(found)/len(relevant_ids), "reciprocal_rank_at_k": reciprocal,
            "hit_at_k": float(bool(found))}


def pack_context(hits: list[dict], max_chars: int) -> dict:
    """Select whole documents in ranking order within an exact character budget."""
    if type(max_chars) is not int or max_chars < 0:
        raise ValueError("max_chars must be a nonnegative integer")
    pieces, selected, skipped = [], [], []
    used = 0
    for hit in hits:
        piece = f"[{hit['id']}]\n{hit['text']}"
        extra = len(piece) + (2 if pieces else 0)
        if used + extra <= max_chars:
            pieces.append(piece)
            selected.append(hit["id"])
            used += extra
        else:
            skipped.append(hit["id"])
    context = "\n\n".join(pieces)
    return {"text": context, "characters": len(context), "selected_ids": selected,
            "skipped_ids": skipped, "budget_characters": max_chars}


def evaluate(documents, queries, k=3, max_chars=600):
    retriever = Retriever(documents)
    if not queries:
        raise ValueError("queries cannot be empty")
    seen = set()
    for q in queries:
        if not isinstance(q, dict) or not isinstance(q.get("id"), str) or not q["id"] or q["id"] in seen:
            raise ValueError("queries require unique nonempty ids")
        seen.add(q["id"])
        if not isinstance(q.get("query"), str) or not q["query"].strip():
            raise ValueError("each query requires nonempty query text")
        relevant = q.get("relevant")
        if not isinstance(relevant, list) or not relevant or any(not isinstance(x, str) for x in relevant):
            raise ValueError("each query requires a nonempty relevant-id list")
        if len(relevant) != len(set(relevant)) or set(relevant)-set(retriever.documents):
            raise ValueError("relevance labels must be unique and refer to existing documents")
    methods = {}
    for method in ("overlap", "bm25"):
        rows = []
        for q in queries:
            hits = retriever.search(q["query"], k=k, method=method)
            scores = metrics([h["id"] for h in hits], set(q["relevant"]), k)
            packed = pack_context(hits, max_chars)
            packed_scores = metrics(packed["selected_ids"], set(q["relevant"]), k)
            rows.append({"query_id": q["id"], "query": q["query"], "relevant_ids": q["relevant"],
                         "ranked_ids": [h["id"] for h in hits], **scores,
                         "packed_recall_at_k": packed_scores["recall_at_k"],
                         "context": packed})
        methods[method] = {
            "mean_recall_at_k": sum(r["recall_at_k"] for r in rows)/len(rows),
            "mrr_at_k": sum(r["reciprocal_rank_at_k"] for r in rows)/len(rows),
            "hit_rate_at_k": sum(r["hit_at_k"] for r in rows)/len(rows),
            "mean_packed_recall_at_k": sum(r["packed_recall_at_k"] for r in rows)/len(rows),
            "queries": rows,
        }
    return {"k": k, "context_budget_characters": max_chars, "documents": len(documents),
            "queries": len(queries), "methods": methods,
            "scope": "Lexical retrieval evaluation; no generation or semantic embedding model is used."}
