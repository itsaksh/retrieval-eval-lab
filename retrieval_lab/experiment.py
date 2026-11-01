"""Ranking and packing do not receive relevance labels. Labels are scoring-only."""
from __future__ import annotations

import math
from .core import Retriever, metrics


def reciprocal_rank_fusion(rankings, constant=60):
    if type(constant) is not int or constant < 1:
        raise ValueError("fusion constant must be a positive integer")
    scores, documents = {}, {}
    for ranking in rankings:
        ids = [hit["id"] for hit in ranking]
        if len(ids) != len(set(ids)):
            raise ValueError("each fusion ranking must have unique ids")
        for rank, hit in enumerate(ranking, 1):
            key = hit["id"]
            scores[key] = scores.get(key, 0) + 1 / (constant + rank)
            documents[key] = hit
    return [{**documents[key], "score": scores[key]}
            for key in sorted(scores, key=lambda key: (-scores[key], key))]


class ComparisonRetriever:
    def __init__(self, documents, encoder=None):
        # Explicit allowlist prevents relevance labels or arbitrary metadata from entering ranking.
        if any(not isinstance(doc, dict) for doc in documents):
            raise ValueError("documents must be objects")
        clean = [{key: doc[key] for key in ("id", "text", "source") if key in doc} for doc in documents]
        self.lexical = Retriever(clean)
        self.documents = self.lexical.documents
        self.ids = list(self.documents)
        self.encoder = encoder
        self.methods = ["overlap", "bm25"]
        self.truncated_documents = []
        self.truncated_queries = []
        self.query_vectors = {}
        if encoder is not None:
            self.vectors, truncated = encoder.encode([self.documents[key]["text"] for key in self.ids])
            self.truncated_documents = [self.ids[i] for i in truncated]
            self.methods += ["dense", "hybrid_rrf"]

    def search(self, query, k, method):
        if not isinstance(query, str) or type(k) is not int or k < 1 or method not in self.methods:
            raise ValueError("invalid query, k or retrieval method")
        if not query.strip():
            return []
        if method in ("overlap", "bm25"):
            return self.lexical.search(query, k, method)
        if method == "hybrid_rrf":
            return reciprocal_rank_fusion([self.search(query, len(self.ids), "bm25"),
                                           self.search(query, len(self.ids), "dense")])[:k]
        if query not in self.query_vectors:
            vector, truncated = self.encoder.encode([query])
            self.query_vectors[query] = vector[0]
            if truncated:
                self.truncated_queries.append(query)
        scores = self.vectors @ self.query_vectors[query]
        return [{**self.documents[self.ids[i]], "score": float(scores[i])}
                for i in sorted(range(len(self.ids)), key=lambda i: (-float(scores[i]), self.ids[i]))[:k]]


def citation(hit):
    source = hit.get("source", {})
    location = (f" | {source['path']}:{source['start_line']}-{source['end_line']}"
                if source else "")
    return f"[{hit['id']}{location}]\n{hit['text']}"


def pack_tokens(hits, budget, counter, strategy="whole_chunks"):
    """Two label-blind policies; count the complete string including citations."""
    if type(budget) is not int or budget < 0 or strategy not in ("whole_chunks", "prefix"):
        raise ValueError("invalid token budget or packing strategy")
    ids = [h["id"] for h in hits]
    if len(ids) != len(set(ids)):
        raise ValueError("packing ids must be unique")
    pieces, selected, partial = [], [], []
    full = "\n\n".join(citation(h) for h in hits)
    if strategy == "whole_chunks":
        for hit in hits:
            piece = citation(hit)
            candidate = "\n\n".join([*pieces, piece])
            if counter.count(candidate) <= budget:
                pieces.append(piece)
                selected.append(hit["id"])
        context = "\n\n".join(pieces)
    else:
        # Preserve all source characters in order until the first token-budget overflow.
        # Binary search on character counts is unsafe for non-monotonic subword counts.
        # Encode/decode at token boundaries when available; otherwise use an exact scan.
        context = counter.prefix(full, budget) if hasattr(counter, "prefix") else ""
        if not hasattr(counter, "prefix"):
            for end in range(1, len(full) + 1):
                if counter.count(full[:end]) <= budget:
                    context = full[:end]
                else:
                    break
        position = 0
        for hit in hits:
            piece = citation(hit)
            end = position + len(piece)
            if len(context) >= end:
                selected.append(hit["id"])
            elif len(context) > position:
                partial.append(hit["id"])
            position = end + 2
    count, original = counter.count(context), counter.count(full)
    if count > budget:
        raise ValueError("token counter returned an over-budget context")
    return {"text": context, "tokens": count, "characters": len(context), "budget_tokens": budget,
            "unpacked_tokens": original, "token_reduction_fraction": 1-count/original if original else 0,
            "selected_ids": selected, "partial_ids": partial,
            "skipped_ids": [key for key in ids if key not in selected and key not in partial]}


def ndcg(ranked_ids, relevant, k):
    gain = sum(1 / math.log2(rank+1) for rank, key in enumerate(ranked_ids[:k], 1) if key in relevant)
    ideal = sum(1 / math.log2(rank+1) for rank in range(1, min(len(relevant), k)+1))
    return gain / ideal


def evaluate_comparison(documents, queries, counter, encoder=None, k=5, budgets=(64, 128, 256)):
    if type(k) is not int or k < 1 or not budgets or len(budgets) != len(set(budgets)):
        raise ValueError("positive k and unique nonempty budgets are required")
    if any(type(b) is not int or b < 0 for b in budgets) or not queries:
        raise ValueError("budgets must be nonnegative integers; queries cannot be empty")
    retriever = ComparisonRetriever(documents, encoder)
    seen = set()
    for q in queries:
        if not isinstance(q, dict) or not isinstance(q.get("id"), str) or not q["id"] or q["id"] in seen:
            raise ValueError("query ids must be unique and nonempty")
        seen.add(q["id"])
        relevant = q.get("relevant")
        if (not isinstance(q.get("query"), str) or not q["query"].strip() or not isinstance(relevant, list)
                or not relevant or any(not isinstance(key, str) for key in relevant)
                or len(relevant) != len(set(relevant)) or set(relevant)-set(retriever.documents)):
            raise ValueError("queries need text and valid unique relevance ids")
    results = {}
    for method in retriever.methods:
        rows = []
        for q in queries:
            hits = retriever.search(q["query"], k, method)
            ids = [h["id"] for h in hits]
            relevant = set(q["relevant"])
            packs = []
            for budget in budgets:
                for strategy in ("prefix", "whole_chunks"):
                    packed = pack_tokens(hits, budget, counter, strategy)
                    retained = len(set(packed["selected_ids"]) & relevant)
                    retrieved = len(set(ids) & relevant)
                    packs.append({"strategy": strategy, **packed, "packed_recall": retained/len(relevant),
                                  "evidence_retention": retained/retrieved if retrieved else None})
            rows.append({"query_id": q["id"], "query": q["query"], "relevant_ids": q["relevant"],
                         "ranked_ids": ids, "hits": hits, **metrics(ids, relevant, k),
                         "ndcg_at_k": ndcg(ids, relevant, k), "packs": packs})
        aggregates = []
        for budget in budgets:
            for strategy in ("prefix", "whole_chunks"):
                packs = [next(p for p in row["packs"] if p["budget_tokens"] == budget and p["strategy"] == strategy)
                         for row in rows]
                retentions = [p["evidence_retention"] for p in packs if p["evidence_retention"] is not None]
                aggregates.append({"budget_tokens": budget, "strategy": strategy,
                                   "mean_packed_recall": sum(p["packed_recall"] for p in packs)/len(packs),
                                   "mean_tokens": sum(p["tokens"] for p in packs)/len(packs),
                                   "mean_token_reduction": sum(p["token_reduction_fraction"] for p in packs)/len(packs),
                                   "mean_evidence_retention_when_retrieved": sum(retentions)/len(retentions) if retentions else None})
        results[method] = {"mean_recall_at_k": sum(r["recall_at_k"] for r in rows)/len(rows),
                           "mrr_at_k": sum(r["reciprocal_rank_at_k"] for r in rows)/len(rows),
                           "mean_ndcg_at_k": sum(r["ndcg_at_k"] for r in rows)/len(rows),
                           "hit_rate_at_k": sum(r["hit_at_k"] for r in rows)/len(rows),
                           "packing": aggregates, "queries": rows}
    return {"schema_version": 2, "documents": len(documents), "queries": len(queries), "k": k,
            "budgets": list(budgets), "tokenizer": counter.name, "methods": results,
            "embedding": encoder.provenance if encoder else None,
            "truncated_document_ids": retriever.truncated_documents,
            "truncated_query_texts": retriever.truncated_queries,
            "scope": "Small authored relevance set, not an independent benchmark. Complete-chunk recall only; no answer generation. WordPiece context-only budgets exclude prompts and special tokens."}
