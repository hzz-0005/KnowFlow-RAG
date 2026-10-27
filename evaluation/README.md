# Retrieval evaluation

逐条检索相关性检测和误采纳/漏采纳回归，见 [检索置信检测说明](../docs/retrieval-confidence.md)。固定 24 对标签位于 `relevance_cases.json`，首轮历史基线位于 `baselines/relevance-v1-mimo-20261021.json`。它与下文的端到端检索指标评测是两种不同测试。

`dataset.json` is a small, original synthetic benchmark. Every positive label points to an exact evidence phrase in `corpus/`; the set covers single facts, exact identifiers, paraphrases, cross-document retrieval, a two-hop path, a no-answer question, and a distractor. It is a smoke/learning dataset, not a claim about production quality.

To prepare a reproducible run:

1. Start the Compose services, apply migrations, register in `/docs`, and create a knowledge base.
2. Upload every file in `corpus/` to that knowledge base. Wait until each document is `processed`.
3. For `graph` and `graph_hybrid`, request `POST /api/v1/documents/{id}/graph-index` for each corpus document and wait for `graph_status` to become `processed`.
4. Configure the providers required by the selected strategy in the local `.env`. Vector strategies need the embedding provider; graph indexing and answer generation need the chat provider; `hybrid_rerank` also needs a reranker endpoint.
5. Run the same dataset and `top_k` for each strategy:

```powershell
.\.venv\Scripts\python.exe -m knowflow.app.evaluate_retrieval `
  --knowledge-base-id <KNOWLEDGE_BASE_UUID> `
  --strategy hybrid `
  --top-k 5
```

The command writes a JSON run file under `evaluation/results/`. Each run records the dataset version, strategy, model identifiers, code revision, per-query evidence matches, Recall@K, Precision@K, MRR, nDCG@K, retrieval latency, and reranker latency when applicable. Output files stay local by default because user datasets and document names may be sensitive.

This evaluator measures retrieval only. It does not claim answer correctness or refusal quality; those require inspecting generated answers against the reference answer and evidence. Keep the corpus, knowledge base, model versions, and parameters fixed when comparing strategies. The synthetic set is intentionally small, so add a larger labeled set before drawing résumé-level conclusions.
