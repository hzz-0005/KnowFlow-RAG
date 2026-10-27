"""Gold-evidence metrics and stage diagnostics; no self-confidence probabilities."""

from knowflow.app.evaluation import ndcg_at_k, reciprocal_rank

METRICS_VERSION = "gold-evidence-v3-citation-excerpts"


def normalize(text):
    return " ".join(text.casefold().split())


def evidence_match(row, gold):
    return row.get("filename") == gold["filename"] and normalize(gold["contains"]) in normalize(
        row.get("content", row.get("quote", ""))
    )


def citation_excerpt_present(quote, content):
    """Validate the API's excerpt join format, not arbitrary generated ellipses."""
    source = normalize(content)
    if normalize(quote) and normalize(quote) in source:
        return True
    # run_query joins already-validated supporting quotes using this exact separator.
    fragments = quote.split("\n…\n")
    return len(fragments) > 1 and all(
        len(fragment.strip()) >= 4 and normalize(fragment) in source for fragment in fragments
    )


def evaluate_trace(case, audit, response, indexed, parsed):
    gold = case["relevant_evidence"]
    base, candidates, context = (audit.get(key, []) for key in ("base", "candidates", "context"))
    citations = response.get("citations", [])
    citation_rows = [{**row, "content": row.get("quote", "")} for row in citations]
    ids = [row["chunk_id"] for row in base]
    relevant_chunks = {
        row["chunk_id"] for row in indexed if any(evidence_match(row, fact) for fact in gold)
    }
    matches = []
    for fact in gold:
        stored = [row for row in indexed if evidence_match(row, fact)]
        # Source use and exact excerpt coverage are different measurements. Credit a
        # source only when its gold evidence reached the model and its actual excerpt
        # is nonempty and present in that same supplied passage.
        citation_source_ranks = [
            i
            for i, cited in enumerate(citations, 1)
            if any(
                cited.get("chunk_id") == supplied.get("chunk_id")
                and cited.get("filename") == supplied.get("filename")
                and evidence_match(supplied, fact)
                and citation_excerpt_present(cited.get("quote", ""), supplied.get("content", ""))
                for supplied in context
            )
        ]
        locations = {
            name: [i for i, row in enumerate(rows, 1) if evidence_match(row, fact)]
            for name, rows in [
                ("base", base),
                ("candidates", candidates),
                ("context", context),
                ("citations", citation_rows),
            ]
        }
        locations["citation_sources"] = citation_source_ranks
        original = [row for row in parsed if evidence_match(row, fact)]
        if not original:
            stage = "parse_or_annotation_mismatch"
        elif not stored:
            stage = "chunking_or_quote_boundary"
        elif not locations["candidates"]:
            stage = "retrieval_miss"
        elif not locations["context"]:
            stage = "assessment_or_context_filter"
        elif response["refused"]:
            stage = "refused_with_evidence"
        elif not locations["citations"]:
            stage = (
                "citation_excerpt_shorter_than_gold"
                if citation_source_ranks
                else "evidence_not_cited"
            )
        else:
            stage = "evidence_chain_complete"
        matches.append(
            {
                **fact,
                "ranks": locations,
                "indexed_chunk_ids": [r["chunk_id"] for r in stored],
                "parsed_quote_found": bool(original),
                "stage": stage,
            }
        )
    k = audit["top_k"]
    metrics = {
        "mrr": reciprocal_rank(ids, relevant_chunks) if gold else None,
        f"ndcg@{k}": ndcg_at_k(ids, relevant_chunks, k) if gold else None,
        f"precision@{k}": sum(any(evidence_match(row, f) for f in gold) for row in base[:k]) / k
        if gold
        else None,
    }
    for stage in ("base", "candidates", "context", "citations", "citation_sources"):
        coverage = sum(bool(row["ranks"][stage]) for row in matches) / len(gold) if gold else None
        metrics[f"{stage}_evidence_recall"] = coverage
        metrics[f"{stage}_complete_evidence"] = float(coverage == 1) if gold else None
    metrics[f"recall@{k}"] = metrics["base_evidence_recall"]
    unavailable = (response.get("retrieval_diagnostics") or {}).get("status") in {
        "unavailable",
        "partial",
    }
    metrics["refusal_decision_correct"] = (
        None if unavailable else float(response["refused"] == case["should_refuse"])
    )
    relevant_returns = [bool(any(evidence_match(row, f) for f in gold)) for row in base]
    metrics["irrelevant_slot_rate"] = (
        1 - sum(relevant_returns) / len(base) if gold and base else None
    )
    return {
        "metrics": metrics,
        "evidence_checks": matches,
        "refused": response["refused"],
        "answer": response["answer"],
        "graph_expansion": audit.get("graph_expansion"),
        "query_plan": response.get("query_plan"),
        "answer_validation": response.get("answer_validation"),
        "diagnostic_note": (
            "文字匹配以固定标注为准，未标注相关性可能被漏算。"
            "引用来源覆盖只证明含金标的上下文来源被引用，不证明节选支持答案；"
            "引文全文匹配仍单独保留，答案依据需另行审核。"
            "阶段标签用于定位，不能证明解析或答案一定错误。"
        ),
    }


def summarize(cases):
    keys = {key for row in cases for key in row.get("metrics", {})}
    result, denominators = {}, {}
    for key in sorted(keys):
        values = [
            row["metrics"][key] for row in cases if row.get("metrics", {}).get(key) is not None
        ]
        result[key] = sum(values) / len(values) if values else None
        denominators[key] = len(values)
    return result, denominators


def compare_variants(before, after):
    earlier = {row["id"]: row for row in before}
    changes = []
    for row in after:
        old = earlier.get(row["id"])
        if not old or row.get("error") or old.get("error"):
            changes.append({"id": row["id"], "status": "unavailable"})
            continue
        left = old["metrics"].get("context_evidence_recall")
        right = row["metrics"].get("context_evidence_recall")
        delta = right - left if left is not None and right is not None else None
        changes.append(
            {
                "id": row["id"],
                "status": "improved"
                if delta and delta > 0
                else "regressed"
                if delta and delta < 0
                else "unchanged"
                if delta == 0
                else "not_applicable",
                "context_recall_delta": delta,
                "mrr_before": old["metrics"].get("mrr"),
                "mrr_after": row["metrics"].get("mrr"),
                "refusal_before": old["metrics"].get("refusal_decision_correct"),
                "refusal_after": row["metrics"].get("refusal_decision_correct"),
            }
        )
    return changes


def equivalent_question_consistency(cases):
    groups = {}
    for row in cases:
        if row.get("equivalence_group"):
            groups.setdefault(row["equivalence_group"], []).append(row)
    results = []
    for group, rows in groups.items():
        if len(rows) < 2:
            continue
        if any(row.get("error") for row in rows):
            results.append({"group": group, "consistent": None, "reason": "case_unavailable"})
            continue
        signatures = [
            {
                (f["filename"], normalize(f["contains"]))
                for f in row["evidence_checks"]
                if f["ranks"]["context"]
            }
            for row in rows
        ]
        results.append(
            {
                "group": group,
                "consistent": all(s == signatures[0] for s in signatures),
                "all_complete": all(
                    row["metrics"].get("context_complete_evidence") == 1 for row in rows
                ),
                "case_ids": [row["id"] for row in rows],
            }
        )
    return results


def regression_check(report, baseline):
    if report.get("guard_budget_contract") != baseline.get("guard_budget_contract"):
        return {
            "status": "not_comparable",
            "reason": "审核/纠错预算契约或额度不同/历史未记录，不能作为同核验策略回归。",
        }
    if report.get("relevance_budget_contract") != baseline.get("relevance_budget_contract"):
        return {
            "status": "not_comparable",
            "reason": "相关性预算契约或额度不同/历史未记录，不能作为同检查策略回归。",
        }
    if report.get("model_routing_contract") != baseline.get("model_routing_contract"):
        return {
            "status": "not_comparable",
            "reason": "模型路由契约或配置不同/历史未记录，不能作为同模型策略回归。",
        }
    if report.get("query_rewrite_contract") != baseline.get("query_rewrite_contract"):
        return {
            "status": "not_comparable",
            "reason": "查询改写契约不同/历史未记录，不能作为同策略回归。",
        }
    if report.get("answer_budget_contract") != baseline.get("answer_budget_contract"):
        return {
            "status": "not_comparable",
            "reason": "首稿预算契约或额度不同/历史未记录，不能作为同预算回归。",
        }
    if any(
        report.get(key) != baseline.get(key) for key in ("dataset_sha256", "corpus_sha256", "top_k")
    ):
        return {
            "status": "not_comparable",
            "reason": "题集、语料快照或 top-k 不同，不能直接判定回归。",
        }
    regressions, unknown = [], []
    for strategy, variant in report["variants"].items():
        earlier = {
            row["id"]: row
            for row in baseline.get("variants", {}).get(strategy, {}).get("cases", [])
        }
        for row in variant["cases"]:
            old = earlier.get(row["id"])
            if old is None:
                unknown.append(f"{strategy}/{row['id']}: 无历史记录")
                continue
            checked_metrics = (
                "base_evidence_recall",
                "context_evidence_recall",
                "citations_evidence_recall",
                "mrr",
                "refusal_decision_correct",
            ) + tuple(
                key for key in old.get("metrics", {}) if key.startswith(("ndcg@", "precision@"))
            )
            if report.get("metrics_version") == baseline.get("metrics_version") == METRICS_VERSION:
                checked_metrics += ("citation_sources_evidence_recall",)
            for metric in checked_metrics:
                before = old.get("metrics", {}).get(metric)
                after = row.get("metrics", {}).get(metric)
                if before is not None and after is None:
                    unknown.append(f"{strategy}/{row['id']}/{metric}: 本次未知")
                elif before is not None and after is not None and after < before - 1e-9:
                    regressions.append(
                        {
                            "strategy": strategy,
                            "case_id": row["id"],
                            "metric": metric,
                            "before": before,
                            "after": after,
                        }
                    )
    return {
        "status": "regressed"
        if regressions
        else "incomplete"
        if unknown
        else "no_regression_found",
        "regressions": regressions,
        "unknown": unknown,
        "note": "只对固定标注的证据/拒答指标判回归；模型评审变化单独查看，不当作确定事实。",
    }
