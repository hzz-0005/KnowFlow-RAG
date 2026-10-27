import test from "node:test";
import assert from "node:assert/strict";
import { JSDOM } from "jsdom";
import { renderDiagnostics } from "../diagnostics-view.js";

function host() {
  globalThis.document = new JSDOM("<main></main>").window.document;
  return document.querySelector("main");
}

function baseData() {
  return {
    note: "机械检查说明",
    schema_version: "persisted-integrity-v2",
    snapshot_sha256: "f".repeat(8),
    scope: {
      complete: true,
      content_complete: true,
      sources_complete: true,
      inspected: { blocks: 2, chunks: 2 },
      total: { blocks: 2, chunks: 2 },
    },
    observations: {
      token_budget: { observed: 0, unknown: 0, max_input_tokens: null, saved_budget: null },
      page_coverage: { status: "not_applicable" },
    },
    findings: [],
  };
}

test("consistency panel shows contract, counts and side-by-side evidence safely", () => {
  const root = host();
  const data = baseData();
  data.observations.source_consistency = {
    contract: "source-consistency-v1",
    status: "observed",
    spans: { compared: 1, match: 0, normalized_match: 0, mismatch: 1, not_comparable: 1 },
    reasons: { source_text_truncated: 1 },
    examples: [
      {
        status: "mismatch", reason: null, normalizations: [], ordinal: 0, block_key: "b0",
        page_number: 3, chunk_range: [0, 11], text_range: [0, 11],
        source: "<img src=x onerror=alert(1)> world", source_truncated: false,
        excerpt: "XXXXXXXXXXX", excerpt_truncated: false,
      },
      {
        status: "not_comparable", reason: "source_text_truncated", normalizations: [],
        ordinal: 1, block_key: "b1", source: null, source_truncated: null,
        excerpt: "节选片段", excerpt_truncated: true,
      },
    ],
    note: "只证明已保存来源文字与切片节选一致。",
  };
  renderDiagnostics(root, data, () => {});
  const text = root.textContent;
  assert.match(text, /source-consistency-v1/);
  assert.match(text, /不一致 1/);
  assert.match(text, /无法比较 1/);
  assert.match(text, /source_text_truncated/);
  assert.match(text, /XXXXXXXXXXX/);
  assert.match(text, /节选片段/);
  assert.match(text, /（截断展示）/);
  assert.match(text, /（本次未读到文字，无法对照）/);
  assert.equal(root.querySelector("img, script"), null);
  assert.equal(root.querySelectorAll(".consistency-grid").length, 2);
  assert.equal(root.querySelectorAll(".consistency-pair.is-mismatch").length, 1);
});

test("v1 semantics report not_run without inventing comparisons", () => {
  const root = host();
  const data = baseData();
  data.schema_version = "persisted-integrity-v1";
  data.observations.source_consistency = {
    contract: null,
    status: "not_run",
    note: "本次运行未读取来源正文，未执行一致性比对。",
  };
  renderDiagnostics(root, data, () => {});
  assert.match(root.textContent, /未执行一致性比对/);
  assert.equal(root.querySelectorAll(".consistency-pair").length, 0);
});

test("unknown coverage and partial body reading are explicit", () => {
  const root = host(), data = baseData();
  data.scope.source_texts_complete = false;
  data.observations.source_consistency = {
    contract: "source-consistency-v2", status: "unknown",
    spans: { compared: 0, match: 0, normalized_match: 0, mismatch: 0, not_comparable: 2 },
    coverage: { chunks_inspected: 2, chunks_without_span_objects: 1, span_objects_inspected: 2 },
    work: { characters_inspected: 30, character_budget: 30 },
    reasons: { ambiguous_character_mapping: 2 }, examples: [], note: "机械检查",
  };
  renderDiagnostics(root, data, () => {});
  assert.match(root.textContent, /没有可判定的文字比对结果/);
  assert.match(root.textContent, /无来源项 1 个（不计入来源项分母）/);
  assert.match(root.textContent, /不能确定一一对应/);
  assert.match(root.textContent, /来源正文读取不完整/);
  assert.match(root.textContent, /30 \/ 30 字符/);
});

test("a late mismatch shows the actual differing window and codepoint position", () => {
  const root = host(), data = baseData();
  data.observations.source_consistency = {
    contract: "source-consistency-v2", status: "observed",
    spans: { compared: 1, match: 0, normalized_match: 0, mismatch: 1, not_comparable: 0 },
    reasons: {}, examples: [{ status: "mismatch", ordinal: 0,
      source: "A".repeat(200) + "OLD", excerpt: "A".repeat(200) + "NEW",
      source_truncated: true, excerpt_truncated: true,
      difference: { source_offset: 900, excerpt_offset: 900 },
    }], note: "机械检查",
  };
  renderDiagnostics(root, data, () => {});
  assert.match(root.textContent, /来源第 901 字符/);
  assert.match(root.querySelector(".consistency-grid").textContent, /OLD.*NEW/s);
});
