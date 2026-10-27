import test from "node:test";
import assert from "node:assert/strict";
import { JSDOM } from "jsdom";
import { renderOcrReview } from "../ocr-view.js";

function host() {
  globalThis.document = new JSDOM("<main></main>").window.document;
  return document.querySelector("main");
}

test("OCR comparison displays unknown retention and stale baseline without executing untrusted text", () => {
  const root = host(); let page;
  renderOcrReview(root, { status: "succeeded", page_number: 2, parser_available: true, can_start: true, document_status: "processed", baseline_changed: true,
    result: { baseline: { text: "<img src=x onerror=alert(1)>" }, candidate: { text: "候选17天" }, comparison: { baseline_token_retention: null, warnings: ["<script>alert(1)</script>"], missing_numbers: [{value: "83%", count: 1}] } }
  }, (number) => { page = number; });
  assert.equal(root.querySelector("img, script"), null);
  assert.match(root.textContent, /历史基线/);
  assert.match(root.textContent, /无法计算/);
  assert.match(root.textContent, /尚未入库/);
  assert.match(root.textContent, /83%（1 次）/);
  root.querySelector("button").click(); assert.equal(page, 2);
});

test("running and failed OCR preserve status and show existing baseline", () => {
  const root = host();
  renderOcrReview(root, {status: "running", page_number: 1, parser_available: true, can_start: true, document_status: "processed", result: {baseline: {text: "旧正文"}}}, () => {});
  assert.match(root.textContent, /可关闭页面后回来查看/);
  assert.match(root.textContent, /旧正文/);
  assert.match(root.textContent, /待生成/);
  renderOcrReview(root, {status: "failed", page_number: 1, parser_available: false, document_status: "failed", error_message: "服务不可用"}, () => {});
  assert.match(root.textContent, /现有索引未修改/);
  assert.match(root.textContent, /服务不可用/);
});
