import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { randomUUID } from "node:crypto";
import { test } from "node:test";
import { JSDOM } from "jsdom";

async function load(url, token=null) {
  const html = await readFile(new URL("../index.html", import.meta.url), "utf8");
  const source = await readFile(new URL("../app.js", import.meta.url), "utf8");
  const dom = new JSDOM(html, { url, runScripts: "outside-only" });
  const { window } = dom;
  window.Headers = Headers;
  window.crypto.randomUUID = randomUUID;
  window.HTMLDialogElement.prototype.showModal = function () { this.setAttribute("open", ""); };
  window.HTMLDialogElement.prototype.close = function () { this.removeAttribute("open"); };
  window.fetch = async () => ({ ok: true, status: 200, json: async () => [] });
  if(token) window.sessionStorage.setItem("knowflow_access_token",token);
  window.eval(source.replace(/^import[^\n]*\n/, "") + "\nglobalThis.harness = { state, signOut };");
  return { dom, window };
}

test("a link cannot sign in or overwrite an existing account token", async () => {
  for(const token of [null,"existing.login.session"]){
    const {dom,window}=await load("http://localhost:8080/#auth=header.payload.signature",token);
    try {
      assert.equal(window.sessionStorage.getItem("knowflow_access_token"),token);
      assert.equal(window.harness.state.token,token);
    } finally {dom.window.close();}
  }
});

test("a plain URL never creates a token", async () => {
  const { dom, window } = await load("http://localhost:8080/");
  try {
  assert.equal(window.sessionStorage.getItem("knowflow_access_token"), null);
  assert.equal(window.harness.state.token, null);
  assert.equal(window.document.querySelector("#auth-view").classList.contains("is-hidden"), false);
  } finally {dom.window.close();}
});

test("malformed or unrelated fragments are ignored instead of stored", async () => {
  for (const fragment of ["#auth=", "#auth=has%20space", "#auth=a/b", "#other=value", ""]) {
    const { dom, window } = await load(`http://localhost:8080/${fragment}`);
    try {
    assert.equal(
      window.sessionStorage.getItem("knowflow_access_token"), null,
      `fragment ${fragment || "(empty)"} must not be stored`,
    );
    } finally {dom.window.close();}
  }
});

test("sign-out clears a legitimate cached session even when a hostile fragment exists", async () => {
  const { dom, window } = await load("http://localhost:8080/#auth=other.account.token","legitimate.session");
  try {
  window.harness.signOut(false);
  assert.equal(window.sessionStorage.getItem("knowflow_access_token"), null);
  assert.equal(window.harness.state.token, null);
  } finally {dom.window.close();}
});
