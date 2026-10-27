# Third-party materials

The MIT license applies to original KnowFlow RAG code and authored synthetic
controls. It does not relicense upstream libraries, document sources or models.

- Bundled tokenizer: unmodified `tokenizer.json` from
  [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B),
  revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` (gzip packaging only).
  Apache-2.0; upstream license, attribution and checksum are retained in
  `backend/knowflow/app/assets/`. No model weights are included. It is a budget
  estimator, not proof of equivalence to a hosted embedding tokenizer.
- Frontend dependencies retain their package licenses. `pnpm-lock.yaml` pins
  source packages; esbuild external legal comments accompany generated chunks.
  Source releases build these assets rather than silently omitting legal files.
- Python dependencies are pinned by `uv.lock` and retain their own licenses.
- Optional Docling runs as a separate pinned container, without bundled weights
  in this source archive. Its upstream image and model licenses apply.
- Noto CJK fonts installed in the API container retain their upstream font
  licensing. They are not copied into this source archive.
- Financial reports, public-paper PDFs, downloaded web documents, original user
  uploads, previous answers and research logs are excluded from source releases.
  Fetching a public URL does not grant redistribution rights. Only explicitly
  authored synthetic documents are bundled as test controls.
