# Offline tokenizer asset

`qwen3-tokenizer.json.gz` contains the unmodified `tokenizer.json` from
[Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B),
revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`, compressed with gzip.
Copyright and attribution belong to the upstream Qwen authors. See the included
`QWEN-LICENSE.txt` (Apache License 2.0). The adjacent JSON records its original URL
and the SHA-256 of the uncompressed bytes. No model weights are included.

It estimates token budgets for the hosted embedding configuration. Exact tokenizer
equivalence to `qwen3.7-text-embedding` has **not** been established. Input counts
include headings and repeated caption context, and reserve 16 tokens by default.
Neither this estimate nor the reserve proves the hosted model's exact token count.

Reproduce with `python evaluation/complex_documents/fetch_chunk_tokenizer.py`.
The application validates the bundled checksum and never fetches a tokenizer at
runtime. `CHUNK_TOKENIZER_PATH` can select a local tokenizer JSON; changing it
requires an application restart to clear the process cache.
