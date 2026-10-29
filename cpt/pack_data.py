"""Tokenize raw text and pack it into a flat uint32 token file for continued pretraining.

Each document is tokenized, an EOS token is appended, and all tokens are written
back to back into ``<out>.bin``. Training samples fixed-length windows from it, so
documents are packed with no padding. A ``<out>.json`` sidecar records metadata.

Supported inputs (files or directories, searched recursively):
  *.jsonl / *.json  one JSON object per line, text in --text-key (default "text")
  *.parquet         text column --text-key
  *.txt             one document per file

Example:
  python cpt/pack_data.py --tokenizer /models/Qwen-32B-base \
      --inputs /data/domain --out /data/packed/domain --workers 64
"""

import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np

_tok = None
_text_key = "text"


def _init_worker(tokenizer_path, text_key):
    global _tok, _text_key
    from transformers import AutoTokenizer

    _tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    _text_key = text_key


def _iter_docs(path):
    suffix = path.suffix.lower()
    if suffix in (".jsonl", ".json"):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    text = json.loads(line).get(_text_key)
                    if text:
                        yield text
    elif suffix == ".parquet":
        import pyarrow.parquet as pq

        for batch in pq.ParquetFile(path).iter_batches(columns=[_text_key]):
            for text in batch.column(0).to_pylist():
                if text:
                    yield text
    elif suffix == ".txt":
        text = path.read_text(encoding="utf-8", errors="ignore")
        if text.strip():
            yield text


def _tokenize_file(path):
    eos = _tok.eos_token_id
    chunks, batch, n_docs = [], [], 0

    def flush():
        for ids in _tok(batch, add_special_tokens=False)["input_ids"]:
            ids.append(eos)
            chunks.append(np.asarray(ids, dtype=np.uint32))
        batch.clear()

    for text in _iter_docs(Path(path)):
        batch.append(text)
        n_docs += 1
        if len(batch) >= 256:
            flush()
    if batch:
        flush()
    tokens = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.uint32)
    return str(path), n_docs, tokens


def _collect(inputs):
    exts = {".jsonl", ".json", ".parquet", ".txt"}
    files = []
    for p in map(Path, inputs):
        if p.is_dir():
            files += sorted(f for f in p.rglob("*") if f.suffix.lower() in exts)
        else:
            files.append(p)
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", required=True, help="path to the base model or its tokenizer")
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True, help="output prefix; writes <out>.bin and <out>.json")
    ap.add_argument("--text-key", default="text")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    files = _collect(args.inputs)
    if not files:
        raise SystemExit("no input files found")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    total_tokens = total_docs = 0
    with open(args.out + ".bin", "wb") as fout, Pool(
        args.workers, initializer=_init_worker, initargs=(args.tokenizer, args.text_key)
    ) as pool:
        for i, (path, n_docs, tokens) in enumerate(pool.imap(_tokenize_file, files), 1):
            tokens.tofile(fout)
            total_tokens += tokens.size
            total_docs += n_docs
            print(f"[{i}/{len(files)}] {path}: {n_docs} docs, {tokens.size:,} tokens "
                  f"(total {total_tokens:,})", flush=True)

    meta = {
        "dtype": "uint32",
        "num_tokens": total_tokens,
        "num_docs": total_docs,
        "tokenizer": args.tokenizer,
        "sources": [str(f) for f in files],
    }
    with open(args.out + ".json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"done: {total_docs:,} docs, {total_tokens:,} tokens -> {args.out}.bin")


if __name__ == "__main__":
    main()
