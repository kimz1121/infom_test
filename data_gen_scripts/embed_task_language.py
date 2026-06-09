"""Precompute Sentence-BERT (all-MiniLM-L6-v2, 384-d) embeddings for task names.

Run in an ISOLATED venv so the main environment's huggingface_hub/transformers
are untouched (sentence-transformers pulls newer versions). The output is a tiny
JSON lookup {task_name: [384 floats]} that the main pipeline reads with plain
numpy — no transformers dependency at train/inference time.

Task NAME (camelCase) -> phrase (e.g. PickPlaceCounterToCabinet ->
"pick place counter to cabinet") -> SBERT embedding.

Usage (isolated venv):
    /tmp/sbertvenv/bin/python data_gen_scripts/embed_task_language.py \
        --stats ~/.robocasa/data/atomic_65_multimodal_precompute_state_pretrain.stats.json \
        --extra LoadDishwasher PlaceVeggiesInDrawer StackBowlsCabinet StartElectricKettle \
        --out ~/.robocasa/data/task_lang_embeddings.json
"""
import argparse
import json
import os
import os.path as osp
import re


def prettify(name: str) -> str:
    """CamelCase task name -> lowercase space-separated phrase."""
    words = re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z][a-z]*|[0-9]+", name)
    return " ".join(w.lower() for w in words) if words else name.lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", default="", help="atomic stats.json (its task keys are embedded).")
    ap.add_argument("--extra", nargs="*", default=[], help="Extra task names (e.g. composites).")
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    names = []
    if args.stats:
        stats = json.load(open(osp.expanduser(args.stats)))
        names += list(stats["tasks"].keys())
    names += list(args.extra)
    # de-dup preserving order
    seen = set()
    names = [n for n in names if not (n in seen or seen.add(n))]

    phrases = {n: prettify(n) for n in names}
    print(f"Embedding {len(names)} task names with {args.model}")
    for n in names[:5]:
        print(f"  {n} -> '{phrases[n]}'")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model)
    embs = model.encode([phrases[n] for n in names], normalize_embeddings=True)
    dim = int(embs.shape[1])

    out = {"model": args.model, "dim": dim,
           "phrases": phrases,
           "embeddings": {n: embs[i].astype(float).tolist() for i, n in enumerate(names)}}
    os.makedirs(osp.dirname(osp.expanduser(args.out)), exist_ok=True)
    with open(osp.expanduser(args.out), "w") as f:
        json.dump(out, f)
    print(f"Saved {len(names)} x {dim}-d embeddings -> {args.out}")


if __name__ == "__main__":
    main()
