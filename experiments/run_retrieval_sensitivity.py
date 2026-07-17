"""Reviewer Q1: retrieval-budget and retriever-strategy sensitivity. Does a smaller
retrieval budget (or a different retriever) reduce grounding-context interference for
strong models while preserving the weak-model rescue? We re-run B2 (naive grounding +
validator) over the 32 defects for a STRONG model (Sonnet, the most anchored) and a
WEAK model (GPT-4o-mini, the most rescued), varying k in {2,4,6} with the default
TF-IDF retriever, plus a BM25 retriever at k=4. Writes results/retrieval_sensitivity.json.
"""
from __future__ import annotations
import json, os, sys, math, re
from collections import Counter
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT)
import lrq.repair as R
from lrq.detectors import detect, Finding
from lrq.harness import run_script, results_match
from lrq.retrieval import KB_DOCS, Retrieved
from lrq.llm import usage_report

BENCH = [b for b in json.load(open(os.path.join(ROOT, "benchmark", "benchmark.json"))) if b["kind"] != "clean"]
FROZEN = {r["id"]: r for r in json.load(open(os.path.join(HERE, "results", "records_FROZEN.json")))}
_WORD = re.compile(r"[A-Za-z_]+")


def findings(inst):
    st = detect(inst["buggy"]); cats = {f.category for f in st}; out = list(st)
    for c in FROZEN[inst["id"]]["detection"]["union"]:
        if c not in cats: out.append(Finding(c, 0, "(LLM)"))
    return out


class BM25Retriever:
    """Minimal BM25 over the same KB, same .retrieve(query,k)->[Retrieved] interface."""
    def __init__(self, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = KB_DOCS
        self.toks = [[w.lower() for w in _WORD.findall(d["text"])] for d in self.docs]
        self.dl = [len(t) for t in self.toks]; self.avgdl = sum(self.dl) / len(self.dl)
        df = Counter()
        for t in self.toks:
            for w in set(t): df[w] += 1
        N = len(self.docs)
        self.idf = {w: math.log(1 + (N - n + 0.5) / (n + 0.5)) for w, n in df.items()}
        self.tf = [Counter(t) for t in self.toks]

    def retrieve(self, query, k=4):
        q = [w.lower() for w in _WORD.findall(query)]
        scores = []
        for i, d in enumerate(self.docs):
            s = 0.0
            for w in q:
                if w in self.tf[i]:
                    f = self.tf[i][w]
                    s += self.idf.get(w, 0) * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl))
            scores.append(s)
        order = sorted(range(len(self.docs)), key=lambda i: -scores[i])[:k]
        return [Retrieved(self.docs[i]["id"], self.docs[i]["tag"],
                          " ".join(self.docs[i]["text"].split()), round(scores[i], 3)) for i in order]


def score_b2(inst, model, k):
    f = findings(inst); ref = inst["reference_result"]
    res = R.full_repair(inst["buggy"], f, run_script, k=k, max_iters=3, debias=False, model=model)
    o = run_script(res.code)
    return bool(results_match(ref, o.result)[0]) if o.status == "ok" else False


def run_config(model, retriever, k, label):
    R._RETRIEVER = retriever
    with ThreadPoolExecutor(max_workers=4) as ex:
        recs = list(ex.map(lambda b: score_b2(b, model, k), BENCH))
    return {"label": label, "model": model, "k": k, "pass1": sum(recs), "n": len(recs)}


def main():
    from lrq.retrieval import TfidfRetriever
    tfidf, bm25 = TfidfRetriever(), BM25Retriever()
    MODELS = ["claude-sonnet-4-6", "gpt-4o-mini"]
    out = []
    for model in MODELS:
        for k in (2, 4, 6):
            r = run_config(model, tfidf, k, f"tfidf-k{k}")
            out.append(r); print(f"  {model:22s} tfidf k={k}: B2 pass@1 = {r['pass1']}/{r['n']}", flush=True)
        r = run_config(model, bm25, 4, "bm25-k4")
        out.append(r); print(f"  {model:22s} bm25  k=4: B2 pass@1 = {r['pass1']}/{r['n']}", flush=True)
        print("    usage:", usage_report(), flush=True)
    json.dump(out, open(os.path.join(HERE, "results", "retrieval_sensitivity.json"), "w"), indent=2)


if __name__ == "__main__":
    print("Retrieval-budget / strategy sensitivity (Q1)\n")
    main()
