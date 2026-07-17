"""Write a tamper-evident manifest of the artifact: SHA-256 + byte size of every
source file and every result JSON (the .llm_cache is summarised by entry count,
not hashed line-by-line, since it is large). Output: MANIFEST.json.

Run:  python make_manifest.py   (no API, no network)
Verify later:  python make_manifest.py --check
"""
from __future__ import annotations
import hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
INCLUDE_DIRS = ["lrq", "benchmark", "experiments"]
INCLUDE_TOP = ["README.md", "requirements.txt", "reproduce.sh", "make_manifest.py"]
SKIP = {"__pycache__", ".llm_cache", ".venv"}

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()

def walk():
    files = [os.path.join(HERE, f) for f in INCLUDE_TOP if os.path.exists(os.path.join(HERE, f))]
    for d in INCLUDE_DIRS:
        for root, dirs, names in os.walk(os.path.join(HERE, d)):
            dirs[:] = [x for x in dirs if x not in SKIP]
            for n in names:
                if n.endswith((".py", ".json", ".sh", ".md", ".txt")):
                    files.append(os.path.join(root, n))
    return sorted(files)

def build():
    entries = {}
    for p in walk():
        rel = os.path.relpath(p, HERE)
        entries[rel] = {"sha256": sha256(p), "bytes": os.path.getsize(p)}
    cache = os.path.join(HERE, ".llm_cache")
    n_cache = len(os.listdir(cache)) if os.path.isdir(cache) else 0
    return {"files": entries, "llm_cache_entries": n_cache, "n_files": len(entries)}

def main():
    out = os.path.join(HERE, "MANIFEST.json")
    if "--check" in sys.argv:
        old = json.load(open(out))["files"]
        cur = build()["files"]
        bad = [f for f in old if f not in cur or cur[f]["sha256"] != old[f]["sha256"]]
        added = [f for f in cur if f not in old]
        if bad or added:
            print("MISMATCH:", {"changed/missing": bad, "added": added}); sys.exit(1)
        print(f"OK: {len(cur)} files match MANIFEST.json"); return
    m = build()
    json.dump(m, open(out, "w"), indent=2, sort_keys=True)
    print(f"wrote MANIFEST.json: {m['n_files']} files, {m['llm_cache_entries']} cache entries")

if __name__ == "__main__":
    main()
