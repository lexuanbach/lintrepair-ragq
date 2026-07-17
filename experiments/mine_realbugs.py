"""Overnight pipeline: mine REAL (non-synthetic) quantum-program bug/fix pairs from
the wild and execution-validate them. Sources:
  (1) Stack Overflow [qiskit] Q&A -- buggy code in the question, fix in the
      accepted answer (self-contained reproducers by nature).
  (2) GitHub bug-fix commits in downstream qiskit-using repos -- single-file .py
      changes; before=buggy, after=fixed.
Each candidate is run under BOTH Qiskit 2.4.2 and legacy 0.45 (to beat API drift);
a pair is kept as a validated real bug when the fixed version runs and the buggy
version either crashes (crash bug) or runs-but-diverges (silent bug).

Robust for unattended runs: per-candidate try/except, incremental checkpoint after
every candidate (atomic rename), resumable (skips seen IDs), rate-limit handling,
and a wall-clock budget. Writes results/realbugs_mined.json + a log line stream.
"""
from __future__ import annotations
import json, os, re, sys, time, html, subprocess, tempfile, traceback
import requests

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "results", "realbugs_mined.json")
VENVS = [("qiskit2.4.2", os.path.join(ROOT, ".venv", "bin", "python")),
         # Legacy Qiskit 0.45 interpreter (external; override with BUGS4Q_LEGACY_PY).
         ("qiskit0.45", os.environ.get("BUGS4Q_LEGACY_PY", "/tmp/bugs4q-venv/bin/python"))]
WALL_BUDGET = int(os.environ.get("MINE_WALL", 8 * 3600))   # stop after ~8h (override for smoke test)
RUN_TIMEOUT = int(os.environ.get("MINE_TIMEOUT", 30))
SO = "https://api.stackexchange.com/2.3"
T0 = time.time()


def log(*a):
    print(f"[{int(time.time()-T0):6d}s]", *a, flush=True)


# ---------- checkpoint ----------
def load_ckpt():
    if os.path.exists(OUT):
        try:
            return json.load(open(OUT))
        except Exception:
            pass
    return {"validated": [], "seen": [], "stats": {"candidates": 0, "so": 0, "github": 0,
            "crash": 0, "silent": 0, "nonrepro": 0, "equiv": 0, "error": 0}}


def save_ckpt(ck):
    tmp = OUT + ".tmp"
    json.dump(ck, open(tmp, "w"), indent=2)
    os.replace(tmp, OUT)


# ---------- code execution / oracle ----------
def run_code(code, python_exe, timeout=RUN_TIMEOUT):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(code); path = fh.name
    try:
        env = dict(os.environ); env["OMP_NUM_THREADS"] = "1"
        p = subprocess.run([python_exe, path], capture_output=True, text=True,
                           timeout=timeout, env=env)
        os.unlink(path)
        return p.returncode, p.stdout.strip(), (p.stderr.strip().splitlines()[-1] if p.stderr.strip() else "")
    except subprocess.TimeoutExpired:
        try: os.unlink(path)
        except Exception: pass
        return -1, "", "timeout"
    except Exception as e:
        try: os.unlink(path)
        except Exception: pass
        return -2, "", f"{type(e).__name__}: {str(e)[:80]}"


def classify(buggy, fixed):
    """Return (verdict, env, fixed_stdout) over the two envs. verdict in
    {crash, silent, equiv, nonrepro}."""
    for env_name, py in VENVS:
        if not os.path.exists(py):
            continue
        frc, fout, ferr = run_code(fixed, py)
        if frc != 0:
            continue                                   # fixed doesn't run here; try next env
        fout2 = run_code(fixed, py)[1]                 # determinism check
        brc, bout, berr = run_code(buggy, py)
        if brc != 0:
            return "crash", env_name, fout
        if fout != fout2:
            continue                                   # fixed non-deterministic -> skip silent here
        if bout != fout:
            return "silent", env_name, fout
        return "equiv", env_name, fout                 # buggy == fixed -> not a real bug
    return "nonrepro", None, None


# ---------- code extraction ----------
SDKS = ("qiskit", "cirq", "pennylane")


def code_blocks(html_body):
    blocks = re.findall(r"<pre[^>]*><code>(.*?)</code></pre>", html_body or "", re.DOTALL)
    blocks += re.findall(r"<code>(.*?)</code>", html_body or "", re.DOTALL)
    out = []
    for b in blocks:
        t = html.unescape(re.sub(r"<[^>]+>", "", b)).strip()
        if any(s in t for s in SDKS) and ("import" in t) and t.count("\n") >= 2:
            out.append(t)
    return out


def largest_qiskit_block(body):
    bs = code_blocks(body)
    return max(bs, key=len) if bs else None


# ---------- Stack Overflow source ----------
def so_get(path, params):
    for attempt in range(6):
        try:
            r = requests.get(f"{SO}/{path}", params={**params, "site": "stackoverflow"}, timeout=30)
            d = r.json()
            if d.get("backoff"):
                log("SO backoff", d["backoff"]); time.sleep(d["backoff"] + 1)
            return d
        except Exception as e:
            log("SO err", str(e)[:60]); time.sleep(5 * (attempt + 1))
    return {}


SO_TAGS = ["qiskit", "pennylane", "cirq"]   # broaden the net (incl. cross-SDK)


def so_candidates(seen):
    for tag in SO_TAGS:
        yield from _so_tag(seen, tag)


def _so_tag(seen, tag):
    page = 1
    while True:
        d = so_get("questions", {"tagged": tag, "sort": "creation", "order": "desc",
                                 "pagesize": 100, "page": page, "filter": "withbody"})
        items = d.get("items", [])
        if not items:
            break
        # collect accepted-answer ids for this page
        acc = {q["accepted_answer_id"]: q for q in items if q.get("accepted_answer_id")}
        if acc:
            ids = ";".join(str(i) for i in list(acc)[:100])
            ad = so_get(f"answers/{ids}", {"filter": "withbody"})
            ans = {a["answer_id"]: a for a in ad.get("items", [])}
            for aid, q in acc.items():
                qid = f"so:{q['question_id']}"
                if qid in seen:
                    continue
                a = ans.get(aid)
                if not a:
                    continue
                buggy = largest_qiskit_block(q.get("body"))
                fixed = largest_qiskit_block(a.get("body"))
                if buggy and fixed and buggy != fixed and 2 <= buggy.count("\n") <= 80:
                    yield {"id": qid, "source": "stackoverflow", "url": q.get("link"),
                           "title": q.get("title", "")[:120], "buggy": buggy, "fixed": fixed}
        if not d.get("has_more"):
            break
        page += 1
        if d.get("quota_remaining", 1) <= 2:
            log("SO quota exhausted"); break


# ---------- GitHub source ----------
def gh(path):
    try:
        p = subprocess.run(["gh", "api", path], capture_output=True, text=True, timeout=40)
        return json.loads(p.stdout) if p.returncode == 0 else None
    except Exception:
        return None


def gh_candidates(seen, max_repos=int(os.environ.get("MINE_REPOS", 400))):
    """Repo-first: independent (non-qiskit-org) qiskit-using repos -> recent
    fix/bug commits -> single-.py before/after. Execution-validation later filters
    out non-self-contained files (they ImportError -> nonrepro)."""
    import base64
    def decode(c):
        try: return base64.b64decode(c["content"]).decode("utf-8", "ignore")
        except Exception: return None
    seen_repos = 0; page = 1
    while seen_repos < max_repos:
        q = "qiskit+language:Python+-org:Qiskit+-org:qiskit-community+-org:PennyLaneAI"
        rd = gh(f"search/repositories?q={q}&sort=updated&order=desc&per_page=30&page={page}")
        repos = (rd or {}).get("items", [])
        if not repos:
            break
        for r in repos:
            if seen_repos >= max_repos:
                break
            seen_repos += 1; repo = r["full_name"]
            commits = gh(f"repos/{repo}/commits?per_page=30") or []
            for c in commits:
                sha = c.get("sha"); msg = c.get("commit", {}).get("message", "")
                if not sha or not any(w in msg.lower() for w in ("fix", "bug", "correct")):
                    continue
                cid = f"gh:{repo}@{sha[:10]}"
                if cid in seen:
                    continue
                detail = gh(f"repos/{repo}/commits/{sha}")
                if not detail:
                    continue
                pyf = [f for f in detail.get("files", []) if f.get("filename", "").endswith(".py")
                       and f.get("changes", 999) <= 60]      # small patches only
                parents = detail.get("parents", [])
                if not pyf or not parents:
                    continue
                for k, pf in enumerate(pyf[:2]):              # up to 2 changed .py per commit
                    fn = pf["filename"]
                    after = gh(f"repos/{repo}/contents/{fn}?ref={sha}")
                    before = gh(f"repos/{repo}/contents/{fn}?ref={parents[0]['sha']}")
                    fixed = decode(after) if after else None
                    buggy = decode(before) if before else None
                    if (buggy and fixed and buggy != fixed and "qiskit" in fixed
                            and 3 <= fixed.count("\n") <= 80):
                        yield {"id": f"{cid}#{k}", "source": "github", "url": c.get("html_url"),
                               "title": msg.splitlines()[0][:120], "buggy": buggy, "fixed": fixed}
        page += 1
        if page > 10:
            break


def main():
    ck = load_ckpt(); seen = set(ck["seen"])
    log(f"resume: {len(ck['validated'])} validated, {len(seen)} seen")
    sources = []
    try: sources.append(("so", so_candidates(seen)))
    except Exception: pass
    try: sources.append(("github", gh_candidates(seen)))
    except Exception: pass
    for sname, gen in sources:
        for cand in gen:
            if time.time() - T0 > WALL_BUDGET:
                log("wall budget reached"); save_ckpt(ck); return
            if cand["id"] in seen:
                continue
            seen.add(cand["id"]); ck["seen"].append(cand["id"]); ck["stats"]["candidates"] += 1
            ck["stats"][sname] = ck["stats"].get(sname, 0) + 1
            try:
                verdict, env, fixed_out = classify(cand["buggy"], cand["fixed"])
                ck["stats"][verdict if verdict in ck["stats"] else "error"] = \
                    ck["stats"].get(verdict, 0) + 1
                if verdict in ("crash", "silent"):
                    cand.update(verdict=verdict, env=env, reference_stdout=fixed_out)
                    ck["validated"].append(cand)
                    log(f"KEEP [{verdict}/{env}] {cand['source']} {cand['title'][:60]}")
            except Exception:
                ck["stats"]["error"] += 1
                log("classify error", traceback.format_exc().splitlines()[-1][:80])
            if ck["stats"]["candidates"] % 5 == 0:
                save_ckpt(ck)
                log(f"progress: {ck['stats']['candidates']} cand, "
                    f"{len(ck['validated'])} validated (crash {ck['stats']['crash']}, "
                    f"silent {ck['stats']['silent']}, nonrepro {ck['stats']['nonrepro']})")
    save_ckpt(ck)
    log("DONE", ck["stats"])


if __name__ == "__main__":
    main()
