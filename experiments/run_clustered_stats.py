"""Reviewer Q1: pooled significance without assuming instance independence.
Regenerates the program-clustered bootstrap and the mixed-effects logistic for
pooled B2-B0 over the 11 models. Run in the project venv (needs statsmodels)."""
import json, os, sys, random, math
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
R=os.path.join(HERE,"results")
frozen={r["id"]:r for r in json.load(open(f"{R}/records_FROZEN.json")) if r["kind"]!="clean"}
extra=json.load(open(f"{R}/models_extra.json")); ids=sorted(frozen)
M=[("claude-sonnet-4-6","f"),("claude-haiku-4-5-20251001","f"),("gpt-4o","e"),("gpt-4o-mini","e"),
   ("gemini-2.5-pro","e"),("gemini-2.5-flash","e"),("gemini-2.5-flash-lite","e"),("gemma2:9b","e"),
   ("qwen2.5:7b","e"),("llama3.1:8b","e"),("mistral:7b","e"),
   ("llama3.2:3b","e"),("qwen2.5-coder:7b","e"),("qwen2.5:14b","e"),("command-r:35b","e"),
   ("qwen2.5-coder:32b","e"),("llama3.3:70b","e"),("qwen2.5:72b","e")]
def sem(mk,s,cfg,i):
    if s=="f": return 1 if frozen[i]["repair"][mk][cfg]["pass1"]["sem"] else 0
    rec={r["id"]:r for r in extra[mk]["records"]}; return 1 if (cfg in rec[i] and rec[i][cfg]["sem"]) else 0
diffs={i:[sem(mk,s,"B2",i)-sem(mk,s,"B0",i) for mk,s in M] for i in ids}
random.seed(12345)
pm=lambda P:sum(d for p in P for d in diffs[p])/(len(P)*len(M))
boot=sorted(pm([random.choice(ids) for _ in ids]) for _ in range(5000))
print(f"program-clustered bootstrap: {pm(ids):+.3f} CI[{boot[125]:+.3f},{boot[4875]:+.3f}]")
try:
    import pandas as pd; from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    df=pd.DataFrame([{"y":sem(mk,s,c,i),"grounded":int(c=="B2"),"prog":i,"model":mk}
                     for mk,s in M for i in ids for c in ["B0","B2"]])
    res=BinomialBayesMixedGLM.from_formula("y ~ grounded",{"prog":"0+C(prog)","model":"0+C(model)"},df).fit_vb()
    b=res.fe_mean[list(res.model.exog_names).index("grounded")]; sd=res.fe_sd[list(res.model.exog_names).index("grounded")]
    print(f"mixed-effects logistic: grounded OR={math.exp(b):.2f} 95%CI[{math.exp(b-1.96*sd):.2f},{math.exp(b+1.96*sd):.2f}]")
except Exception as e: print("mixed-effects skipped:", str(e)[:60])
