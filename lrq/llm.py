"""Multi-provider LLM wrapper with on-disk caching and token/latency accounting.

Dispatch by model id:
  claude-*            -> Anthropic   (API credits exhausted: cache hits only)
  gpt-*               -> OpenAI
  gemini-* / gemma-*  -> Google Gemini API
  qwen* llama* mistral* / ollama:*  -> local Ollama (OpenAI-compatible endpoint)

Cold runs read API keys lazily from environment variables
(ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY), falling back to a keys
directory (LRQ_KEYS_DIR, default ~/Desktop/keys); keys are never printed. Cached
reproduction needs no keys. Caching pins every response the first time it is
produced (key includes the model), so the study reproduces deterministically and
re-runs at zero cost from the in-repo .llm_cache/.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".llm_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

MODEL = os.environ.get("LRQ_MODEL", "claude-sonnet-4-6")
# Cold runs read API keys from environment variables first (portable/self-contained);
# an optional keys directory (override with LRQ_KEYS_DIR) is a fallback only.
_KEYS = os.environ.get("LRQ_KEYS_DIR", os.path.expanduser("~/Desktop/keys"))

USAGE = {"calls": 0, "cache_hits": 0, "in_tokens": 0, "out_tokens": 0}
_TL = threading.local()


def last_latency() -> float:
    return getattr(_TL, "last_latency", 0.0)


def usage_report() -> dict:
    return dict(USAGE)


def _key(model, system, prompt, temperature, max_tokens, sample) -> str:
    h = hashlib.sha256()
    h.update(json.dumps([model, system, prompt, temperature, max_tokens, sample]).encode())
    return h.hexdigest()


# ---- lazy clients ---------------------------------------------------------
_clients = {}


def _read_key(name: str, env: str) -> str:
    v = os.environ.get(env)                       # portable: env var wins
    if v:
        return v.strip()
    return open(os.path.join(_KEYS, name)).read().strip()   # optional fallback


def _anthropic():
    if "anthropic" not in _clients:
        import anthropic
        _clients["anthropic"] = anthropic.Anthropic()
    return _clients["anthropic"]


def _openai():
    if "openai" not in _clients:
        import openai
        _clients["openai"] = openai.OpenAI(api_key=_read_key("openai.md", "OPENAI_API_KEY"))
    return _clients["openai"]


def _gemini():
    if "gemini" not in _clients:
        from google import genai
        _clients["gemini"] = genai.Client(api_key=_read_key("gemini.md", "GEMINI_API_KEY"))
    return _clients["gemini"]


def _ollama():
    if "ollama" not in _clients:
        import openai
        _clients["ollama"] = openai.OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    return _clients["ollama"]


def provider_of(model: str) -> str:
    m = model.lower()
    # Ollama models use tag-style names ("gemma2:9b", "qwen2.5:14b"); the colon
    # disambiguates a local Gemma from Gemma served by the Gemini API.
    if ":" in m or m.startswith("ollama"):
        return "ollama"
    if m.startswith("claude"):
        return "anthropic"
    if m.startswith("gpt") or m.startswith("o1") or m.startswith("o3"):
        return "openai"
    if m.startswith("gemini") or m.startswith("gemma"):
        return "gemini"
    return "ollama"


# ---- provider calls: return (text, in_tokens, out_tokens) -----------------
def _call_anthropic(model, system, prompt, temperature, max_tokens):
    r = _anthropic().messages.create(
        model=model, max_tokens=max_tokens, temperature=temperature,
        system=system, messages=[{"role": "user", "content": prompt}])
    text = "".join(b.text for b in r.content if b.type == "text")
    return text, r.usage.input_tokens, r.usage.output_tokens


def _call_openai(model, system, prompt, temperature, max_tokens):
    r = _openai().chat.completions.create(
        model=model, temperature=temperature, max_tokens=max_tokens,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}])
    u = r.usage
    return r.choices[0].message.content or "", (u.prompt_tokens if u else 0), (u.completion_tokens if u else 0)


def _call_gemini(model, system, prompt, temperature, max_tokens):
    from google.genai import types
    cfg = types.GenerateContentConfig(temperature=temperature, max_output_tokens=max_tokens,
                                      system_instruction=system)
    r = _gemini().models.generate_content(model=model, contents=prompt, config=cfg)
    text = r.text or ""
    um = getattr(r, "usage_metadata", None)
    it = getattr(um, "prompt_token_count", 0) or 0
    ot = getattr(um, "candidates_token_count", 0) or 0
    return text, it, ot


def _call_ollama(model, system, prompt, temperature, max_tokens):
    name = model.split("ollama:", 1)[1] if model.startswith("ollama:") else model
    r = _ollama().chat.completions.create(
        model=name, temperature=temperature, max_tokens=max_tokens,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}])
    u = r.usage
    return r.choices[0].message.content or "", (u.prompt_tokens if u else 0), (u.completion_tokens if u else 0)


_DISPATCH = {"anthropic": _call_anthropic, "openai": _call_openai,
             "gemini": _call_gemini, "ollama": _call_ollama}


# Reasoning models spend output budget on hidden thoughts; give them headroom so
# the code is not truncated. Keyed on the effective budget -> a fresh, valid entry.
_THINKING = {"gemini-2.5-pro", "gemini-2.5-flash", "o1", "o3", "o1-mini", "o3-mini"}


def complete(system: str, prompt: str, temperature: float = 0.2,
             max_tokens: int = 1500, sample: int = 0, model: str | None = None) -> str:
    model = model or MODEL
    if model in _THINKING:
        max_tokens = max(max_tokens, 8000)
    key = _key(model, system, prompt, temperature, max_tokens, sample)
    path = os.path.join(CACHE_DIR, key + ".json")
    if os.path.exists(path):
        rec = json.load(open(path))
        USAGE["cache_hits"] += 1
        _TL.last_latency = rec.get("latency", 0.0)
        return rec["text"]

    provider = provider_of(model)
    fn = _DISPATCH[provider]
    last_err = None
    for attempt in range(8):
        try:
            t0 = time.time()
            text, it, ot = fn(model, system, prompt, temperature, max_tokens)
            lat = time.time() - t0
            USAGE["calls"] += 1
            USAGE["in_tokens"] += it
            USAGE["out_tokens"] += ot
            _TL.last_latency = lat
            json.dump({"text": text, "in": it, "out": ot, "model": model, "latency": lat},
                      open(path, "w"))
            return text
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            transient = any(s in msg for s in ("503", "429", "overload", "unavailable", "high demand", "rate", "timeout", "500"))
            time.sleep(min(30.0, (5.0 if transient else 2.0) * (attempt + 1)))
    raise RuntimeError(f"LLM call failed ({provider}/{model}) after retries: {last_err}")
