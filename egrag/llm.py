"""LLM access through OpenRouter (one API key, many models).

Every request and response is stored in data/cache/<hash>.json. This gives two things the
methodology needs: a log of every prompt, response and model version, and repeatable runs
(the same prompt to the same model is answered from the cache instead of the API).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

from . import ROOT, load_config

OPENROUTER_URL = "https://openrouter.ai/api/v1"


def parse_json(text: str) -> dict:
    """Models sometimes wrap JSON in ``` fences or add a sentence; keep only the object."""
    text = re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.M).strip()
    for candidate in (text, (re.search(r"\{.*\}", text, flags=re.S) or [None])[0]):
        if candidate:
            try:
                data = json.loads(candidate)
                return data if isinstance(data, dict) else {"answer": str(data)}
            except json.JSONDecodeError:
                continue
    return {}


class LLM:
    def __init__(self, model: str, cfg: dict | None = None, cache_dir: str | Path | None = None):
        cfg = cfg or load_config()
        self.model = model
        self.temperature = cfg["models"]["temperature"]
        self.max_tokens = cfg["models"]["max_tokens"]
        self.cache = Path(cache_dir or ROOT / cfg["paths"]["cache"])
        self.cache.mkdir(parents=True, exist_ok=True)
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from dotenv import load_dotenv
            from openai import OpenAI
            load_dotenv(ROOT / ".env")
            key = os.getenv("OPENROUTER_API_KEY")
            if not key:
                raise RuntimeError("OPENROUTER_API_KEY is missing: copy .env.example to .env and add your key.")
            self._client = OpenAI(base_url=os.getenv("OPENROUTER_BASE_URL", OPENROUTER_URL), api_key=key)
        return self._client

    def chat(self, system: str, user: str, json_mode: bool = True) -> dict:
        """Return {"text", "json", "model", "latency_s", "usage", "cached"}."""
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        key = hashlib.sha256(json.dumps([self.model, self.temperature, messages, json_mode]).encode()).hexdigest()
        path = self.cache / f"{key}.json"
        if path.exists():
            out = json.loads(path.read_text(encoding="utf-8"))
            out["cached"] = True
            return out
        extra = {"response_format": {"type": "json_object"}} if json_mode else {}
        for attempt in range(4):
            try:
                start = time.time()
                resp = self.client.chat.completions.create(
                    model=self.model, messages=messages, temperature=self.temperature,
                    max_tokens=self.max_tokens, **extra)
                if not getattr(resp, "choices", None):
                    raise RuntimeError(f"empty response: {getattr(resp, 'error', resp)}")
                text = resp.choices[0].message.content or ""
                break
            except Exception as err:            # rate limits / timeouts: back off and retry
                if "401" in str(err) or "403" in str(err) or "404" in str(err):
                    raise RuntimeError(f"OpenRouter refused the request for {self.model}: {err}") from err
                if attempt == 3:
                    raise RuntimeError(f"OpenRouter did not answer after 4 tries ({self.model}): {err}") from err
                print(f"[llm] {self.model}: {err} - retrying", file=sys.stderr)
                time.sleep(2 ** attempt * 3)
        out = {"text": text, "json": parse_json(text) if json_mode else None,
               "model": getattr(resp, "model", self.model), "latency_s": round(time.time() - start, 2),
               "usage": resp.usage.model_dump() if resp.usage else None,
               "request": messages, "cached": False}
        path.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
        return out


def check_models(cfg: dict | None = None) -> None:
    """python -m egrag.llm --check : confirm the configured model ids exist on OpenRouter."""
    import urllib.request
    cfg = cfg or load_config()
    with urllib.request.urlopen(f"{OPENROUTER_URL}/models", timeout=30) as r:
        available = {m["id"] for m in json.load(r)["data"]}
    wanted = list(cfg["models"]["generators"].values()) + [cfg["models"]["judge"]]
    for m in wanted:
        print(("OK      " if m in available else "MISSING ") + m)


if __name__ == "__main__" and "--check" in sys.argv:
    check_models()
