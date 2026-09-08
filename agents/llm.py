"""LLM access layer.

Two things live here:

1. A thin multi-provider client (Groq / Gemini / OpenAI / Anthropic) with
   strict-JSON parsing and repair.

2. An `offline` provider that is NOT an LLM at all. It is a deterministic,
   rule-based implementation of the same three agent interfaces.

The offline backend has a real scientific purpose beyond portability: it is
the rules-only control condition in the ablation study. Comparing
`offline` against a real LLM on the same benchmark answers "does the language
model actually contribute, or is the metadata heuristic doing all the work?"
That is a question a reviewer will ask, and most student projects cannot
answer it because they never built the control.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import Any

import config


class LLMError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# JSON extraction
# --------------------------------------------------------------------------

def extract_json(text: str) -> Any:
    """Pull the first JSON object/array out of a model response.

    Models wrap JSON in prose or fences no matter how firmly you ask them not
    to, so this tries progressively more forgiving strategies rather than
    failing on the first parse error.
    """
    if text is None:
        raise LLMError("Empty LLM response")

    text = text.strip()
    if not text:
        raise LLMError("Empty LLM response")

    # Strip markdown fences.
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find the outermost balanced {...} or [...] block.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        # Trailing commas are the usual culprit.
                        cleaned = re.sub(r",\s*([}\]])", r"\1", candidate)
                        try:
                            return json.loads(cleaned)
                        except json.JSONDecodeError:
                            break

    raise LLMError(f"Could not parse JSON from response: {text[:250]}")


# --------------------------------------------------------------------------
# Provider client
# --------------------------------------------------------------------------

class LLMClient:
    def __init__(self, provider: str | None = None, model: str | None = None):
        self.provider = (provider or config.LLM_PROVIDER).lower()
        self.model = model or config.LLM_MODELS.get(
            self.provider, "offline-deterministic"
        )
        self.call_count = 0
        self.total_latency = 0.0
        # Provider failures are swallowed by every agent so one bad response
        # cannot kill a 32-question run. That resilience hid a total outage:
        # a run in which EVERY call failed still produced a clean 1.000 table,
        # because each agent quietly fell back to its deterministic path.
        # Counting failures here, and surfacing the first one immediately, is
        # what makes "the model never answered" visible instead of invisible.
        self.error_count = 0
        self.last_error: str | None = None
        self._announced_error = False

        if self.provider != "offline":
            key_env = config.API_KEY_ENV.get(self.provider)
            if not key_env:
                raise LLMError(f"Unknown provider: {self.provider}")
            self.api_key = os.getenv(key_env)
            if not self.api_key:
                raise LLMError(
                    f"{key_env} is not set. Export it, or set "
                    f"VERIRAG_LLM_PROVIDER=offline to run without an API key."
                )

    @property
    def is_offline(self) -> bool:
        return self.provider == "offline"

    def complete(self, system: str, user: str) -> str:
        """Send a single-turn request and return raw text."""
        if self.is_offline:
            raise LLMError(
                "Offline provider has no generic completion; agents implement "
                "their own deterministic logic."
            )

        start = time.time()
        try:
            text = self._dispatch(system, user)
        except Exception as exc:
            self.error_count += 1
            self.last_error = str(exc)[:300]
            if not self._announced_error:
                # Print once, not per call: a dead API key would otherwise
                # emit hundreds of identical lines and bury the run output.
                self._announced_error = True
                print(
                    f"\n[!] {self.provider} call failed: {self.last_error}\n"
                    f"[!] Agents will fall back to deterministic rules. "
                    f"Further errors are counted, not printed.\n",
                    file=sys.stderr,
                )
            raise
        finally:
            self.total_latency += time.time() - start
            self.call_count += 1
        return text

    def complete_json(self, system: str, user: str, retries: int = 2) -> Any:
        """Complete and parse as JSON, retrying with a stricter nudge."""
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                suffix = (
                    ""
                    if attempt == 0
                    else "\n\nYour previous reply was not valid JSON. "
                    "Reply with ONLY a JSON value. No prose, no code fences."
                )
                return extract_json(self.complete(system, user + suffix))
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                time.sleep(0.4 * (attempt + 1))
        raise LLMError(f"JSON completion failed after retries: {last_error}")

    # -- provider implementations ------------------------------------------

    def _dispatch(self, system: str, user: str) -> str:
        if self.provider == "groq":
            return self._openai_compatible(
                system, user, "https://api.groq.com/openai/v1/chat/completions"
            )
        if self.provider == "openai":
            return self._openai_compatible(
                system, user, "https://api.openai.com/v1/chat/completions"
            )
        if self.provider == "gemini":
            return self._gemini(system, user)
        if self.provider == "anthropic":
            return self._anthropic(system, user)
        raise LLMError(f"Unsupported provider: {self.provider}")

    def _post(self, url: str, headers: dict, payload: dict) -> dict:
        import urllib.error
        import urllib.request

        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url, data=data, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(
                request, timeout=config.LLM_TIMEOUT
            ) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:400]
            raise LLMError(f"HTTP {exc.code} from {self.provider}: {body}") from exc
        except urllib.error.URLError as exc:
            raise LLMError(f"Network error calling {self.provider}: {exc}") from exc

    def _openai_compatible(self, system: str, user: str, url: str) -> str:
        payload = {
            "model": self.model,
            "temperature": config.LLM_TEMPERATURE,
            "max_tokens": config.LLM_MAX_TOKENS,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        result = self._post(url, headers, payload)
        return result["choices"][0]["message"]["content"]

    def _gemini(self, system: str, user: str) -> str:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "temperature": config.LLM_TEMPERATURE,
                "maxOutputTokens": config.LLM_MAX_TOKENS,
            },
        }
        result = self._post(url, {"Content-Type": "application/json"}, payload)
        return result["candidates"][0]["content"]["parts"][0]["text"]

    def _anthropic(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "max_tokens": config.LLM_MAX_TOKENS,
            "temperature": config.LLM_TEMPERATURE,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }
        result = self._post(
            "https://api.anthropic.com/v1/messages", headers, payload
        )
        return "".join(
            block.get("text", "")
            for block in result.get("content", [])
            if block.get("type") == "text"
        )
