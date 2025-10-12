from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import requests

from .config import load_env_map


class LLMAPIError(Exception):
    def __init__(
        self,
        message: str,
        *,
        url: str | None = None,
        endpoint: str | None = None,
        status: int | None = None,
        response_text: str | None = None,
        response_json: Any | None = None,
    ) -> None:
        super().__init__(message)
        self.url = url
        self.endpoint = endpoint
        self.status = status
        self.response_text = response_text
        self.response_json = response_json


class LLMClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        timeout: Any | None = None,
        default_max_output_tokens: int | None = None,
        provider: str = "openai-compatible",
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout if timeout is not None else 60
        self.default_max_output_tokens = default_max_output_tokens
        self.provider = provider
        self.extra_headers = extra_headers or {}

    @classmethod
    def from_env(cls, provider_hint: Optional[str] = None) -> "LLMClient":
        env, _ = load_env_map()
        provider = (
            provider_hint
            or env.get("LLM_PROVIDER")
            or os.environ.get("LLM_PROVIDER")
            or "openai-compatible"
        ).strip().lower()

        def _parse_timeout(val: Optional[str]) -> Any | None:
            if not val:
                return None
            try:
                if "," in val:
                    parts = [p.strip() for p in val.split(",")]
                    return (float(parts[0]), float(parts[1]))
                return float(val)
            except Exception:
                return None

        def _parse_max_tokens(val: Optional[str]) -> Optional[int]:
            if not val:
                return None
            try:
                return int(val)
            except Exception:
                return None

        if provider in {"google", "gemini"}:
            base = (
                env.get("GOOGLE_API_BASE")
                or os.environ.get("GOOGLE_API_BASE")
                or "https://generativelanguage.googleapis.com"
            )
            key = (
                env.get("GOOGLE_API_KEY")
                or env.get("GEMINI_API_KEY")
                or os.environ.get("GOOGLE_API_KEY")
                or os.environ.get("GEMINI_API_KEY")
            )
            if not key:
                raise RuntimeError("Google Gemini API key not found (GOOGLE_API_KEY or GEMINI_API_KEY)")
            model = (
                env.get("MODEL_NAME")
                or env.get("MODEL")
                or os.environ.get("MODEL_NAME")
                or os.environ.get("MODEL")
                or "gemini-2.5-pro"
            )
            timeout = _parse_timeout(env.get("LLM_TIMEOUT") or os.environ.get("LLM_TIMEOUT")) or 60
            max_out = _parse_max_tokens(env.get("LLM_MAX_OUTPUT_TOKENS") or os.environ.get("LLM_MAX_OUTPUT_TOKENS"))
            extra_headers = {"x-goog-api-key": key}
            return cls(
                base,
                key,
                model,
                timeout=timeout,
                default_max_output_tokens=max_out,
                provider="google",
                extra_headers=extra_headers,
            )

        if provider in {"anthropic", "claude"}:
            base = (
                env.get("ANTHROPIC_API_BASE")
                or os.environ.get("ANTHROPIC_API_BASE")
                or "https://api.anthropic.com"
            )
            key = env.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise RuntimeError("Anthropic API key not found (ANTHROPIC_API_KEY)")
            model = (
                env.get("MODEL_NAME")
                or env.get("MODEL")
                or os.environ.get("MODEL_NAME")
                or os.environ.get("MODEL")
                or "claude-4-sonnet"
            )
            timeout = _parse_timeout(env.get("LLM_TIMEOUT") or os.environ.get("LLM_TIMEOUT")) or 60
            max_out = _parse_max_tokens(env.get("LLM_MAX_OUTPUT_TOKENS") or os.environ.get("LLM_MAX_OUTPUT_TOKENS"))
            extra_headers = {
                "x-api-key": key,
                "anthropic-version": env.get("ANTHROPIC_VERSION")
                or os.environ.get("ANTHROPIC_VERSION")
                or "2023-06-01",
            }
            return cls(
                base,
                key,
                model,
                timeout=timeout,
                default_max_output_tokens=max_out,
                provider="anthropic",
                extra_headers=extra_headers,
            )

        # Default openai-compatible
        base = (
            env.get("BASE_URL")
            or env.get("OPENAI_BASE_URL")
            or env.get("API_BASE_URL")
            or env.get("LLM_BASE_URL")
            or os.environ.get("BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("API_BASE_URL")
            or os.environ.get("LLM_BASE_URL")
            or "https://api.zhizengzeng.com/v1"
        )
        key = (
            env.get("OPENAI_API_KEY")
            or env.get("API_KEY")
            or env.get("LLM_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("API_KEY")
            or os.environ.get("LLM_API_KEY")
        )
        if not key:
            raise RuntimeError("API key not set in .env (OPENAI_API_KEY or API_KEY) or environment")
        model = (
            env.get("MODEL_NAME")
            or env.get("MODEL")
            or os.environ.get("MODEL_NAME")
            or os.environ.get("MODEL")
            or "gpt-4o-mini"
        )
        timeout = _parse_timeout(env.get("LLM_TIMEOUT") or os.environ.get("LLM_TIMEOUT")) or 60
        max_out = _parse_max_tokens(env.get("LLM_MAX_OUTPUT_TOKENS") or os.environ.get("LLM_MAX_OUTPUT_TOKENS"))
        return cls(
            base,
            key,
            model,
            timeout=timeout,
            default_max_output_tokens=max_out,
            provider="openai-compatible",
        )

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.2, max_tokens: Optional[int] = None) -> str:
        if self.provider == "google":
            return self._chat_google(messages, temperature, max_tokens)
        if self.provider == "anthropic":
            return self._chat_anthropic(messages, temperature, max_tokens)
        return self._chat_openai(messages, temperature, max_tokens)

    # --- Provider specific helpers -------------------------------------------------
    def _chat_openai(self, messages: List[Dict[str, str]], temperature: float, max_tokens: Optional[int]) -> str:
        def _parse_chat_completions(data: Dict[str, Any]) -> str:
            return data["choices"][0]["message"]["content"].strip()

        def _parse_responses(data: Dict[str, Any]) -> str:
            if isinstance(data, dict):
                if "output_text" in data and isinstance(data["output_text"], str):
                    return data["output_text"].strip()
                out = data.get("output")
                if isinstance(out, list) and out:
                    texts: List[str] = []
                    for item in out:
                        if not isinstance(item, dict):
                            continue
                        if item.get("type") == "message":
                            contents = item.get("content")
                            if isinstance(contents, list):
                                for c in contents:
                                    if isinstance(c, dict) and "text" in c:
                                        texts.append(str(c["text"]))
                    if texts:
                        return "\n".join(texts).strip()
            raise RuntimeError(f"Bad LLM responses format: {data}")

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        headers.update(self.extra_headers)

        prefer_responses = any(k in self.base_url for k in ["zhizengzeng", "responses-only"])

        if not prefer_responses:
            try:
                url = f"{self.base_url}/chat/completions"
                payload: Dict[str, Any] = {"model": self.model, "messages": messages, "temperature": temperature}
                if max_tokens is not None:
                    payload["max_tokens"] = max_tokens
                resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=self.timeout)
                status = resp.status_code
                txt = resp.text
                try:
                    data = resp.json()
                except Exception:
                    data = None
                if status >= 400:
                    raise LLMAPIError(f"HTTP {status}", url=url, endpoint="chat.completions", status=status, response_text=txt)
                if isinstance(data, dict) and "error" in data:
                    err = data.get("error")
                    msg = err.get("message") if isinstance(err, dict) else data
                    if isinstance(msg, str) and "v1/responses" in msg:
                        raise LLMAPIError(f"fallback_to_responses: {msg}", url=url, endpoint="chat.completions", status=status, response_json=data)
                    raise LLMAPIError(f"LLM API error: {msg}", url=url, endpoint="chat.completions", status=status, response_json=data)
                if isinstance(data, dict) and "code" in data and data.get("code") not in (0, 200):
                    raise LLMAPIError(f"LLM API error code={data.get('code')}: {data.get('msg')}", url=url, endpoint="chat.completions", status=status, response_json=data)
                return _parse_chat_completions(data)
            except Exception:
                pass

        url = f"{self.base_url}/responses"
        payload: Dict[str, Any] = {"model": self.model, "input": messages, "temperature": temperature}
        eff_max = max_tokens if max_tokens is not None else self.default_max_output_tokens
        if eff_max is not None:
            payload["max_output_tokens"] = eff_max
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=self.timeout)
        status = resp.status_code
        txt = resp.text
        try:
            data = resp.json()
        except Exception:
            data = None
        if status >= 400:
            raise LLMAPIError(f"HTTP {status}", url=url, endpoint="responses", status=status, response_text=txt)
        if isinstance(data, dict) and data.get("error"):
            err = data.get("error")
            msg = err.get("message") if isinstance(err, dict) else data
            raise LLMAPIError(f"LLM API error: {msg}", url=url, endpoint="responses", status=status, response_json=data)
        if isinstance(data, dict) and "code" in data and data.get("code") not in (0, 200):
            raise LLMAPIError(f"LLM API error code={data.get('code')}: {data.get('msg')}", url=url, endpoint="responses", status=status, response_json=data)
        return _parse_responses(data)

    def _chat_google(self, messages: List[Dict[str, str]], temperature: float, max_tokens: Optional[int]) -> str:
        system_parts: List[str] = []
        contents: List[Dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "user")
            text = str(msg.get("content", ""))
            if role == "system":
                system_parts.append(text)
            elif role == "assistant":
                contents.append({"role": "model", "parts": [{"text": text}]})
            else:
                contents.append({"role": "user", "parts": [{"text": text}]})

        payload: Dict[str, Any] = {
            "model": self.model,
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
            },
        }
        eff_max = max_tokens if max_tokens is not None else self.default_max_output_tokens
        if eff_max is not None:
            payload["generationConfig"]["maxOutputTokens"] = eff_max
        if system_parts:
            payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}
        url = f"{self.base_url}/v1beta/models/{self.model}:generateContent"
        headers = {"Content-Type": "application/json"}
        headers.update(self.extra_headers)
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=self.timeout)
        if resp.status_code >= 400:
            raise LLMAPIError(f"HTTP {resp.status_code}", url=url, endpoint="google.generateContent", status=resp.status_code, response_text=resp.text)
        try:
            data = resp.json()
        except Exception as e:
            raise LLMAPIError("Non-JSON response", url=url, endpoint="google.generateContent", status=resp.status_code, response_text=resp.text) from e
        try:
            candidates = data.get("candidates") or []
            first = candidates[0]
            parts = first.get("content", {}).get("parts", [])
            text_parts = [p.get("text", "") for p in parts if isinstance(p, dict)]
            return "\n".join(t.strip() for t in text_parts if isinstance(t, str)).strip()
        except Exception as e:
            raise LLMAPIError("Failed to parse Gemini response", url=url, endpoint="google.generateContent", status=resp.status_code, response_json=data) from e

    def _chat_anthropic(self, messages: List[Dict[str, str]], temperature: float, max_tokens: Optional[int]) -> str:
        system_prompts: List[str] = []
        converted: List[Dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "user")
            text = str(msg.get("content", ""))
            if role == "system":
                system_prompts.append(text)
                continue
            anth_role = "user" if role == "user" else "assistant"
            converted.append({"role": anth_role, "content": [{"type": "text", "text": text}]})
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": converted,
            "temperature": temperature,
            "max_tokens": max_tokens or self.default_max_output_tokens or 1024,
        }
        if system_prompts:
            payload["system"] = "\n\n".join(system_prompts)
        url = f"{self.base_url}/v1/messages"
        headers = {"Content-Type": "application/json"}
        headers.update(self.extra_headers)
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=self.timeout)
        if resp.status_code >= 400:
            raise LLMAPIError(f"HTTP {resp.status_code}", url=url, endpoint="anthropic.messages", status=resp.status_code, response_text=resp.text)
        try:
            data = resp.json()
        except Exception as e:
            raise LLMAPIError("Non-JSON response", url=url, endpoint="anthropic.messages", status=resp.status_code, response_text=resp.text) from e
        try:
            content_list = data.get("content") or []
            texts = [
                part.get("text", "")
                for part in content_list
                if isinstance(part, dict) and part.get("type") == "text"
            ]
            if texts:
                return "\n".join(t.strip() for t in texts if isinstance(t, str)).strip()
        except Exception as e:
            raise LLMAPIError("Failed to parse Anthropic response", url=url, endpoint="anthropic.messages", status=resp.status_code, response_json=data) from e
        raise LLMAPIError("Empty response content", url=url, endpoint="anthropic.messages", status=resp.status_code, response_json=data)

    # --- Metadata helpers ----------------------------------------------------------
    def list_models(self) -> List[str]:
        if self.provider == "google":
            url = f"{self.base_url}/v1beta/models"
            headers = {"Content-Type": "application/json"}
            headers.update(self.extra_headers)
            resp = requests.get(url, headers=headers, timeout=self.timeout if isinstance(self.timeout, (int, float)) else 30)
            if resp.status_code >= 400:
                raise LLMAPIError(f"HTTP {resp.status_code}", url=url, endpoint="google.models", status=resp.status_code, response_text=resp.text)
            try:
                data = resp.json()
            except Exception as e:
                raise LLMAPIError("Non-JSON response", url=url, endpoint="google.models", status=resp.status_code, response_text=resp.text) from e
            models = []
            for m in data.get("models", []):
                name = m.get("name")
                if isinstance(name, str):
                    models.append(name.split("/")[-1])
            return models

        if self.provider == "anthropic":
            url = f"{self.base_url}/v1/models"
            headers = {"Content-Type": "application/json"}
            headers.update(self.extra_headers)
            resp = requests.get(url, headers=headers, timeout=self.timeout if isinstance(self.timeout, (int, float)) else 30)
            if resp.status_code >= 400:
                raise LLMAPIError(f"HTTP {resp.status_code}", url=url, endpoint="anthropic.models", status=resp.status_code, response_text=resp.text)
            try:
                data = resp.json()
            except Exception as e:
                raise LLMAPIError("Non-JSON response", url=url, endpoint="anthropic.models", status=resp.status_code, response_text=resp.text) from e
            models = []
            for m in data.get("data", []):
                mid = m.get("id")
                if isinstance(mid, str):
                    models.append(mid)
            return models

        url = f"{self.base_url}/models"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        headers.update(self.extra_headers)
        resp = requests.get(url, headers=headers, timeout=self.timeout if isinstance(self.timeout, (int, float)) else 30)
        if resp.status_code >= 400:
            raise LLMAPIError(f"HTTP {resp.status_code}", url=url, endpoint="models", status=resp.status_code, response_text=resp.text)
        try:
            data = resp.json()
        except Exception as e:
            raise LLMAPIError("Non-JSON response", url=url, endpoint="models", status=resp.status_code, response_text=resp.text) from e
        ids: List[str] = []
        for m in data.get("data", []):
            mid = m.get("id") or m.get("name")
            if isinstance(mid, str):
                ids.append(mid)
        return ids
