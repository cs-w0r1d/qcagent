# -*- coding: utf-8 -*-
"""
LLM Clients: Qwen (DashScope OpenAI-compatible) and Dify Workflow
"""

import os
import re
import json
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


# ============================================================
# Qwen Client (DashScope OpenAI-compatible API)
# ============================================================
@dataclass
class QwenClient:
    """
    Qwen API Client via DashScope OpenAI-compatible endpoint.
    """
    api_key: str = ""
    model: str = "Qwen/Qwen3-VL-30B-A3B-Thinking"  # core QCAgent model used in the paper
    base_url: str = "https://api.siliconflow.cn/v1"  # OpenAI-compatible endpoint (override per provider)
    temperature: float = 0.7
    max_tokens: int = 16384
    timeout: int = 600  # 235B thinking model needs longer time
    max_retries: int = 3
    enable_thinking: bool = True  # qwen3 thinking mode

    def __post_init__(self):
        if not self.api_key:
            self.api_key = os.getenv("QWEN_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "Qwen API key is required. Set QWEN_API_KEY env var or pass api_key."
            )

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
    ) -> Dict[str, str]:
        """
        Call Qwen Chat API.

        Returns:
            {
                "content": str,       # Final answer (thinking stripped)
                "thinking": str,      # Thinking process (if any)
                "raw_content": str,   # Raw full output
                "usage": dict,        # Token usage stats
            }
        """
        url = self.base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature or self.temperature,
            "max_tokens": self.max_tokens,
        }

        # qwen3 thinking mode
        if self.enable_thinking and "thinking" in self.model:
            payload["extra_body"] = {"enable_thinking": True}

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(f"Qwen API call (attempt {attempt}/{self.max_retries})")
                r = requests.post(
                    url, headers=headers, json=payload,
                    timeout=self.timeout,
                )

                if r.status_code == 429:
                    wait = min(2 ** attempt, 30)
                    logger.warning(f"Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                    continue

                r.raise_for_status()
                data = r.json()

                choice = data["choices"][0]
                message = choice.get("message", {})
                raw_content = message.get("content", "")

                thinking = ""
                content = raw_content

                # Handle <think>...</think> tags
                think_match = re.search(
                    r"<think>(.*?)</think>",
                    raw_content, flags=re.DOTALL,
                )
                if think_match:
                    thinking = think_match.group(1).strip()
                    content = re.sub(
                        r"<think>.*?</think>", "",
                        raw_content, flags=re.DOTALL,
                    ).strip()

                # Some models put thinking in a separate field
                if "reasoning_content" in message:
                    thinking = message["reasoning_content"]

                usage = data.get("usage", {})

                return {
                    "content": content,
                    "thinking": thinking,
                    "raw_content": raw_content,
                    "usage": usage,
                }

            except requests.exceptions.Timeout:
                last_error = f"Request timeout (timeout={self.timeout}s)"
                logger.warning(f"Attempt {attempt}: {last_error}")
            except requests.exceptions.HTTPError as e:
                last_error = f"HTTP error: {e}, body={r.text[:500]}"
                logger.warning(f"Attempt {attempt}: {last_error}")
                if r.status_code in (400, 401, 403):
                    break  # Don't retry auth errors
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Attempt {attempt}: {last_error}")

            if attempt < self.max_retries:
                time.sleep(2 ** attempt)

        raise RuntimeError(
            f"Qwen API call failed ({self.max_retries} attempts): {last_error}"
        )


# ============================================================
# Dify Workflow Client
# ============================================================
@dataclass
class DifyClient:
    """
    Dify Workflow API Client.
    Calls a deployed "Pathology Report QC Agent" workflow.
    """
    base_url: str = ""
    api_key: str = ""
    timeout: int = 300
    run_path: str = "/v1/workflows/run"

    def __post_init__(self):
        if not self.base_url:
            self.base_url = os.getenv("DIFY_BASE_URL", "http://localhost")
        if not self.api_key:
            self.api_key = os.getenv("DIFY_API_KEY", "")
        forced = os.getenv("AGENT_RUN_PATH", "").strip()
        if forced:
            self.run_path = forced if forced.startswith("/") else "/" + forced

    def _headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def run(
        self,
        reports: str,
        qc_rules: str = "",
    ) -> Dict[str, Any]:
        """
        Call Dify workflow.

        Args:
            reports: Report text (JSON string or plain text)
            qc_rules: QC rules

        Returns:
            Workflow outputs dict containing:
            - non_compliant_texts
            - need_more_info_texts
            - qc_json
        """
        url = self.base_url.rstrip("/") + self.run_path

        payload = {
            "inputs": {
                "reports": reports,
                "qc_rules": qc_rules,
            },
            "response_mode": "blocking",
            "user": "qc-pipeline",
        }

        logger.info(f"Dify workflow call: {url}")
        r = requests.post(
            url, headers=self._headers(), json=payload,
            timeout=self.timeout,
        )
        r.raise_for_status()
        raw = r.json()

        outputs = raw
        if isinstance(raw, dict):
            if "data" in raw and isinstance(raw["data"], dict):
                outputs = raw["data"].get("outputs", raw["data"])
            elif "outputs" in raw:
                outputs = raw["outputs"]

        if not isinstance(outputs, dict):
            raise ValueError(f"Cannot parse Dify output: {raw}")

        return outputs


if __name__ == "__main__":
    # Quick test
    client = QwenClient()
    result = client.chat(
        system_prompt="You are a helpful assistant.",
        user_prompt="Say 'hello world'",
    )
    print("Content:", result["content"])
    print("Usage:", result["usage"])
