"""Local LLM interaction client via Ollama.

Strictly local-first: Never connects to external or cloud APIs.
Includes timeouts, clear health checks, and JSON response extraction.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Generator, List, Optional, Tuple

import httpx

from config import AppConfig
from src.exceptions import (
    ModelNotFoundError,
    OllamaConnectionError,
    StudyBuddyError,
)
from src.privacy import assert_local_configuration, sanitize_for_log


class LocalLLM:
    """Manages local inference against an Ollama daemon running on localhost."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        self.base_url = (base_url or AppConfig.OLLAMA_BASE_URL).rstrip("/")
        self.model_name = model_name or AppConfig.DEFAULT_MODEL
        self.timeout = timeout or AppConfig.OLLAMA_TIMEOUT
        # Strict privacy assertion on initialization
        assert_local_configuration(self.base_url)

    def check_connection(self) -> Tuple[bool, str, List[str]]:
        """Verify Ollama connectivity and return available downloaded models.

        Returns:
            Tuple of (is_connected, status_message, list_of_model_names)
        """
        assert_local_configuration(self.base_url)
        url = f"{self.base_url}/api/tags"
        try:
            with httpx.Client(timeout=4.0) as client:
                resp = client.get(url)
                if resp.status_code == 200:
                    data = resp.json()
                    models = [m.get("name", "") for m in data.get("models", [])]
                    if self.model_name in models or any(self.model_name in m for m in models):
                        return (True, f"Connected to Ollama (Model '{self.model_name}' ready)", models)
                    return (
                        True,
                        f"Ollama is running, but model '{self.model_name}' is not downloaded yet.",
                        models,
                    )
                return (False, f"Ollama returned HTTP {resp.status_code}", [])
        except httpx.ConnectError:
            return (
                False,
                f"Cannot connect to Ollama at {self.base_url}. Please ensure Ollama is installed and running (`ollama serve`).",
                [],
            )
        except httpx.TimeoutException:
            return (
                False,
                f"Connection to Ollama timed out at {self.base_url}.",
                [],
            )
        except Exception as e:
            return (False, f"Ollama connection error: {str(e)}", [])

    def is_model_available(self, model: Optional[str] = None) -> bool:
        """Check if a specific model is downloaded and ready."""
        target = model or self.model_name
        connected, _, models = self.check_connection()
        if not connected:
            return False
        return target in models or any(target in m for m in models)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Execute a text generation query against the local model.

        Args:
            prompt: User/Task prompt.
            system_prompt: Optional system prompt instructions.
            model: Model tag override.
            temperature: Sampling temperature.

        Returns:
            Generated response string.

        Raises:
            OllamaConnectionError: If Ollama cannot be reached.
            ModelNotFoundError: If the model is missing.
        """
        assert_local_configuration(self.base_url)
        target_model = model or self.model_name
        url = f"{self.base_url}/api/generate"

        payload: Dict[str, Any] = {
            "model": target_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else AppConfig.TEMPERATURE,
            },
        }
        if system_prompt:
            payload["system"] = system_prompt

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(url, json=payload)
                if resp.status_code == 404:
                    raise ModelNotFoundError(
                        f"Model '{target_model}' not found in Ollama. "
                        f"Run `ollama pull {target_model}` to download it locally."
                    )
                if resp.status_code != 200:
                    raise StudyBuddyError(
                        f"Ollama generation failed with status {resp.status_code}: {resp.text}"
                    )
                data = resp.json()
                return data.get("response", "").strip()
        except httpx.ConnectError as err:
            raise OllamaConnectionError(
                f"Could not connect to Ollama at {self.base_url}. "
                "Please run `ollama serve` in a terminal."
            ) from err
        except httpx.TimeoutException as err:
            raise OllamaConnectionError(
                f"Ollama request timed out after {self.timeout}s."
            ) from err

    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """Stream generated tokens from the local model."""
        assert_local_configuration(self.base_url)
        target_model = model or self.model_name
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": target_model,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": AppConfig.TEMPERATURE},
        }
        if system_prompt:
            payload["system"] = system_prompt

        try:
            with httpx.Client(timeout=self.timeout) as client:
                with client.stream("POST", url, json=payload) as response:
                    if response.status_code == 404:
                        raise ModelNotFoundError(f"Model '{target_model}' not found.")
                    for line in response.iter_lines():
                        if line:
                            data = json.loads(line)
                            yield data.get("response", "")
        except httpx.ConnectError as err:
            raise OllamaConnectionError(
                f"Ollama stream connection failed at {self.base_url}."
            ) from err

    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a response constrained to JSON and parse it.

        Extracts the first valid JSON object or array from the model output.
        """
        raw = self.generate(prompt=prompt, system_prompt=system_prompt, model=model)
        # Attempt direct JSON parse
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Attempt to extract JSON from markdown code block or brackets
        json_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
        match = re.search(json_pattern, raw)
        if match:
            candidate = match.group(1).strip()
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        # Try to locate outer braces { ... }
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = raw[start : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        raise ValueError(
            f"Failed to parse valid JSON from local LLM response: {sanitize_for_log(raw)}"
        )
