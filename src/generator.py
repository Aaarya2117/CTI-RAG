"""
generator.py
Feed retrieved context + query to a language model and return an answer.
Supports:
  - Local: google/flan-t5-base (no GPU needed, ~250MB)
  - Remote: OpenRouter Chat Completions API
  - Remote: HuggingFace Inference API (Mistral or any instruction model)
"""

import os
os.environ.setdefault("USE_TF", "0")
import requests

from typing import Optional

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

try:
    from config_utils import load_config
except ImportError:
    from .config_utils import load_config

try:
    from laya_layer import gate_answer
except ImportError:
    try:
        from .laya_layer import gate_answer
    except ImportError:
        gate_answer = None


def is_grounded(gate_result: dict, threshold: float = 0.5) -> bool:
    """Helper to check if gate result meets grounding threshold."""
    grounded = gate_result.get("grounded")
    if grounded is None:
        return True
    if isinstance(grounded, bool):
        return grounded
    if isinstance(grounded, (int, float)):
        return float(grounded) >= threshold
    if isinstance(grounded, dict):
        if "choice" in grounded:
            val = grounded["choice"]
            return val if isinstance(val, bool) else str(val).lower() in ("yes", "true", "1")
        if "prob" in grounded:
            return float(grounded["prob"]) >= threshold
        if "score" in grounded:
            return float(grounded["score"]) >= threshold
        if "answer" in grounded:
            ans = grounded["answer"]
            return ans if isinstance(ans, bool) else str(ans).lower() in ("yes", "true", "1")
    return True


def gate(answer_text: str, retrieved_ids: Optional[list] = None) -> dict:
    """
    Run Laya post-generation gating on answer text.
    Evaluates whether the answer is grounded in provided context and cites IDs.
    Returns:
        dict with gate decision metadata, e.g.
        {"grounded": bool/float, "cites_id": bool, "available": bool}
    """
    if retrieved_ids is None:
        retrieved_ids = []

    if gate_answer is None:
        return {
            "grounded": True,
            "cites_id": False,
            "available": False,
            "note": "Laya gate_answer unavailable",
        }

    try:
        decision = gate_answer(answer_text, retrieved_ids)
        if isinstance(decision, dict):
            res = dict(decision)
            res["available"] = True
            return res
        return {"grounded": True, "cites_id": False, "raw": decision, "available": True}
    except Exception as exc:
        print(f"Warning: Laya gate_answer failed: {exc}")
        return {
            "grounded": True,
            "cites_id": False,
            "available": False,
            "error": str(exc),
        }


class BaseGenerator:
    """Base class providing common Laya gating interface across all generators."""

    def generate(self, context: str, question: str) -> str:
        raise NotImplementedError

    def gate(self, answer_text: str, retrieved_ids: Optional[list] = None) -> dict:
        """Run Laya gate on answer."""
        return gate(answer_text, retrieved_ids)

    def generate_with_gate(
        self,
        context: str,
        question: str,
        retrieved_ids: Optional[list] = None,
    ) -> dict:
        """Generate answer and evaluate with Laya gating."""
        answer = self.generate(context, question)
        gate_decision = self.gate(answer, retrieved_ids or [])
        return {
            "answer": answer,
            "gate": gate_decision,
            "grounded": gate_decision.get("grounded", True),
            "cites_id": gate_decision.get("cites_id", False),
        }



PROMPT_TEMPLATE = """You are a Cyber Threat Intelligence (CTI) analyst assistant.
Answer the analyst's question using ONLY the provided threat intelligence context below.
If the answer is not found in the context, say "Insufficient intelligence in the loaded documents to answer this."

When relevant, reference:
- MITRE ATT&CK tactic and technique IDs (e.g., T1059 - Command and Scripting Interpreter)
- Threat actor names or groups mentioned in the context
- CVE identifiers if present
- IOCs (IPs, domains, file hashes) if mentioned

Context:
{context}

Analyst Question: {question}

Intelligence Assessment:"""


class OpenRouterRateLimitError(RuntimeError):
    """Raised when a free OpenRouter provider is temporarily rate-limited."""


class LocalGenerator(BaseGenerator):
    """
    Uses Flan-T5 locally via HuggingFace Transformers.
    Works on CPU. Good for development and demo.
    """

    def __init__(self, model_name: str, max_new_tokens: int = 256, max_input_tokens: int = 512):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        print(f"Loading local generator: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.max_input_tokens = max_input_tokens
        print("Local generator ready.")

    def generate(self, context: str, question: str) -> str:
        import torch

        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


class OpenRouterLangChainGenerator(BaseGenerator):
    """
    Uses OpenRouter via LangChain ChatOpenAI.
    API key is read from OPENROUTER_API_KEY or config.
    Offline initialization is supported gracefully without raising errors.
    """

    def __init__(
        self,
        model_name: str = "google/gemini-2.5-flash",
        api_key: Optional[str] = None,
        temperature: float = 0.2,
        base_url: str = "https://openrouter.ai/api/v1",
        default_headers: Optional[dict] = None,
        max_new_tokens: Optional[int] = None,
        site_url: Optional[str] = None,
        app_name: Optional[str] = None,
        **kwargs,
    ):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY") or ""
        self.temperature = temperature
        self.base_url = base_url
        self.max_new_tokens = max_new_tokens

        headers = {
            "HTTP-Referer": site_url or "https://github.com/Aaarya2117/CTI-RAG",
            "X-Title": app_name or "CTI-RAG",
        }
        if default_headers:
            headers.update(default_headers)
        self.default_headers = headers

        self.llm = None
        if self.api_key:
            self._init_llm()
        else:
            print(
                "Notice: OPENROUTER_API_KEY is not set. OpenRouter generator initialized in offline mode. "
                "Set OPENROUTER_API_KEY before invoking .generate()."
            )

    def _init_llm(self):
        global ChatOpenAI
        if ChatOpenAI is None:
            from langchain_openai import ChatOpenAI

        init_kwargs = {
            "model": self.model_name,
            "api_key": self.api_key,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "default_headers": self.default_headers,
        }
        if self.max_new_tokens:
            init_kwargs["max_tokens"] = self.max_new_tokens

        self.llm = ChatOpenAI(**init_kwargs)
        print(f"OpenRouter ChatOpenAI generator configured: {self.model_name}")

    def generate(self, context: str, question: str) -> str:
        if not self.api_key:
            self.api_key = os.environ.get("OPENROUTER_API_KEY") or ""
        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is required to generate answers with OpenRouter. "
                "Set OPENROUTER_API_KEY in your environment or local .env file."
            )
        if self.llm is None:
            self._init_llm()

        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            if isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, str):
                        parts.append(part)
                    elif isinstance(part, dict) and "text" in part:
                        parts.append(part["text"])
                    else:
                        parts.append(str(part))
                content = "".join(parts)
            return content.strip() if content else ""
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "rate limit" in err_str.lower():
                raise OpenRouterRateLimitError(
                    f"OpenRouter provider is temporarily rate-limited: {err_str}"
                ) from e
            raise


OpenRouterGenerator = OpenRouterLangChainGenerator


class LegacyOpenRouterGenerator(BaseGenerator):
    """
    Direct HTTP requests generator for OpenRouter API.
    Preserved for backward-compatibility.
    """

    API_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        model_name: str,
        api_key: str,
        max_new_tokens: int = 256,
        temperature: float = 0.3,
        site_url: str = "",
        app_name: str = "RAG Document QA",
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if site_url:
            self.headers["HTTP-Referer"] = site_url
        if app_name:
            self.headers["X-Title"] = app_name
        print(f"Legacy OpenRouter generator configured: {model_name}")

    def generate(self, context: str, question: str) -> str:
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }
        response = requests.post(self.API_URL, headers=self.headers, json=payload, timeout=90)

        if response.status_code != 200:
            message = self._extract_error_message(response)
            if response.status_code == 429:
                raise OpenRouterRateLimitError(
                    "OpenRouter's free provider for this model is temporarily rate-limited. "
                    "Wait and retry, use another OpenRouter model, or add your own provider key "
                    "in OpenRouter settings."
                )
            raise RuntimeError(f"OpenRouter API error {response.status_code}: {message}")

        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError(f"OpenRouter returned no choices: {data}")

        message_data = choices[0]["message"]
        content = message_data.get("content")

        if content is None:
            refusal = message_data.get("refusal")
            if refusal:
                return f"Model refused to answer: {refusal}"
            return "Error: The model returned an empty response (content is null)."

        return content.strip()

    @staticmethod
    def _extract_error_message(response) -> str:
        try:
            data = response.json()
        except ValueError:
            return response.text[:500]

        error = data.get("error", data)
        if isinstance(error, dict):
            return str(error.get("message", error))
        return str(error)


class LocalFallbackGenerator(BaseGenerator):
    """
    Wraps a hosted generator and falls back to a local model when free hosted
    inference is temporarily unavailable.
    """

    def __init__(
        self,
        primary,
        fallback_model_name: str,
        max_new_tokens: int = 256,
        max_input_tokens: int = 512,
    ):
        self.primary = primary
        self.fallback_model_name = fallback_model_name
        self.max_new_tokens = max_new_tokens
        self.max_input_tokens = max_input_tokens
        self.fallback = None

    def generate(self, context: str, question: str) -> str:
        try:
            return self.primary.generate(context, question)
        except OpenRouterRateLimitError as exc:
            print(f"{exc} Falling back to local model: {self.fallback_model_name}")
            if self.fallback is None:
                self.fallback = LocalGenerator(
                    self.fallback_model_name,
                    self.max_new_tokens,
                    self.max_input_tokens,
                )
            return self.fallback.generate(context, question)


class HFInferenceAPIGenerator(BaseGenerator):
    """
    Uses the HuggingFace Inference API for larger models (e.g., Mistral-7B).
    Requires a free HF token — get one at huggingface.co/settings/tokens
    """

    HF_API_URL = "https://api-inference.huggingface.co/models/{model}"

    def __init__(self, model_name: str, hf_token: str, max_new_tokens: int = 256, temperature: float = 0.3):
        self.url = self.HF_API_URL.format(model=model_name)
        self.headers = {"Authorization": f"Bearer {hf_token}"}
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        print(f"HF Inference API generator configured: {model_name}")

    def generate(self, context: str, question: str) -> str:
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "return_full_text": False,
            },
        }
        response = requests.post(self.url, headers=self.headers, json=payload, timeout=60)

        if response.status_code != 200:
            raise RuntimeError(f"HF API error {response.status_code}: {response.text}")

        data = response.json()
        if isinstance(data, list):
            return data[0].get("generated_text", "").strip()
        return str(data).strip()


class GeminiLangChainGenerator(BaseGenerator):
    """Uses Google's Gemini API via LangChain ChatGoogleGenerativeAI."""

    def __init__(self, model_name: str, api_key: Optional[str] = None, temperature: float = 0.2, **kwargs):
        if model_name.startswith("models/"):
            model_name = model_name[7:]
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or ""
        self.temperature = temperature
        self.llm = None
        if self.api_key:
            self._init_llm()
        else:
            print("Notice: GEMINI_API_KEY not set. Gemini generator initialized in offline mode.")

    def _init_llm(self):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            self.llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                api_key=self.api_key,
                temperature=self.temperature,
            )
            print(f"Gemini ChatGoogleGenerativeAI configured: {self.model_name}")
        except Exception as e:
            print(f"Notice: ChatGoogleGenerativeAI init deferred ({e})")

    def generate(self, context: str, question: str) -> str:
        if not self.api_key:
            self.api_key = os.environ.get("GEMINI_API_KEY") or ""
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is required to generate answers with Gemini. "
                "Set GEMINI_API_KEY in your environment or local .env file."
            )
        if self.llm is None:
            self._init_llm()

        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        response = self.llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict) and "text" in part:
                    parts.append(part["text"])
                else:
                    parts.append(str(part))
            content = "".join(parts)
        return content.strip() if content else ""


class GeminiAPIGenerator(BaseGenerator):
    """Uses Google's Gemini API via google-genai SDK."""
    
    def __init__(self, model_name: str, api_key: str, temperature: float = 0.3):
        from google import genai
        self.client = genai.Client(api_key=api_key)
        # the SDK expects models to be named like 'gemini-2.5-flash' without 'models/'
        if model_name.startswith("models/"):
            model_name = model_name[7:]
        self.model_name = model_name
        self.temperature = temperature
        print(f"Gemini API generator configured: {model_name}")

    def generate(self, context: str, question: str) -> str:
        from google.genai import types
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=self.temperature,
                ),
            )
            return response.text.strip() if response.text else "Error: Gemini returned an empty response."
        except Exception as e:
            return f"Gemini API Error: {str(e)}"

def build_generator(cfg: dict):
    provider = cfg.get("generation_provider", "gemini_api")
    if not provider:
        provider = "gemini_api"
    provider = provider.lower()

    if provider in ("gemini_api", "gemini"):
        token = cfg.get("gemini_api_key") or os.environ.get("GEMINI_API_KEY") or ""
        return GeminiLangChainGenerator(
            cfg.get("generation_model", "gemini-2.5-flash"),
            token,
            cfg.get("temperature", 0.2),
        )


    if provider == "hf_inference_api":
        token = (
            cfg.get("hf_api_token")
            or os.getenv("HF_API_TOKEN")
            or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        )
        if not token:
            raise ValueError(
                "use_hf_inference_api is true but no Hugging Face token was found. "
                "Set HF_API_TOKEN in your environment, or add hf_api_token to config.yaml."
            )
        return HFInferenceAPIGenerator(
            cfg["generation_model"], token,
            cfg["max_new_tokens"], cfg["temperature"]
        )

    if provider == "local":
        return LocalGenerator(
            cfg["generation_model"],
            cfg["max_new_tokens"],
            cfg.get("max_input_tokens", 512),
        )

    raise ValueError(
        f"Unsupported generation_provider: {provider}. "
        "Use 'openrouter', 'gemini_api', 'local', or 'hf_inference_api'."
    )


if __name__ == "__main__":
    cfg = load_config()
    generator = build_generator(cfg)
    test_context = "The Eiffel Tower is located in Paris, France. It was built in 1889 by Gustave Eiffel."
    test_question = "When was the Eiffel Tower built?"
    try:
        answer = generator.generate(test_context, test_question)
        print(f"Q: {test_question}\nA: {answer}")
    except ValueError as exc:
        print(f"Generator initialized successfully ({type(generator).__name__}). Offline mode check: {exc}")

    print("\nTesting Laya gate helper:")
    sample_answer = "The vulnerability CVE-2023-38606 is an elevation of privilege flaw associated with T1059."
    gate_decision = generator.gate(sample_answer, ["CVE-2023-38606", "T1059"])
    print(f"Gate output: {gate_decision}")

