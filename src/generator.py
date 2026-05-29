"""
generator.py
Feed retrieved context + query to a language model and return an answer.
Supports:
  - Local: google/flan-t5-base (no GPU needed, ~250MB)
  - Remote: OpenRouter Chat Completions API
  - Remote: HuggingFace Inference API (Mistral or any instruction model)
"""

import os
import requests

try:
    from config_utils import load_config
except ImportError:
    from .config_utils import load_config


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


class LocalGenerator:
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


class OpenRouterGenerator:
    """
    Uses OpenRouter's OpenAI-compatible chat completions API.
    Requires OPENROUTER_API_KEY in the environment or a local .env file.
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
        print(f"OpenRouter generator configured: {model_name}")

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


class LocalFallbackGenerator:
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


class HFInferenceAPIGenerator:
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


class GeminiAPIGenerator:
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
    provider = cfg.get("generation_provider")
    if not provider:
        provider = "hf_inference_api" if cfg.get("use_hf_inference_api") else "local"
    provider = provider.lower()

    if provider == "openrouter":
        token = cfg.get("openrouter_api_key") or os.getenv("OPENROUTER_API_KEY")
        if not token:
            raise ValueError(
                "generation_provider is openrouter but no OpenRouter key was found. "
                "Set OPENROUTER_API_KEY in your environment or local .env file."
            )
        generator = OpenRouterGenerator(
            cfg["generation_model"],
            token,
            cfg["max_new_tokens"],
            cfg["temperature"],
            cfg.get("openrouter_site_url", ""),
            cfg.get("openrouter_app_name", "RAG Document QA"),
        )
        if cfg.get("fallback_to_local_on_rate_limit", True):
            return LocalFallbackGenerator(
                generator,
                cfg.get("local_fallback_model", "google/flan-t5-small"),
                cfg["max_new_tokens"],
                cfg.get("max_input_tokens", 512),
            )
        return generator

    if provider == "gemini_api":
        token = cfg.get("gemini_api_key") or os.environ.get("GEMINI_API_KEY")
        if not token:
            raise ValueError("GEMINI_API_KEY must be set in .env to use the Gemini API.")
        return GeminiAPIGenerator(
            cfg["generation_model"], token,
            cfg.get("temperature", 0.3)
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
        "Use 'openrouter', 'local', or 'hf_inference_api'."
    )


if __name__ == "__main__":
    cfg = load_config()
    generator = build_generator(cfg)
    test_context = "The Eiffel Tower is located in Paris, France. It was built in 1889 by Gustave Eiffel."
    test_question = "When was the Eiffel Tower built?"
    answer = generator.generate(test_context, test_question)
    print(f"Q: {test_question}\nA: {answer}")
