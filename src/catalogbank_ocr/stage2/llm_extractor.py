"""Stage 2 — Step 4: LLM Extractor.

Sends each ContextChunk's prompt (built by llm_input_builder) to a local LLM
and parses the JSON response into a structured ExtractionResult.

Two backends are supported — whichever is available on the machine:

  1. Ollama (recommended for POC)
     - Requires `ollama` running locally: https://ollama.com
     - Model pulled with: `ollama pull qwen3:8b`
     - Python client: `pip install ollama`
     - Handles GPU/CPU memory automatically; no Python VRAM management needed.

  2. Transformers (fallback / research use)
     - Requires: `pip install transformers>=4.40 accelerate bitsandbytes`
     - Loads Qwen/Qwen3-8B-Instruct with 4-bit quantization via bitsandbytes.
     - Slower to initialize (~30 s first call) but works without Ollama.

  3. Dry-run / mock (testing without any LLM)
     - Returns a placeholder ExtractionResult with no products/attributes/relations.
     - Useful for verifying the full pipeline wiring without GPU access.

Backend selection:
  - Pass backend="ollama" | "transformers" | "mock" explicitly, OR
  - Leave backend=None (default) to auto-detect: tries Ollama first, then
    transformers, then mock with a warning.

Usage example::

    from catalogbank_ocr.stage2.llm_extractor import LLMExtractor, ExtractionResult
    from catalogbank_ocr.stage2.context_builder import build_context_chunks_from_file
    from catalogbank_ocr.stage2.llm_input_builder import build_llm_input
    from pathlib import Path

    extractor = LLMExtractor()           # auto-detects backend
    chunks = build_context_chunks_from_file(Path("outputs/canonical/.../doc.json"))
    for chunk in chunks:
        llm_input = build_llm_input(chunk)
        result = extractor.extract(llm_input)
        print(result)
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output schema — mirrors EXTRACTION_SCHEMA in llm_input_builder.py
# ---------------------------------------------------------------------------

@dataclass
class ExtractedProduct:
    name: str
    model: Optional[str] = None
    sku: Optional[str] = None
    category: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "model": self.model, "sku": self.sku, "category": self.category}


@dataclass
class ExtractedAttribute:
    entity: str
    key: str
    value: str
    unit: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"entity": self.entity, "key": self.key, "value": self.value, "unit": self.unit}


@dataclass
class ExtractedRelation:
    subject: str
    predicate: str
    object: str

    def to_dict(self) -> Dict[str, Any]:
        return {"subject": self.subject, "predicate": self.predicate, "object": self.object}


@dataclass
class ExtractionResult:
    """All entities and relations extracted from one ContextChunk."""

    chunk_id: str
    doc_stem: str
    section_path: List[str]
    products: List[ExtractedProduct] = field(default_factory=list)
    attributes: List[ExtractedAttribute] = field(default_factory=list)
    relations: List[ExtractedRelation] = field(default_factory=list)
    raw_response: str = ""
    backend_used: str = ""
    latency_s: float = 0.0
    parse_error: Optional[str] = None

    @property
    def is_empty(self) -> bool:
        return not self.products and not self.attributes and not self.relations

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_stem": self.doc_stem,
            "section_path": self.section_path,
            "products": [p.to_dict() for p in self.products],
            "attributes": [a.to_dict() for a in self.attributes],
            "relations": [r.to_dict() for r in self.relations],
            "meta": {
                "backend_used": self.backend_used,
                "latency_s": round(self.latency_s, 3),
                "parse_error": self.parse_error,
            },
        }

    def __repr__(self) -> str:
        return (
            f"ExtractionResult(chunk={self.chunk_id!r}, "
            f"products={len(self.products)}, "
            f"attrs={len(self.attributes)}, "
            f"relations={len(self.relations)}, "
            f"backend={self.backend_used!r}, "
            f"latency={self.latency_s:.1f}s)"
        )


# ---------------------------------------------------------------------------
# JSON response parser
# ---------------------------------------------------------------------------

# Compiled once — finds the first {...} block in the LLM response, handling
# models that wrap their answer in markdown fences or prose.
_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}", re.MULTILINE)

_VALID_PREDICATES = {
    "HAS_ATTRIBUTE",
    "IS_VARIANT_OF",
    "BELONGS_TO_CATEGORY",
    "HAS_MATERIAL",
}


def _parse_llm_response(
    raw: str,
    chunk_id: str,
    doc_stem: str,
    section_path: List[str],
) -> ExtractionResult:
    """Parse raw LLM text into an ExtractionResult.

    Strategy:
      1. Strip markdown fences (```json ... ```) if present.
      2. Find the first {...} block via regex.
      3. json.loads() — if it fails, try to repair common issues (trailing
         commas, single quotes) before giving up.
      4. Validate structure and coerce each list item into typed dataclasses.
    """
    result = ExtractionResult(
        chunk_id=chunk_id,
        doc_stem=doc_stem,
        section_path=section_path,
        raw_response=raw,
    )

    # Strip markdown fences
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip()
    cleaned = cleaned.replace("```", "").strip()

    # Find JSON block
    match = _JSON_BLOCK_RE.search(cleaned)
    if not match:
        result.parse_error = "No JSON object found in response"
        logger.warning("LLM response for %s: no JSON block found", chunk_id)
        return result

    json_str = match.group(0)

    # Attempt parse with progressive repairs
    parsed: Optional[Dict[str, Any]] = None
    for attempt, text in enumerate([json_str, _repair_json(json_str)]):
        try:
            parsed = json.loads(text)
            break
        except json.JSONDecodeError as exc:
            if attempt == 1:
                result.parse_error = f"JSONDecodeError: {exc}"
                logger.warning("LLM response for %s: JSON parse failed: %s", chunk_id, exc)
                return result

    if not isinstance(parsed, dict):
        result.parse_error = "Parsed value is not a dict"
        return result

    # --- Products ---
    for item in parsed.get("products") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        result.products.append(ExtractedProduct(
            name=name,
            model=_str_or_none(item.get("model")),
            sku=_str_or_none(item.get("sku")),
            category=_str_or_none(item.get("category")),
        ))

    # --- Attributes ---
    for item in parsed.get("attributes") or []:
        if not isinstance(item, dict):
            continue
        entity = str(item.get("entity") or "").strip()
        key = str(item.get("key") or "").strip()
        value = str(item.get("value") or "").strip()
        if not (entity and key and value):
            continue
        result.attributes.append(ExtractedAttribute(
            entity=entity,
            key=key,
            value=value,
            unit=_str_or_none(item.get("unit")),
        ))

    # --- Relations ---
    for item in parsed.get("relations") or []:
        if not isinstance(item, dict):
            continue
        subject = str(item.get("subject") or "").strip()
        predicate = str(item.get("predicate") or "").strip().upper()
        obj = str(item.get("object") or "").strip()
        if not (subject and predicate and obj):
            continue
        # Normalise unknown predicates rather than dropping them
        if predicate not in _VALID_PREDICATES:
            predicate = "HAS_ATTRIBUTE"
        result.relations.append(ExtractedRelation(subject=subject, predicate=predicate, object=obj))

    return result


def _str_or_none(v: Any) -> Optional[str]:
    if v is None or str(v).lower() in {"null", "none", ""}:
        return None
    return str(v).strip()


def _repair_json(text: str) -> str:
    """Light-touch JSON repair for common LLM output quirks."""
    # Trailing commas before } or ]
    text = re.sub(r",\s*([}\]])", r"\1", text)
    # Single-quoted strings → double-quoted (naïve but often good enough)
    text = re.sub(r"(?<![\\])'", '"', text)
    return text


# ---------------------------------------------------------------------------
# Backend implementations
# ---------------------------------------------------------------------------

class _OllamaBackend:
    """Thin wrapper around the `ollama` Python client."""

    def __init__(self, model: str, host: Optional[str] = None, timeout: int = 120) -> None:
        try:
            import ollama as _ollama
            self._ollama = _ollama
        except ImportError as exc:
            raise ImportError(
                "ollama package not installed. Run: pip install ollama"
            ) from exc

        self.model = model
        self.host = host
        self.timeout = timeout
        # Optionally configure a non-default host
        if host:
            self._client = self._ollama.Client(host=host)
        else:
            self._client = self._ollama.Client()

    @property
    def name(self) -> str:
        return f"ollama/{self.model}"

    def is_available(self) -> bool:
        """Check that Ollama is running and the model is pulled."""
        try:
            models = self._client.list()
            available = [m.model for m in models.models]
            # Accept prefix match so "qwen3:8b" matches "qwen3:8b-instruct" etc.
            return any(m.startswith(self.model.split(":")[0]) for m in available)
        except Exception:
            return False

    def call(self, prompt: str) -> str:
        response = self._client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0, "num_predict": 1024},
        )
        return response.message.content or ""


class _TransformersBackend:
    """Qwen via HuggingFace transformers with optional 4-bit quantization."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8B-Instruct",
        use_4bit: bool = True,
        device: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.use_4bit = use_4bit
        self.device = device
        self._pipeline = None  # lazy load

    @property
    def name(self) -> str:
        quant = "-4bit" if self.use_4bit else ""
        return f"transformers/{self.model_name}{quant}"

    def is_available(self) -> bool:
        try:
            import transformers  # noqa: F401
            return True
        except ImportError:
            return False

    def _load(self) -> None:
        if self._pipeline is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

        logger.info("Loading %s (this takes ~30 s on first call)…", self.model_name)

        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

        if self.use_4bit:
            try:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=bnb_config,
                    device_map=self.device,
                    trust_remote_code=True,
                )
            except Exception as exc:
                logger.warning("4-bit quantization failed (%s); falling back to fp16", exc)
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map=self.device,
                    trust_remote_code=True,
                )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map=self.device,
                trust_remote_code=True,
            )

        self._pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=1024,
            do_sample=False,         # greedy — consistent JSON output
            temperature=1.0,         # ignored when do_sample=False, avoids warning
            pad_token_id=tokenizer.eos_token_id,
        )

    def call(self, prompt: str) -> str:
        self._load()
        # Qwen uses a chat template; wrap prompt as user message
        messages = [{"role": "user", "content": prompt}]
        tokenizer = self._pipeline.tokenizer
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        outputs = self._pipeline(formatted)
        generated = outputs[0]["generated_text"]
        # Strip the input prefix from the output
        if generated.startswith(formatted):
            generated = generated[len(formatted):]
        return generated.strip()


class _MockBackend:
    """Returns empty extractions — for testing pipeline wiring without a GPU."""

    @property
    def name(self) -> str:
        return "mock"

    def is_available(self) -> bool:
        return True

    def call(self, prompt: str) -> str:
        # Return a minimal valid JSON so the parser has something to process
        return json.dumps({"products": [], "attributes": [], "relations": []})


# ---------------------------------------------------------------------------
# LLMExtractor — public entry point
# ---------------------------------------------------------------------------

BackendType = Literal["ollama", "transformers", "mock"]

# Default model names
_DEFAULT_OLLAMA_MODEL = "qwen3:8b"
_DEFAULT_HF_MODEL = "Qwen/Qwen3-8B-Instruct"


class LLMExtractor:
    """Orchestrates LLM calls and response parsing for one or many chunks.

    Parameters
    ----------
    backend
        ``"ollama"`` | ``"transformers"`` | ``"mock"`` | ``None`` (auto-detect).
    model
        Override the model name/tag.
        - For Ollama: e.g. ``"qwen3:8b"``, ``"qwen3:14b"``, ``"llama3.1:8b"``
        - For transformers: HuggingFace model ID, e.g. ``"Qwen/Qwen3-8B-Instruct"``
    ollama_host
        Override Ollama host URL (default: ``http://localhost:11434``).
    use_4bit
        Enable 4-bit quantization for the transformers backend (default: True).
    """

    def __init__(
        self,
        backend: Optional[BackendType] = None,
        model: Optional[str] = None,
        ollama_host: Optional[str] = None,
        use_4bit: bool = True,
    ) -> None:
        self._backend = self._resolve_backend(backend, model, ollama_host, use_4bit)
        logger.info("LLMExtractor: using backend %r", self._backend.name)

    @property
    def backend_name(self) -> str:
        return self._backend.name

    # ------------------------------------------------------------------
    # Core extraction
    # ------------------------------------------------------------------

    def extract(self, llm_input: Dict[str, Any]) -> ExtractionResult:
        """Run extraction for one LLM input dict (from llm_input_builder).

        Parameters
        ----------
        llm_input
            Dict produced by ``build_llm_input()``.  Must have keys:
            ``chunk_id``, ``doc_stem``, ``section_path``, ``prompt``.

        Returns
        -------
        ExtractionResult
            Always returns a result — parse errors are captured in
            ``result.parse_error`` rather than raised.
        """
        chunk_id = llm_input.get("chunk_id", "unknown")
        doc_stem = llm_input.get("doc_stem", "unknown")
        section_path = llm_input.get("section_path", [])
        prompt = llm_input.get("prompt", "")

        if not prompt:
            result = ExtractionResult(
                chunk_id=chunk_id, doc_stem=doc_stem, section_path=section_path
            )
            result.parse_error = "Empty prompt — skipped"
            result.backend_used = self._backend.name
            return result

        t0 = time.monotonic()
        try:
            raw_response = self._backend.call(prompt)
        except Exception as exc:
            result = ExtractionResult(
                chunk_id=chunk_id, doc_stem=doc_stem, section_path=section_path
            )
            result.parse_error = f"Backend call failed: {exc}"
            result.backend_used = self._backend.name
            result.latency_s = time.monotonic() - t0
            logger.error("LLM call failed for chunk %s: %s", chunk_id, exc)
            return result

        latency = time.monotonic() - t0

        result = _parse_llm_response(raw_response, chunk_id, doc_stem, section_path)
        result.backend_used = self._backend.name
        result.latency_s = latency
        return result

    def extract_batch(
        self,
        llm_inputs: List[Dict[str, Any]],
        progress: bool = True,
    ) -> List[ExtractionResult]:
        """Extract from a list of LLM input dicts sequentially.

        Parameters
        ----------
        llm_inputs
            List of dicts from ``build_llm_inputs_for_chunks()``.
        progress
            Print a progress line for each chunk (default: True).
        """
        results: List[ExtractionResult] = []
        total = len(llm_inputs)
        for i, llm_input in enumerate(llm_inputs, start=1):
            chunk_id = llm_input.get("chunk_id", f"chunk-{i}")
            if progress:
                print(f"  [{i}/{total}] extracting {chunk_id} …", end=" ", flush=True)
            result = self.extract(llm_input)
            if progress:
                status = (
                    f"✓  products={len(result.products)} "
                    f"attrs={len(result.attributes)} "
                    f"relations={len(result.relations)} "
                    f"({result.latency_s:.1f}s)"
                    if not result.parse_error
                    else f"✗  {result.parse_error}"
                )
                print(status)
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Backend resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_backend(
        backend: Optional[str],
        model: Optional[str],
        ollama_host: Optional[str],
        use_4bit: bool,
    ):
        if backend == "mock":
            return _MockBackend()

        if backend == "ollama" or backend is None:
            ollama_model = model or _DEFAULT_OLLAMA_MODEL
            try:
                b = _OllamaBackend(model=ollama_model, host=ollama_host)
                if b.is_available():
                    return b
                elif backend == "ollama":
                    # Explicitly requested but unavailable — still return it
                    # so the error surfaces on the first .call(), not here
                    logger.warning(
                        "Ollama requested but model %r not found. "
                        "Run: ollama pull %s",
                        ollama_model,
                        ollama_model,
                    )
                    return b
                else:
                    logger.info("Ollama not available, trying transformers backend…")
            except ImportError:
                if backend == "ollama":
                    raise
                logger.info("ollama package not installed, trying transformers backend…")

        if backend == "transformers" or backend is None:
            hf_model = model or _DEFAULT_HF_MODEL
            b = _TransformersBackend(model_name=hf_model, use_4bit=use_4bit)
            if b.is_available():
                return b
            elif backend == "transformers":
                raise ImportError(
                    "transformers package not installed. "
                    "Run: pip install transformers>=4.40 accelerate bitsandbytes"
                )
            else:
                logger.warning(
                    "Neither Ollama nor transformers available. "
                    "Falling back to mock backend — no real extraction will happen."
                )

        return _MockBackend()
