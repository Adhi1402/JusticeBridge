"""
Pluggable LLM client — the ONE place that talks to a language model.

Primary path is Qualcomm's own on-device stack, not a generic cloud API:

    JB_LLM_BACKEND=geniex     -> GenieX (github.com/qualcomm/GenieX), a Python
                                  wrapper over QAIRT/Genie. On the Snapdragon
                                  AI PC it loads a Qualcomm AI Hub NPU bundle;
                                  the same call also runs a GGUF model via
                                  GenieX's llama.cpp backend, so the identical
                                  code path works for local dev on any machine
                                  with an NPU or GPU. NOTE: verified on this
                                  x64 dev box that `pip install geniex` fails
                                  to build ("Unsupported platform ('win32',
                                  'amd64')") — its native SDK only ships
                                  prebuilt binaries for win32/arm64 and
                                  linux/aarch64. That's expected: it's gated to
                                  real Snapdragon silicon by design, so this
                                  backend can only be exercised on the actual
                                  hackathon hardware, not here.
    JB_LLM_BACKEND=onnx_qnn   -> onnxruntime-genai + the QNN execution
                                  provider, running a Qualcomm AI Hub-exported
                                  genai_config.json bundle directly on the
                                  Hexagon NPU. Lower-level Python-native
                                  alternative to GenieX for the same hardware.
    JB_LLM_BACKEND=openai     -> any OpenAI-compatible /v1 endpoint (e.g. a
                                  llama.cpp server a teammate stood up
                                  separately). Dev-machine convenience only.
    JB_LLM_BACKEND=extractive -> no model at all; reasoning_agent.py builds the
                                  answer directly from retrieved statute text.
                                  Zero hallucination, always available. This is
                                  what actually runs end-to-end on this x64 dev
                                  box, since geniex/onnx_qnn need real
                                  Snapdragon NPU hardware.

Every backend raises LLMUnavailable on any failure (missing package, wrong
platform, missing model files, generation error, network down, ...) so
reasoning_agent.py's fallback to the extractive path is unconditional — the
pipeline never hard-fails on stage regardless of which backend is configured.

`chat()` returns a string. `chat_json()` asks for and parses a JSON object,
retrying once with a stricter instruction — used by the reasoning agent to get
claim→section mappings the grounding agent can verify.
"""

import json
import re
import requests

from . import config


class LLMUnavailable(RuntimeError):
    """Raised when the configured backend can't be reached — callers degrade."""


# ---------------------------------------------------------------------------
# GenieX (primary on-device backend — QAIRT/Genie under the hood)
# ---------------------------------------------------------------------------
_geniex_model = None


def _get_geniex_model():
    global _geniex_model
    if _geniex_model is None:
        try:
            from geniex import AutoModelForCausalLM
        except ImportError as e:
            raise LLMUnavailable(
                f"geniex not installed or unsupported on this platform: {e}"
            )
        try:
            kwargs = {}
            # precision only applies to GGUF (llama.cpp-backend) models, not
            # pre-compiled Qualcomm AI Hub NPU bundles.
            if "GGUF" in config.GENIEX_MODEL.upper() or "/" in config.GENIEX_MODEL \
                    and not config.GENIEX_MODEL.startswith("ai-hub-models/"):
                kwargs["precision"] = config.GENIEX_PRECISION
            _geniex_model = AutoModelForCausalLM.from_pretrained(config.GENIEX_MODEL, **kwargs)
        except Exception as e:
            raise LLMUnavailable(f"geniex failed to load '{config.GENIEX_MODEL}': {e}")
    return _geniex_model


def _geniex_chat(system: str, user: str, temperature: float) -> str:
    model = _get_geniex_model()
    try:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = model.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        result = model.generate(
            prompt,
            max_new_tokens=config.GENIEX_MAX_NEW_TOKENS,
            temperature=temperature,
            stream=False,
        )
        # Defensive: some GenieX versions may return a generator even with
        # stream=False depending on backend; handle both shapes.
        text = result if isinstance(result, str) else "".join(chunk for chunk in result)
        return text.strip()
    except Exception as e:
        raise LLMUnavailable(f"geniex generation failed: {e}")


# ---------------------------------------------------------------------------
# onnxruntime-genai + QNN execution provider (alternative on-device backend)
# ---------------------------------------------------------------------------
_onnx_qnn_model = None
_onnx_qnn_tokenizer = None


def _get_onnx_qnn_model():
    global _onnx_qnn_model, _onnx_qnn_tokenizer
    if _onnx_qnn_model is None:
        if not config.ONNX_QNN_MODEL_DIR:
            raise LLMUnavailable(
                "JB_ONNX_QNN_MODEL_DIR not set — point it at a Qualcomm AI Hub "
                "genai_config.json bundle directory"
            )
        try:
            import onnxruntime_genai as og
        except ImportError as e:
            raise LLMUnavailable(f"onnxruntime-genai not installed: {e}")
        try:
            og_config = og.Config(config.ONNX_QNN_MODEL_DIR)
            _onnx_qnn_model = og.Model(og_config)
            _onnx_qnn_tokenizer = og.Tokenizer(_onnx_qnn_model)
        except Exception as e:
            raise LLMUnavailable(
                f"onnxruntime-genai failed to load '{config.ONNX_QNN_MODEL_DIR}' "
                f"(is the QNN execution provider / NPU driver available?): {e}"
            )
    return _onnx_qnn_model, _onnx_qnn_tokenizer


def _onnx_qnn_chat(system: str, user: str, temperature: float) -> str:
    import onnxruntime_genai as og

    model, tokenizer = _get_onnx_qnn_model()
    try:
        prompt = f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"
        input_tokens = tokenizer.encode(prompt)

        params = og.GeneratorParams(model)
        params.set_search_options(
            max_length=config.ONNX_QNN_MAX_LENGTH, temperature=max(temperature, 0.01)
        )
        generator = og.Generator(model, params)
        generator.append_tokens(input_tokens)

        stream = tokenizer.create_stream()
        chunks = []
        while not generator.is_done():
            generator.generate_next_token()
            token = generator.get_next_tokens()[0]
            chunks.append(stream.decode(token))
        del generator
        return "".join(chunks).strip()
    except Exception as e:
        raise LLMUnavailable(f"onnxruntime-genai generation failed: {e}")


# ---------------------------------------------------------------------------
# OpenAI-compatible endpoint (dev-machine convenience, e.g. a llama.cpp server)
# ---------------------------------------------------------------------------
def _openai_chat(system: str, user: str, temperature: float) -> str:
    url = f"{config.OPENAI_BASE_URL.rstrip('/')}/chat/completions"
    payload = {
        "model": config.OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "stream": False,
    }
    headers = {"Authorization": f"Bearer {config.OPENAI_API_KEY}"}
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=config.LLM_TIMEOUT)
    except requests.RequestException as e:
        raise LLMUnavailable(f"openai-compatible endpoint unreachable: {e}")
    if r.status_code != 200:
        raise LLMUnavailable(f"openai http {r.status_code}: {r.text[:200]}")
    data = r.json()
    return data["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
def chat(system: str, user: str, temperature: float = 0.2) -> str:
    """Single-turn chat. Raises LLMUnavailable if the backend can't answer."""
    backend = config.LLM_BACKEND
    if backend == "geniex":
        return _geniex_chat(system, user, temperature)
    if backend == "onnx_qnn":
        return _onnx_qnn_chat(system, user, temperature)
    if backend == "openai":
        return _openai_chat(system, user, temperature)
    # "extractive" (or anything unknown): there is no model to call.
    raise LLMUnavailable(f"backend '{backend}' has no live model")


def _extract_json(text: str):
    """Pull the first JSON object out of a model response (models love to wrap
    JSON in prose or ```json fences)."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        candidate = brace.group(0) if brace else None
    if candidate is None:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def chat_json(system: str, user: str, temperature: float = 0.1):
    """Chat expecting a JSON object back. Returns a dict, or raises
    LLMUnavailable if the backend is down. Retries once on unparseable output."""
    raw = chat(system, user, temperature)
    obj = _extract_json(raw)
    if obj is not None:
        return obj
    # one stricter retry
    raw = chat(
        system,
        user + "\n\nRespond with ONLY a single valid JSON object. No prose, no code fences.",
        temperature,
    )
    obj = _extract_json(raw)
    if obj is None:
        raise LLMUnavailable(f"model did not return parseable JSON: {raw[:200]}")
    return obj


def is_live() -> bool:
    """Cheap probe used by the CLI/UI to show which backend is actually active."""
    if config.LLM_BACKEND == "extractive":
        return False
    try:
        chat("You are a health check.", "Reply with the single word: OK", 0.0)
        return True
    except Exception:
        return False
