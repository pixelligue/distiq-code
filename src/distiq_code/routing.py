"""Smart model routing based on query complexity heuristics (RU + EN).

Supports two routing backends:
  1. ML router (RouteLLM BERT) — if torch + transformers installed
  2. Regex fallback — always available
"""

from __future__ import annotations

import re

# Tier order for bumping
_TIERS = ["haiku", "sonnet", "opus"]

# --- Patterns for each complexity level (English + Russian) ---

_SIMPLE_PATTERNS = re.compile(
    # English
    r"(?:\b(?:what is|what are|explain|translate|rename|fix typo|how to|how do|"
    r"what does|tell me|define|meaning of|difference between|convert|"
    r"show me|list|summarize|clarify)\b"
    r"|"
    # Russian
    r"(?:что такое|что значит|объясни|расскажи|переведи|переименуй|"
    r"как сделать|как работает|как использовать|как называется|"
    r"покажи|перечисли|опиши|уточни|в чём разница|чем отличается|"
    r"подскажи|помоги понять|что означает|зачем нужен|для чего))",
    re.IGNORECASE,
)

_MEDIUM_PATTERNS = re.compile(
    # English
    r"(?:\b(?:write|implement|refactor|debug|fix bug|fix error|add function|"
    r"create class|create function|unit test|add test|modify|update|"
    r"change|optimize|improve|handle|add endpoint|add route|parse|"
    r"validate|generate|build|code|search|find|look for|analyze|review|"
    r"check|scan|inspect|examine|read|show|compare|count)\b"
    r"|"
    # Russian
    r"(?:напиши|реализуй|добавь|создай|исправь|почини|отрефактори|"
    r"оптимизируй|улучши|измени|обнови|модифицируй|сгенерируй|"
    r"допиши|перепиши функцию|добавь тест|напиши тест|"
    r"добавь эндпоинт|добавь роут|сделай парсер|"
    r"валидируй|собери|построй|закодь|пофикси|отладь|дебагни|"
    r"поищи|найди|ищи|проанализируй|анализируй|проверь|посмотри|"
    r"покажи|сравни|просканируй|прочитай|открой|глянь|загугли))",
    re.IGNORECASE,
)

_COMPLEX_PATTERNS = re.compile(
    # English
    r"(?:\b(?:architect|architecture|design|multi.?file|migrate|migration|"
    r"rewrite|complex|system design|microservice|refactor.*(entire|whole|all)|"
    r"redesign|overhaul|infrastructure|scalab|distributed|"
    r"database schema|full.?stack|end.?to.?end|pipeline)\b"
    r"|"
    # Russian
    r"(?:спроектируй|архитектур|дизайн системы|микросервис|"
    r"перепиши (всё|весь|целиком|полностью)|миграция|мигрируй|"
    r"редизайн|переделай (всё|весь|целиком)|инфраструктур|"
    r"масштабируем|распределённ|схема базы|база данных|"
    r"полный стек|от начала до конца|пайплайн|конвейер|"
    r"перестрой|спланируй|продумай архитектуру))",
    re.IGNORECASE,
)

# Model pricing per 1M tokens (input)
MODEL_INPUT_PRICE = {"haiku": 0.25, "sonnet": 3.00, "opus": 15.00}
MODEL_OUTPUT_PRICE = {"haiku": 1.25, "sonnet": 15.00, "opus": 75.00}

# --- Embedding Router (lazy singleton) ---

_embedding_router = None
_embedding_router_loaded = False


def _get_embedding_router():
    """Lazy-load EmbeddingRouter. Returns None on failure."""
    global _embedding_router, _embedding_router_loaded
    if _embedding_router_loaded:
        return _embedding_router
    _embedding_router_loaded = True
    try:
        from distiq_code.embedding_router import EmbeddingRouter
        _embedding_router = EmbeddingRouter()
    except Exception:
        _embedding_router = None
    return _embedding_router


def load_embedding_router(model=None) -> bool:
    """Eagerly load embedding router. Returns True if loaded successfully.

    Args:
        model: Optional pre-loaded SentenceTransformer (reuse from cache).
    """
    global _embedding_router, _embedding_router_loaded
    if _embedding_router_loaded:
        return _embedding_router is not None
    _embedding_router_loaded = True
    try:
        from distiq_code.embedding_router import EmbeddingRouter
        _embedding_router = EmbeddingRouter(model=model)
    except Exception:
        _embedding_router = None
    return _embedding_router is not None


def is_ml_routing_active() -> bool:
    """Check if embedding routing is active (loaded and enabled in config)."""
    from distiq_code.config import settings
    return settings.ml_routing_enabled and _embedding_router is not None


def _bump_tier(model: str) -> str:
    """Bump model one tier up (haiku->sonnet, sonnet->opus)."""
    try:
        idx = _TIERS.index(model)
        return _TIERS[min(idx + 1, len(_TIERS) - 1)]
    except ValueError:
        return model


def estimate_savings(
    routed_model: str,
    default_model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Estimate how much money was saved by routing to a cheaper model.

    Returns savings in USD (positive = saved, 0 = no savings).
    """
    if routed_model == default_model:
        return 0.0

    default_cost = (
        input_tokens * MODEL_INPUT_PRICE.get(default_model, 15.0)
        + output_tokens * MODEL_OUTPUT_PRICE.get(default_model, 75.0)
    ) / 1_000_000

    routed_cost = (
        input_tokens * MODEL_INPUT_PRICE.get(routed_model, 15.0)
        + output_tokens * MODEL_OUTPUT_PRICE.get(routed_model, 75.0)
    ) / 1_000_000

    return max(0.0, default_cost - routed_cost)


def classify_and_route(
    user_input: str, message_count: int, default_model: str
) -> tuple[str, str]:
    """Return (model, complexity) based on ML router or regex fallback.

    Pipeline:
    1. Try ML router (BERT) if available and enabled
    2. Fallback to regex heuristics
    3. Long conversation (> 6 messages) -> bump up one tier
    4. No signal -> default_model
    """
    from distiq_code.config import settings

    model = ""
    complexity = ""

    # --- Embedding Router ---
    if settings.ml_routing_enabled and _embedding_router is not None:
        try:
            model, complexity, _confidence = _embedding_router.route(user_input)
        except Exception:
            model = ""
            complexity = ""

    # --- Regex fallback ---
    if not complexity:
        words = user_input.split()
        word_count = len(words)

        if word_count > 150 or _COMPLEX_PATTERNS.search(user_input):
            complexity = "complex"
            model = "opus"
        elif _MEDIUM_PATTERNS.search(user_input):
            complexity = "medium"
            model = "sonnet"
        elif word_count < 30 and _SIMPLE_PATTERNS.search(user_input):
            complexity = "simple"
            model = "haiku"
        else:
            # Default: sonnet for everything that doesn't match complex.
            # Sonnet handles most coding tasks well and is 5x cheaper than opus.
            complexity = "default"
            model = "sonnet"

    # Long conversation bump: > 6 messages means context is heavy
    if message_count > 6:
        model = _bump_tier(model)

    return model, complexity
