"""Embedding-based model router using sentence-transformers.

Uses K-NN voting over reference examples for each complexity tier.
Reuses the same all-mpnet-base-v2 model loaded for semantic cache.
Zero extra dependencies, ~5ms per query.
"""

from __future__ import annotations

from collections import Counter

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

# --- Reference examples for each tier (EN + RU) ---

_HAIKU_EXAMPLES = [
    # English - simple questions, explanations, short tasks
    "what is python?",
    "what is a list in javascript?",
    "explain what a variable is",
    "how do I print hello world?",
    "what does this error mean?",
    "what is the difference between let and const?",
    "how to install numpy",
    "show me an example of a for loop",
    "what is REST API?",
    "convert this string to integer",
    "rename this variable to camelCase",
    "fix this typo in the comment",
    "what does async mean?",
    "how to import a module in python",
    "list all HTTP status codes",
    # Short / conversational
    "hi",
    "hello",
    "thanks",
    "yes",
    "no",
    "ok",
    "got it",
    "what is 2+2?",
    "summarize this",
    "clarify this point",
    # Russian
    "что такое python?",
    "объясни что такое переменная",
    "как установить numpy",
    "покажи пример цикла for",
    "в чём разница между let и const?",
    "что значит эта ошибка?",
    "как импортировать модуль",
    "переведи этот код на python",
    "что такое REST API?",
    "подскажи как вывести hello world",
]

_SONNET_EXAMPLES = [
    # English - code generation, debugging, moderate tasks
    "write a binary search function in python",
    "implement a linked list with insert and delete",
    "fix this bug in my async handler",
    "add unit tests for the user service",
    "refactor this function to use async/await",
    "create a REST API endpoint for user registration",
    "write a parser for CSV files with custom delimiter",
    "add input validation to this form component",
    "optimize this SQL query with proper indexing",
    "implement JWT authentication middleware",
    "debug this race condition in the worker pool",
    "write a React component for a data table with sorting",
    "create a Python decorator for rate limiting",
    "add error handling to this API client",
    "implement a WebSocket chat handler",
    "implement a LRU cache class in Python",
    "write a function to parse JSON config files",
    "fix this bug in my code where the loop never exits",
    "create a Flask REST API with CRUD endpoints for users",
    "add pagination to this database query",
    "fix this bug in my code where the API returns 500",
    "implement a cache class with TTL expiration",
    "write a function to validate email addresses",
    "create a middleware that logs all requests",
    "implement a simple key-value store in memory",
    # Russian
    "напиши функцию бинарного поиска",
    "реализуй связный список",
    "исправь баг в асинхронном обработчике",
    "добавь тесты для сервиса пользователей",
    "отрефактори эту функцию на async/await",
    "создай эндпоинт регистрации пользователя",
    "напиши парсер CSV файлов",
    "добавь валидацию формы",
    "оптимизируй этот SQL запрос",
    "пофикси этот TypeError в хендлере",
]

_OPUS_EXAMPLES = [
    # English - architecture, system design, complex multi-file tasks
    "design a microservice authentication system with OAuth2 and JWT",
    "architect a distributed caching layer with Redis cluster",
    "plan migration from monolith to microservices with zero downtime",
    "design database schema for multi-tenant SaaS with sharding strategy",
    "rewrite the entire backend to event-driven architecture",
    "design a CI/CD pipeline with blue-green deployment",
    "architect a real-time notification system with message queues",
    "design a plugin system with dynamic loading and dependency injection",
    "plan the full-stack architecture for an e-commerce platform",
    "design a distributed task queue with retry logic and dead letter queues",
    "architect a data pipeline for real-time analytics with Kafka",
    "design an API gateway with rate limiting, circuit breaker and service mesh",
    "redesign the entire frontend with micro-frontends architecture",
    "plan infrastructure for handling 100k concurrent WebSocket connections",
    "design a multi-region database replication strategy with conflict resolution",
    # Russian
    "спроектируй систему аутентификации микросервисов с OAuth2",
    "продумай архитектуру распределённого кэширования",
    "спланируй миграцию с монолита на микросервисы",
    "спроектируй схему БД для мультитенантного SaaS",
    "перепиши весь бэкенд на событийную архитектуру",
    "спроектируй CI/CD пайплайн с blue-green деплоем",
    "продумай архитектуру системы уведомлений реального времени",
    "спроектируй API gateway с rate limiting и circuit breaker",
    "переделай весь фронтенд на micro-frontends",
    "спланируй инфраструктуру для 100к одновременных соединений",
]

TIER_COMPLEXITY = {
    "haiku": "simple",
    "sonnet": "medium",
    "opus": "complex",
}

_K = 5  # Number of nearest neighbors for voting


class EmbeddingRouter:
    """Route prompts to models using K-NN voting over reference examples."""

    def __init__(self, model=None):
        """Initialize router.

        Args:
            model: Pre-loaded SentenceTransformer instance (reuse from cache).
                   If None, loads a new instance.

        Raises:
            ImportError: If numpy or sentence-transformers are not installed.
        """
        if np is None:
            raise ImportError(
                "ML routing requires numpy. Install with: pip install distiq-code[ml]"
            )
        if model is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        self._model = model
        self._embeddings, self._labels = self._build_index()

    def _build_index(self) -> tuple[np.ndarray, list[str]]:
        """Pre-compute normalized embeddings for all reference examples."""
        all_texts: list[str] = []
        labels: list[str] = []
        for tier, examples in [
            ("haiku", _HAIKU_EXAMPLES),
            ("sonnet", _SONNET_EXAMPLES),
            ("opus", _OPUS_EXAMPLES),
        ]:
            all_texts.extend(examples)
            labels.extend([tier] * len(examples))

        embeddings = self._model.encode(all_texts, normalize_embeddings=True)
        return embeddings, labels

    def route(self, prompt: str) -> tuple[str, str, float]:
        """Route prompt to best model via K-NN majority vote.

        Returns:
            (model, complexity, confidence) where confidence is the
            fraction of K neighbors that voted for the winning tier.
        """
        embedding = self._model.encode([prompt], normalize_embeddings=True)[0]

        # Cosine similarities (dot product of normalized vectors)
        sims = self._embeddings @ embedding
        top_k_idx = np.argpartition(sims, -_K)[-_K:]
        top_k_idx = top_k_idx[np.argsort(sims[top_k_idx])[::-1]]

        # Majority vote
        votes = [self._labels[i] for i in top_k_idx]
        counter = Counter(votes)
        best_tier, best_count = counter.most_common(1)[0]
        confidence = best_count / _K

        return best_tier, TIER_COMPLEXITY[best_tier], confidence
