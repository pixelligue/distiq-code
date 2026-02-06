# distiq-code

> Экономьте 3-5x на подписке Claude Code с помощью умного роутинга и кэширования.

**distiq-code** — локальный прокси между Claude Code и Anthropic API. Автоматически направляет простые задачи на дешёвые модели и кэширует повторные запросы — ваша подписка Claude Max за $200/мес живёт в разы дольше.

```
Claude Code → localhost:11434 → [Роутинг / Кэш] → api.anthropic.com
```

## Как это работает

**Умный роутинг** — большинство запросов Claude Code не требуют Opus. distiq-code анализирует каждый промпт и направляет его на самую дешёвую модель, которая справится:

| Задача | Модель | Стоимость |
|--------|--------|-----------|
| Архитектура, системный дизайн | Opus | $15/1M токенов |
| Генерация кода, отладка | Sonnet | $3/1M (в 5 раз дешевле) |
| Простые вопросы, объяснения | Haiku | $0.25/1M (в 60 раз дешевле) |

**Семантический кэш** — похожие вопросы получают мгновенный ответ из локального FAISS-кэша вместо повторного обращения к API.

**Anthropic Prompt Caching** — автоматически добавляет `cache_control` breakpoints, чтобы повторяющийся контекст (tools, system prompt) получал 90% скидку от Anthropic.

**Live-статистика** — видно что происходит на каждом запросе:
```
[proxy] sonnet (from opus) | 6.1K in / 795 out | $0.0303 | saved $0.1210 | 20.3s | session saved: $0.134
[proxy] CACHE HIT (sim=0.95) | saved 342 tokens (~$0.0308) | session saved: $0.165 | 36ms
```

## Быстрый старт

### 1. Установка

```bash
pip install distiq-code
```

Или из исходников:

```bash
git clone https://github.com/pixelligue/distiq-code.git
cd distiq-code
pip install -e .
```

Опциональные ML-фичи (семантический кэш + BERT-роутинг):

```bash
pip install distiq-code[ml]          # FAISS + sentence-transformers
pip install distiq-code[compression] # LLMLingua-2 сжатие промптов
pip install distiq-code[all]         # Всё вместе
```

### 2. Настройка

```bash
distiq-code setup
```

Что произойдёт:
- Проверка установки Claude CLI
- Загрузка ML-моделей (~400 МБ, если установлен `[ml]`)
- Настройка `ANTHROPIC_BASE_URL` для Claude Code

### 3. Запуск прокси

```bash
distiq-code start
```

### 4. Использование Claude Code

В другом терминале:

```bash
claude
```

Всё. Все запросы теперь идут через прокси автоматически.

## Возможности

### Умный роутинг моделей

Два бэкенда роутинга:

- **ML-роутер** (по умолчанию с `[ml]`) — BERT-классификатор на основе K-NN по 75 примерам. ~5мс на запрос.
- **Regex-роутер** (фоллбэк) — паттерн-матчинг для запросов на RU + EN. Без зависимостей.

Роутинг только **понижает** — если Claude Code запрашивает Opus, а задача простая, она уходит на Sonnet. Никогда не повышает.

### Семантический кэш

Использует FAISS + sentence-transformers (`all-mpnet-base-v2`) для поиска похожих запросов:

```
Запрос 1: "Как создать React компонент?"
Запрос 2: "Как сделать компонент в React?" → Cache hit (sim=0.94)
```

- Настраиваемый порог схожести (по умолчанию: 0.85)
- TTL 7 дней
- До 10 000 кэшированных записей
- Tool-use разговоры никогда не кэшируются (устаревшие результаты)

### Anthropic Prompt Caching

Автоматически добавляет `cache_control` breakpoints к:
1. Последнему определению инструмента (tool)
2. Системному промпту
3. Последнему сообщению пользователя

Закэшированные токены получают 90% скидку на input. Настройка не требуется.

### Сжатие промптов (опционально)

С `pip install distiq-code[compression]`:

- LLMLingua-2 сжимает контекст разговора до 5x
- Последний запрос пользователя никогда не сжимается
- Ключевые слова кода (`def`, `class`, `import` и т.д.) сохраняются

## CLI-команды

```bash
distiq-code start    # Запустить прокси-сервер
distiq-code setup    # Одноразовая настройка (модели + env)
distiq-code stats    # Показать статистику использования
distiq-code config   # Показать текущую конфигурацию
distiq-code chat     # Интерактивный чат (опционально)
distiq-code version  # Показать версию
```

### Статистика

```bash
distiq-code stats --period week

# Вывод:
# Requests: 347
# Cache hits: 72%
# Tokens saved: 1,084,000
# Cost saved: $12.40
```

## Конфигурация

Все настройки через переменные окружения или `.env` файл:

```bash
# Сервер
PROXY_HOST=127.0.0.1
PROXY_PORT=11434

# Роутинг
SMART_ROUTING=true          # Умный роутинг моделей
ML_ROUTING_ENABLED=true     # BERT-роутер (требует [ml])

# Кэширование
CACHE_ENABLED=true
CACHE_TTL_HOURS=168                  # 7 дней
CACHE_SIMILARITY_THRESHOLD=0.85

# Anthropic Prompt Caching
PROMPT_CACHING_ENABLED=true          # Инъекция cache_control breakpoints

# Сжатие
COMPRESSION_ENABLED=true
COMPRESSION_TARGET_TOKENS=500

# Отладка
DEBUG=false
LOG_LEVEL=INFO
```

## Архитектура

```
src/distiq_code/
├── cli.py                 # Typer CLI + чат REPL
├── config.py              # Pydantic Settings
├── routing.py             # Умный роутинг (regex + embedding)
├── embedding_router.py    # K-NN embedding-роутер
├── clipboard.py           # Вставка скриншотов
├── auth/
│   └── cli_provider.py    # Claude CLI subprocess (OAuth)
├── cache/
│   └── semantic_cache.py  # FAISS + sentence-transformers
├── compression/
│   └── compressor.py      # LLMLingua-2
├── stats/
│   └── tracker.py         # Метрики + трекинг стоимости
└── server/
    ├── app.py             # FastAPI factory
    ├── main.py            # Uvicorn entry point
    └── routes/
        ├── messages.py    # /v1/messages (Anthropic proxy)
        ├── chat.py        # /v1/chat/completions (OpenAI)
        └── health.py      # /health, /ready
```

## Требования

- Python 3.11+
- [Claude CLI](https://docs.anthropic.com/en/docs/claude-code) установлен и авторизован

## Лицензия

MIT — см. [LICENSE](LICENSE).

---

**Сделано в [Distiq](https://distiq.ru)**
