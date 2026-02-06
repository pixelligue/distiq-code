# Distiq-Code v2.0 — Progress & Roadmap

> **Последнее обновление:** 6 февраля 2026, 21:45  
> **Цель:** AI coding assistant с $25/месяц себестоимостью  

---

## 📊 Текущий статус: 6/8 фаз готово

| Фаза | Статус | Описание |
|------|--------|----------|
| Phase 1 | ✅ ГОТОВО | Multi-Provider Infrastructure |
| Phase 2 | ✅ ГОТОВО | Code Indexing & RAG |
| Phase 3 | ✅ ГОТОВО | Orchestrator & Routing |
| Phase 4 | ✅ ГОТОВО | Prompt Optimization |
| Phase 5 | ✅ ГОТОВО | Tools & Skills System |
| Phase 6 | ⏳ ЧАСТИЧНО | Integration & Polish |
| Phase 7 | 🆕 TODO | Advanced Features (NEW) |
| Phase 8 | 🆕 TODO | TUI & LSP (NEW) |

---

## ✅ Готовые компоненты

### Phase 1: Multi-Provider ✅
- [x] BaseProvider interface
- [x] Anthropic Provider (prompt caching)
- [x] OpenAI-compatible provider (DeepSeek, OpenRouter)
- [x] Claude Code SDK integration
- [x] Provider Registry & Factory
- [x] Cost Tracker

### Phase 2: Code Indexing ✅
- [x] Tree-sitter Parser (Python, JavaScript)
- [x] Nomic Embed embeddings (local)
- [x] FAISS Vector Store + SQLite metadata
- [x] Context Builder
- [x] CLI commands: index, search
- [x] **BUG FIX:** exclude patterns now match path parts

### Phase 3: Orchestrator ✅
- [x] Complexity Classifier (simple/medium/complex)
- [x] Planning prompts (Haiku/Sonnet)
- [x] Execution prompts
- [x] Full orchestration pipeline

### Phase 4: Prompt Optimization ✅
- [x] History Summarization (каждые 10 сообщений)
- [x] Compression pipeline
- [x] Context injection

### Phase 5: Tools & Skills ✅
- [x] Tools Registry: Read, Write, Glob, Grep, Bash, Search
- [x] **WebSearch** — Jina AI + Tavily fallback
- [x] **ReadURL** — Jina Reader (бесплатно!)
- [x] Skills system: refactor, add-tests, explain, fix-bug, review, docstrings, optimize

### Phase 6: Integration ⏳
- [x] E2E Tests — **7/7 passed!**
- [x] Environment setup (.env, dependencies)
- [x] OpenRouter integration
- [ ] Live Chat testing (in progress)
- [ ] CLI polish

---

## 🆕 Phase 7: Advanced Features (TODO)

### 7.1 Remote Embeddings — Voyage AI
**Приоритет:** 🔴 Высокий  
**Зачем:** 0MB скачивания при установке, 200M токенов бесплатно!

```
Файл: src/distiq_code/indexing/remote_embedder.py (✅ создан)

TODO:
- [ ] Интегрировать в indexer.py
- [ ] Добавить VOYAGE_API_KEY в .env.example
- [ ] Fallback: Voyage → Jina → Local
- [ ] Тест: проиндексировать проект через Voyage
```

**Pricing:**
| Provider | Free Tier | Цена после |
|----------|-----------|------------|
| Voyage AI | 200M tokens | $0.02/1M |
| Jina AI | 10M tokens | $0.05/1M |
| Local | ∞ | 0 (но 547MB download) |

### 7.2 Agent System (Build/Plan)
**Приоритет:** 🔴 Высокий  
**Зачем:** Как в OpenCode — режимы работы

```
Файл: src/distiq_code/agents/__init__.py (✅ создан)

TODO:
- [ ] Интегрировать в orchestrator
- [ ] Build Agent: полный доступ к tools
- [ ] Plan Agent: read-only режим
- [ ] Переключение Tab или /agent build|plan
- [ ] AGENTS.md генерация при /init
```

**Режимы:**
| Agent | Файлы | Команды | Подтверждение |
|-------|-------|---------|---------------|
| Build | ✅ R/W | ✅ Bash | Auto-approve |
| Plan | ✅ Read | ❌ | Всегда спрашивать |

### 7.3 AGENTS.md Support
**Приоритет:** 🟡 Средний

```
TODO:
- [ ] Парсинг AGENTS.md
- [ ] Custom instructions per agent
- [ ] /init команда создаёт файл
- [ ] Git commit AGENTS.md
```

---

## 🆕 Phase 8: TUI & LSP (TODO)

### 8.1 Textual TUI
**Приоритет:** 🟡 Средний  
**Зачем:** Красивый интерфейс как Ink в OpenCode

```
TODO:
- [ ] pip install textual
- [ ] Заменить Rich console на Textual App
- [ ] Layout: input + output + sidebar
- [ ] Tab переключение агентов
- [ ] Hotkeys: Ctrl+C, Ctrl+L, etc.
- [ ] Animations & progress
```

**Textual features:**
- Flexbox layout (как CSS)
- Reactive state management
- Built-in widgets (buttons, tables, trees)
- Mouse support
- Themes

### 8.2 LSP Integration
**Приоритет:** 🟠 Низкий (сложно)  
**Зачем:** Autocomplete, go-to-definition, hover

```
TODO:
- [ ] Research: python-lsp-server vs pygls
- [ ] Подключение к существующему LSP серверу
- [ ] Go-to-definition для контекста
- [ ] Hover info injection
- [ ] Find references
```

**Ресурсы:**
- https://github.com/python-lsp/python-lsp-server
- https://microsoft.github.io/language-server-protocol/

---

## 📋 Immediate TODO (следующая сессия)

### Высокий приоритет
1. [ ] Протестировать OpenRouter chat
2. [ ] Интегрировать remote_embedder.py
3. [ ] Интегрировать agents/ в orchestrator

### Средний приоритет
4. [ ] Добавить Voyage API key flow
5. [ ] AGENTS.md генерация
6. [ ] /agent команда в CLI

### Низкий приоритет
7. [ ] Textual TUI prototype
8. [ ] LSP research

---

## 🔧 Технические заметки

### API Keys (в .env)
```bash
# LLM
OPENROUTER_API_KEY=sk-or-v1-...

# Embeddings
VOYAGE_API_KEY=...        # 200M free tokens
JINA_API_KEY=jina_...     # 10M free tokens

# Web Search
TAVILY_API_KEY=tvly-...   # 1000/month free
```

### Зависимости для расширений
```bash
# Remote embeddings (выбрать один)
# Voyage/Jina — через httpx (уже установлен)

# TUI
pip install textual

# LSP
pip install python-lsp-server
```

### Benchmarks
| Операция | Время |
|----------|-------|
| Indexing 379 chunks | 93 сек (первый раз) |
| Search | <100ms |
| Web search (Jina) | ~10 сек |
| E2E test suite | ~2.5 мин |

---

## 🏆 Сравнение с OpenCode

| Feature | OpenCode | Distiq-Code |
|---------|----------|-------------|
| Open Source | ✅ | ✅ |
| Code Indexing | ❌ | ✅ |
| Semantic Search | ❌ | ✅ |
| Web Search | ❌ | ✅ |
| Build/Plan Agents | ✅ | ⏳ |
| Textual TUI | ✅ (Ink) | 🔄 (Rich → Textual) |
| LSP Integration | ✅ | 🔄 |
| Client/Server | ✅ | ❌ |
| 0MB Install | ❌ | ⏳ (remote embeddings) |

---

*Последнее обновление: 6 февраля 2026, 21:45*
