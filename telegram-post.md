# distiq-code — экономия подписки Claude Code в 3-5 раз

Выпустил open-source прокси для Claude Code, который растягивает лимиты подписки через умный роутинг и кэширование.

## Как работает

Claude Code по умолчанию использует Opus для всех запросов. distiq-code анализирует каждый промпт и автоматически переключает на дешёвые модели там, где это возможно:

- Поиск файлов, чтение кода, grep → Sonnet (5x дешевле)
- Простые вопросы → Haiku (60x дешевле)
- Архитектура, сложные задачи → Opus

Плюс семантический кэш на похожие вопросы и автоматическое использование Anthropic Prompt Caching (90% скидка на повторяющийся контекст).

## Реальная экономия

Claude Code Pro (~45 сообщений за 5-часовую сессию):
- Без прокси: упираетесь в лимит через 2-3 часа активной работы
- С distiq-code: используете всю 5-часовую сессию полностью

Claude Code Max 5x (~225 сообщений/сессия, $100/мес):
- Без прокси: лимит заканчивается через 3-4 часа
- С distiq-code: работаете все 5 часов без упора в лимит

Claude Code Max 20x (~900 сообщений/сессия, $200/мес):
- Без прокси: лимит расходуется на 70%
- С distiq-code: используете только 30-40% лимита за сессию

## Установка

```bash
pip install distiq-code[all]
distiq-code setup
distiq-code start
```

В другом терминале:
```bash
claude
```

Всё. Прокси работает прозрачно, никаких изменений в рабочем процессе.

## Технологии

- Smart routing: regex + BERT K-NN (75 примеров RU+EN)
- Semantic cache: EmbeddingGemma-300M + FAISS
- Prompt compression: LLMLingua-2 BERT-base (опционально)
- Prompt caching: автоматическая инъекция cache_control
- Tool-use optimization: все Read/Grep/Bash → Sonnet вместо Opus

## Код

https://github.com/pixelligue/distiq-code

MIT лицензия, Python 3.11+
