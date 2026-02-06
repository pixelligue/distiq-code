# distiq-code — экономия подписки Claude Code в 3-5 раз

Выпустил open-source прокси для Claude Code, который растягивает лимиты подписки через умный роутинг и кэширование.

## Как работает

Claude Code по умолчанию использует Opus для всех запросов. distiq-code анализирует каждый промпт и автоматически переключает на дешёвые модели там, где это возможно:

- Поиск файлов, чтение кода, grep → Sonnet (5x дешевле)
- Простые вопросы → Haiku (60x дешевле)
- Архитектура, сложные задачи → Opus

Плюс семантический кэш на похожие вопросы и автоматическое использование Anthropic Prompt Caching (90% скидка на повторяющийся контекст).

## Реальная экономия

Claude Code Pro (500 запросов/месяц):
- Без прокси: живёте 30-40% месяца
- С distiq-code: хватает на весь месяц

Claude Code Max (безлимит по запросам, лимит по токенам):
- Без прокси: съедаете лимит за 30% месяца
- С distiq-code: используете только 70% токенов за весь месяц

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
