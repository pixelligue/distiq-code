# Cursor Models Guide (Feb 2026)

## Бесплатные модели (FREE)

| Модель | Характеристики | Когда использовать |
|--------|----------------|-------------------|
| **cursor-small** | Unlimited, быстрая | Простые вопросы, объяснения |
| **deepseek-v3.1** | Unlimited, 97% accuracy | Чтение кода, tool-use, поиск |
| **gpt-4o-mini** | 500 req/day | Ограниченное использование |
| **gemini-2.5-flash** | $1.40/1M (~free) | Быстрые ответы |

## Платные модели

### Для генерации кода (лучшие)

| Модель | Цена/1M | Качество | Рекомендация |
|--------|---------|----------|--------------|
| **gpt-5.1-codex** | $0.75 | ⭐⭐⭐⭐ | Лучший для кода, дешевый |
| **claude-4.5-sonnet** | $9.00 | ⭐⭐⭐⭐⭐ | Высокое качество |
| **gemini-3-flash** | $1.75 | ⭐⭐⭐ | Быстро, дешево |
| **claude-4.5-haiku** | $3.00 | ⭐⭐⭐ | Компромисс |

### Дорогие (не рекомендуется)

| Модель | Цена/1M | Когда оправдано |
|--------|---------|-----------------|
| **claude-4.6-opus** | $15.00 | Очень сложные задачи |
| **gpt-5.2** | $7.88 | Reasoning tasks |
| **gemini-3-pro** | $7.00 | Большой контекст |

---

## Правила роутинга distiq-code

### Простые вопросы → cursor-small (FREE)
```
"Что такое React?"
"Объясни async/await"
"Как работает замыкание?"
```
**Экономия:** 100%

### Чтение кода → deepseek-v3.1 (FREE)
```
"Прочитай файл utils.py"
"Найди все TODO"
"Что делает эта функция?"
```
**Экономия:** 100%

### Генерация кода → claude-4.5-sonnet / gpt-5.1-codex
```
"Напиши функцию сортировки"
"Создай React компонент"
"Добавь error handling"
```
**Экономия:** 40-80%

---

## Сравнение моделей

### По скорости
1. **deepseek-v3.1** — 2.3s average
2. **gemini-3-flash** — ultra fast
3. **gpt-5.1-codex** — fast
4. **claude-4.5-sonnet** — medium

### По качеству кода
1. **claude-4.5-sonnet** — best overall
2. **gpt-5.1-codex** — optimized for code
3. **claude-4.6-opus** — overkill для большинства задач
4. **deepseek-v3.1** — good for simple code

### По цене (дешевле → дороже)
1. **cursor-small, deepseek-v3.1** — FREE
2. **gpt-5.1-codex** — $0.75
3. **gemini-3-flash** — $1.75
4. **claude-4.5-haiku** — $3.00
5. **claude-4.5-sonnet** — $9.00
6. **claude-4.6-opus** — $15.00

---

## Рекомендации

### Для Cursor Free
- Используй только: cursor-small, deepseek-v3.1, gpt-4o-mini
- Избегай premium моделей (быстро кончатся 50 запросов)

### Для Cursor Pro ($20/мес)
- **Простые задачи:** cursor-small, deepseek-v3.1 (не расходуют кредиты)
- **Генерация кода:** gpt-5.1-codex или claude-4.5-sonnet
- **$20 хватит на весь месяц** с distiq-code роутингом

### Для Cursor Max ($200/мес)
- Можешь использовать любые модели
- Но distiq-code всё равно экономит 60-70% кредитов
- Используй claude-4.6-opus только для сложных задач

---

## Источники

- [Cursor Models Documentation](https://cursor.com/docs/models)
- [Cursor Pricing Guide](https://cursor.com/pricing)
- [Model Performance Comparison](https://forum.cursor.com/t/models-comparison-table/61926)
- [Free Models Guide](https://dredyson.com/fix-cursors-free-model-confusion-in-5-minutes-a-beginners-step-by-step-guide-to-accessing-gpt-4o-mini-deepseek-hidden-gems/)
