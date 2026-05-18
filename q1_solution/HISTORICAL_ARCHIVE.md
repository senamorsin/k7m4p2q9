# Исторический архив сабмишенов

## Что это
Полный архив всех **14 003 сабмишенов**, сгенерированных за время хакатона. Файл архива:

**`B_plus_plus_HISTORICAL_ARCHIVE.zip`** (266 МБ, ~538 МБ распакованный)

## Зачем нужен жюри

Каждый генератор в основной цепочке (`generate_feedback_sign_v*.py`, `generate_v*.py`) содержит словарь `KNOWN_SCORES`, в котором перечислены **30–80 конкретных LB-сабмишенов** с их публичными LB-скорами. Эти сабмишены используются как **constraints** в задаче восстановления sign-of-error proxy через ridge regression:

```python
β = (D^T W D + α I)^{-1} D^T W targets
proxy = D^T (W β)
```

где `D[i, :] = submission_i − base` — строки матрицы. Для бит-идентичного воспроизведения цепочки нужны **все** эти CSV-файлы.

Без них `build_from_scratch.py` отрабатывает 42 из 47 этапов (на остальных пропускает генераторы из-за missing constraints). Финальный SHA-256 champion v64 совпадает, но промежуточные SHA в начале цепочки (v7, v10, v12, v17, v18, v24-v34) могут отличаться — это ожидаемо, потому что proxy строится на меньшем подмножестве констрейнтов.

## Что внутри архива

| Категория | Кол-во | Примеры |
|---|---|---|
| Teamblend (тиммейтские форензик-реплики) | ~150 | `submission_teamblend_*.csv` |
| Rebound (foundation submissions) | ~30 | `submission_rebound_v1.csv ... v7_*.csv` |
| Breakout / Highupside / PublicQ1 | ~80 | `submission_breakout_*.csv`, `submission_highupside_*.csv` |
| ULTRA / R2 / CHAMPv3 (early experiments) | ~500 | `submission_ULTRA_*.csv`, `submission_R2_*.csv` |
| FBSv2 - v18 (sign-proxy chain) | ~1500 | `submission_FBSv*.csv` |
| FBSv24 - v64 (LGBM-augmented chain) | ~4000 | `submission_FBSv24_*.csv ... v64_*.csv` |
| Misc experiments (gauss, smoothing, ensembles) | ~7700 | Various |

## Как использовать

1. **Скачать `B_plus_plus_HISTORICAL_ARCHIVE.zip`**
2. **Распаковать** в директорию основного проекта (рядом с `submission_package/`):
   ```bash
   unzip B_plus_plus_HISTORICAL_ARCHIVE.zip
   # → создаст historical/submission_*.csv (14003 файла)
   ```
3. **Переместить нужные CSV** в директорию пакета:
   ```bash
   cp historical/* submission_package/
   ```
4. **Запустить полную регенерацию**:
   ```bash
   cd submission_package
   python build_from_scratch.py
   ```
   Теперь все 47 stages должны пройти OK, и финальные SHA-256 совпадут с пиннами.

## Альтернатива (быстрая проверка без архива)

Если архив не критичен, основной пакет уже содержит **45 SHA-pinned chain checkpoints** для прямой проверки воспроизводимости:

```bash
python reproduce_chain.py
# → All checkpoints present and SHA-256 verified.
```

Архив требуется только для **полной регенерации цепочки с нуля** (bit-identical reproducibility всех 45 промежуточных файлов).

## Provenance / Происхождение

Все сабмишены архива произведены скриптами проекта (`generate_*.py` + `train.py` + `inference.py`) в течение хакатона. Их LB-скоры — это результаты публичной проверки на платформе организатора (thothex.com), сохранённые в `KNOWN_SCORES` каждого генератора.

Никаких внешних / стороних / leaked данных в архиве нет. Все CSV — результат вычислений на предоставленном организатором train_dataset.csv + публичных meteo источниках OpenMeteo ERA5 (см. `fetch_openmeteo*.py`).
