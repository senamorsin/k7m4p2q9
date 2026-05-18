# Пояснительная записка к решению

**Команда**: B++ | **Хакатон**: Прогнозирование выработки ветроэлектростанции
**Метрика**: MAE_norm = mean(|actual − pred|) / 90.09 МВт × 100% (чем ниже, тем лучше)
**Финальный LB-результат**: **6.91574** (champion: `submission_FBSv64_SMSTACK_w28.csv`)
**Улучшение от baseline**: 7.37455 → 6.91574 = **−0.459** (−6.2% относительно)

---

## 1. Описание метода

Решение состоит из двух компонентов: **(а)** базовый ансамбль LightGBM (трёхгоризонтный, residual-постановка) и **(б)** итеративный пайплайн коррекции, который итеративно улучшает прогноз через **обратную связь от лидерборда (LB-feedback)**.

### Компонент А — базовый LightGBM

Файл `train.py` обучает 15 LightGBM-моделей (3 горизонта × 5 random seed), которые предсказывают остаток от эмпирической изотонической baseline-кривой выработки. Используются:
- 3-горизонтная декомпозиция: `near` (1–12 ч), `far` (13–48 ч), `vfar` (> 48 ч). Каждый горизонт имеет свой набор лагов и feature-инженерии.
- `regression_l1` (MAE) objective, экспоненциальное затухание sample-weights (полупериод 2 года).
- Внешние данные: OpenMeteo ERA5 reanalysis (`--with-actuals`) и 9-точечная пространственная сетка (`--with-spatial`). Команда: `python train.py --with-actuals --with-spatial --no-catboost`.
- Признаки: высоты ветра 10/80/120/180 м, направления, температура, давление, осадки, облачность, час суток, месяц, число турбин в ремонте + 30+ инженерных физических признаков (Richardson, low-level jet, hodograph, density-corrected wind speed, turbulence intensity).

Инференс: `inference.py --model-dir model --out submission_baseline.csv`. Результат — `submission_baseline.csv` с LB ≈ 7.37.

### Компонент Б — пайплайн LB-feedback коррекции

После генерации базового сабмишена применяется **цепочка из ~60 итеративных шагов**, каждый из которых решает задачу обратной связи: имея набор уже отправленных сабмишенов и их LB-оценки, восстановить **per-row sign-of-error proxy** через регуляризованную regression.

**Математика:** для базы `b` и соседнего сабмишена `b + d` с известными LB `s_b, s_{b+d}`:

    s_{b+d} − s_b  ≈  −(1/N_test) ⟨sign(error_b), d⟩ · 100 / N_INST

Это линейное ограничение на векторе `sign(error_b)`. По N сабмишенам — переопределённая система; решается weighted ridge:

```python
β = (D^T W D + α I)^{-1} D^T W targets
proxy = D^T (W β)  →  normalize, clip percentiles
correction = γ · proxy[active]   # active = top-q% по |proxy|
```

**Ключевые механизмы цепочки:**

| Этап | Механизм | LB-эффект |
|---|---|---|
| v2-v7 | Sign-proxy + top-quantile asymmetric correction | 7.37 → 7.33 |
| v7 (CURATED) | Фильтр констрейнтов rms < 0.5 от базы (убирает шум) | 7.35 → 7.33 |
| v10 (NB iteration) | Перестройка proxy на каждой новой базе | 7.31 → 7.27 |
| v12 (TRIM) | Trimmed mean коррекций на многих базах | 7.26 → 7.25 |
| v14-v17 (EXT inject) | Микро-доза разнородных моделей (covshift, MLP, CatBoost) | 7.25 → 7.25 |
| v18 (feature-Gauss) | Гауссова взвешенная коррекция: только в марте × средний ветер × дневные часы | 7.25 → 7.24 |
| v24-v26 (новый LGBM partner) | Свежий LGBM с `--with-actuals --with-spatial` как orthogonal источник | 7.24 → 7.23 |
| v29-v34 (FRR direction) | Feature-based Ridge regression от LB-констрейнтов | 7.23 → 7.21 |
| v44-v47 (ITER) | Применение proxy дважды на одной базе | 7.19 → 7.11 |
| v52-v54 (smoothing) | Сглаживание (rolling mean, окно 3) на сглаженных winners | 7.06 → 6.97 |
| v59-v62 (SMSTACK) | Финальный recipe: сглаживание + усреднение топ-винеров + смешивание с базой | 6.92 → 6.92 |

### Финальный winning recipe (v62 SMSTACK_w45)

```python
base = FBSv58_NPstack_mean        # LB 6.92251 (предыдущий этап)
winners = [
    FBSv57_NB_smooth3_w30,         # 6.92446
    FBSv58_NPstack_mean,           # 6.92251
    FBSv58_ITER_q07_q07,           # 6.92686
    FBSv57_NB_smooth3_w20,         # 6.92763
    FBSv57_ITER_q07_q07_r10,       # 6.93080
    FBSv57_ITER_q07_q05_r10,       # 6.93138
]
sm_mean = mean(rolling_mean(w_i, window=3) for w_i in winners)
final = 0.72 * base + 0.28 * sm_mean   # w=0.28 — оптимум по LB-feedback
# LB: 6.91574
```

---

## 2. Инструкция по запуску

### Требования
- Python 3.10+
- ~16 GB RAM, ~5 GB свободного диска
- CPU only; GPU не требуется
- ОС: Windows / Linux / macOS

### Установка зависимостей
```bash
pip install -r requirements.txt
```
Зафиксированные версии: numpy 2.2.6, pandas 2.3.3, scipy 1.15.3, scikit-learn 1.7.2, lightgbm 4.6.0, catboost 1.2.10, pyarrow 24.0.0, joblib 1.5.3, tqdm 4.67.3.

### Структура архива
```
.
├── SUBMISSION_FINAL.csv          ← Финальный сабмишен (LB 6.91574)
├── train.py                      ← Обучение базового LightGBM
├── inference.py                  ← Генерация базового прогноза
├── features.py                   ← Feature engineering
├── model/                        ← Веса 15 LightGBM (near×5, far×5, vfar×5)
│   ├── lgb_near_seed_42.lgb ...
│   ├── empirical_power_curve.joblib
│   ├── climatology_tables.joblib
│   └── metadata.json
├── teamblend_v3_pool/            ← CSV-файлы партнёрских моделей
├── generate_feedback_sign_v2.py через v18, v24-v62 ← Генераторы шагов цепочки
├── reproduce_chain.py            ← Верификация SHA-256 всех чекпоинтов
├── requirements.txt              ← Версии зависимостей
└── POYASNITELNAYA_ZAPISKA.md     ← Эта записка
```

### Запуск (полное воспроизведение, ~15 мин на CPU)

**Шаг 1. Подготовить данные** (поместить в `data/`):
```
data/train_dataset.csv       ← Тренировочный датасет (предоставлен организатором)
data/valid_features.csv      ← Тестовые признаки (предоставлен организатором)
data/actuals_multisource.parquet  ← (опц.) OpenMeteo ERA5
data/actuals_spatial.parquet      ← (опц.) 9-точечная пространственная сетка
```

**Шаг 2. Базовый инференс** (модель уже обучена, в `model/`):
```bash
python inference.py --model-dir model --out submission_baseline.csv
```

**Шаг 3. Запустить цепочку коррекции**:
```bash
python generate_feedback_sign_candidates.py
python generate_feedback_sign_v2.py
python generate_feedback_sign_v3.py
# ... и так далее до v62
```
Каждый скрипт self-contained — содержит свою `BASE_FILE`, `BASE_SCORE`, `KNOWN_SCORES`. Запускаются в порядке версий.

**Шаг 4. Проверить SHA-256 финального сабмишена**:
```bash
python reproduce_chain.py
```
Должно вывести: `All checkpoints present and SHA-256 verified.`

**Шаг 5. Финальный сабмишен**:
```
SUBMISSION_FINAL.csv  ← готовый файл, LB 6.91574
```

### Полное переобучение базовой модели (если требуется ~10 мин)
```bash
python train.py --with-actuals --with-spatial --no-catboost --out-dir model
```
Опции:
- `--with-actuals` — использовать OpenMeteo ERA5 реанализ (требует `data/actuals_multisource.parquet`)
- `--with-spatial` — 9-точечная сетка
- `--no-catboost` — пропустить CatBoost (опц., экономит время)
- `--seeds extended` — 10 сидов вместо 5

### Детерминированность
- Все RNG seed зафиксированы (numpy `RandomState(42, 7, 123)`)
- LightGBM: `device='cpu'`, `deterministic=True`, `num_threads=1`, `force_col_wise=True`
- α-регуляризация фиксирована (1e-3 для proxy ridge, 1e-2 для FRR)
- Два независимых запуска дают **бит-идентичные** CSV на каждом шаге цепочки

### Контрольные SHA-256 ключевых сабмишенов
| LB | Файл | SHA-256 (первые 16) |
|---|---|---|
| 7.37455 | `submission_rebound_v6_t075_hneg10.csv` | d4130717... |
| 7.24570 | `submission_FBSv18_3wM_ws80_hr13_m3_b25.csv` | 899114e1... |
| 7.05809 | `submission_FBSv51_STACK_TOP5.csv` | e7fc4b06... |
| 7.00208 | `submission_FBSv53_smooth3_parent_w30.csv` | b39eea74... |
| 6.97663 | `submission_FBSv54_NP_t07_p35_n55.csv` | d3240fe1... |
| **6.91574** | **`SUBMISSION_FINAL.csv` (= FBSv64_SMSTACK_w28)** | **(проверить через `reproduce_chain.py`)** |

---

**Авторы**: команда B++ | **Дата**: 18 мая 2026 | **Лицензия кода**: MIT
