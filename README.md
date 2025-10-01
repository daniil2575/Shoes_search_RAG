# Shoes_search_RAG

RAG‑сервис и инструменты для интернет‑магазина **обуви**: подготовка данных, модель прогноза спроса и рекомендации цен. Проект — адаптация от «велосипедов» к **`shoes`** (схема БД и ноутбуки обновлены).

## Стек
- Python 3.11+, Jupyter
- PostgreSQL (рекомендуется 14+), SQLAlchemy
- pandas, numpy, scikit‑learn, joblib
- Docker Compose (для локальной БД)
- (опционально) Git LFS для больших датасетов/артефактов

## Структура репозитория
```
.
├── mcp/                          # служебные скрипты/конфиги ассистента
├── schema.sql                    # DDL: схема и таблицы `shoes.*`
├── requirements.txt              # зависимости Python
├── docker-compose.yml            # Postgres (+ при необходимости pgAdmin)
├── .env_example                  # пример переменных окружения
├── .gitignore
├── data_mining_shoes.ipynb       # подготовка данных под обувь
├── price_prediction_shoes.ipynb  # модель спроса и рекомендации цен
├── data_mining.ipynb             # (старое/для справки)
├── price_prediction.ipynb        # (старое/для справки)
├── shoes.json                    # пример данных по обуви
└── bikes.json                    # пример (источник из старого проекта)
```

## Быстрый старт (локально)

1) **Python окружение**
```powershell
python -m venv .venv
.\.venv\Scriptsctivate          # PowerShell
pip install -U pip
pip install -r requirements.txt
```

2) **Переменные окружения**
```powershell
copy .env_example .env
# В .env задайте:
# SHOES_DB_URL=postgresql+psycopg2://user:password@localhost:5432/shoes
# ARTIFACT_DIR=./artifacts
# MODEL_FILENAME=price_model_shoes.joblib
```

3) **База данных**
- Запустите Postgres (удобно через Docker):
```powershell
docker-compose up -d
```
- Инициализируйте схему/таблицы:
```powershell
# Windows (psql в PATH PostgreSQL)
psql -U postgres -d shoes -f schema.sql
```
> Если базы `shoes` нет — создайте её через pgAdmin или `createdb shoes`/`CREATE DATABASE shoes;`.

4) **Подготовка данных**
- Откройте `data_mining_shoes.ipynb` → выполните ячейки по порядку (загрузка/очистка/запись в БД).

5) **Модель и рекомендации цен**
- Откройте `price_prediction_shoes.ipynb` и выполните блоки:
  - загрузка из `shoes.price_history`, `shoes.sales_daily`, `shoes.products` (+ опц. `shoes.costs`);
  - генерация фич (лаги/окна, скидки, GAP к конкурентам, сезонность, сток);
  - CV по `product_id`, обучение `HistGradientBoostingRegressor`, `joblib.dump`;
  - подбор цен сеткой кандидатов (шаг/границы/мин.маржа);
  - запись результатов в `shoes.price_recommendations`.

### Основные переменные окружения
| Переменная | Назначение | Пример |
|---|---|---|
| `SHOES_DB_URL` | строка подключения SQLAlchemy к PostgreSQL | `postgresql+psycopg2://postgres:postgres@localhost:5432/shoes` |
| `ARTIFACT_DIR` | папка для артефактов модели | `./artifacts` |
| `MODEL_FILENAME` | имя файла модели | `price_model_shoes.joblib` |
| `RANDOM_STATE` | сид | `42` |
| `CV_SPLITS` | число фолдов GroupKFold | `5` |
| `DEFAULT_PRICE_STEP` | шаг сетки цен | `100` |
| `DEFAULT_PRICE_MIN_MULT` | нижний множитель от текущей цены | `0.7` |
| `DEFAULT_PRICE_MAX_MULT` | верхний множитель | `1.3` |
| `DEFAULT_OPTIMIZE_FOR` | целевая метрика оптимизации | `revenue`/`profit` |
| `DEFAULT_MIN_MARGIN_PCT` | мин.маржа на ед., доля | `0.05` |

## Ожидаемая схема БД (`shoes`)
Минимальные таблицы:
- `shoes.products(id, brand_id, category_id, gender, season, material_upper, material_sole, color, created_at)`  
- `shoes.brands(id, name)`  
- `shoes.categories(id, name)`  
- `shoes.price_history(product_id, dt, price, promo_price, competitor_min_price, stock)`  
- `shoes.sales_daily(product_id, date, units_sold, revenue)`  
- `shoes.costs(product_id, cost)` — *опционально*  
- `shoes.price_recommendations(product_id, as_of_date, recommended_price, expected_units, expected_revenue, expected_profit, current_price, current_promo_price, unit_cost, method, created_at)`

> Полный DDL см. в `schema.sql`. Формирование рекомендаций выполняется из ноутбука `price_prediction_shoes.ipynb` (функция `recommend_prices_batch`).

## Полезные команды
```powershell
# Запуск/остановка БД
docker-compose up -d
docker-compose down

# Проверка подключения psql
psql -U postgres -d shoes -c "\dt shoes.*"

# Git (первый пуш)
git remote -v
git push -u origin main
```

## Частые проблемы
- **FATAL: password authentication failed** — проверьте логин/пароль и `SHOES_DB_URL`.  
- **port already in use** — порт Postgres, заданный в `docker-compose.yml`, занят; измените порт или остановите конфликтующий сервис.  
- **psycopg2 errors** — установите клиентские библиотеки PostgreSQL и переустановите зависимости (`pip install -r requirements.txt`).

## Лицензия
MIT
