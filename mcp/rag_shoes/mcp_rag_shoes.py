"""
MCP Shoes RAG Server — поиск ОБУВИ с использованием векторного и полнотекстового индексов.
Совместим по структуре и зависимостям с прежним shoes-сервером (psycopg + pgvector + fastmcp).
"""

# Стандартные библиотеки
import os
import sys
import logging
import argparse
from typing import List, Dict, Any, Tuple, Optional

# Сторонние библиотеки
import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer
from fastmcp import FastMCP

# -----------------------------------
# Логирование
# -----------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("mcp_rag_shoes")

# -----------------------------------
# MCP-сервер (имя сервера сменено)
# -----------------------------------
mcp = FastMCP("shoes-rag")

# -----------------------------------
# Глобальные объекты/конфиг
# -----------------------------------
model: Optional[SentenceTransformer] = None

# Важно: оставляем дефолты совместимыми с прежним проектом
DB_CONNECTION_PARAMS = {
    "dbname": os.getenv("DB_NAME", "shoes"),          # не переименовываем, чтобы не трогать схему/compose
    "user": os.getenv("DB_USER", "postgres"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5430"),
    "password": os.getenv("DB_PASSWORD", ""),
}

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
USE_GPU = os.getenv("USE_GPU", "false").lower() == "true"


# -----------------------------------
# Инициализация модели и БД
# -----------------------------------
def initialize_model() -> None:
    """Загружает модель эмбеддингов (Qwen3-Embedding), GPU по флагу USE_GPU."""
    global model
    device = "cuda" if USE_GPU else "cpu"
    logger.info(f"Загрузка модели эмбеддингов: {EMBEDDING_MODEL_NAME} (device={device})")
    try:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
        logger.info("Модель эмбеддингов успешно загружена (каталог: обувь)")
    except Exception as e:
        logger.exception("Не удалось загрузить модель эмбеддингов")
        raise RuntimeError(str(e))


def connect_db():
    """Подключение к Postgres и регистрация pgvector."""
    try:
        conn = psycopg.connect(**DB_CONNECTION_PARAMS)
        register_vector(conn)
        return conn
    except Exception as e:
        logger.exception("Ошибка подключения к БД")
        raise RuntimeError(str(e))


# -----------------------------------
# Форматирование результата
# -----------------------------------
def format_product_row(row: Tuple) -> Dict[str, Any]:
    """
    Ожидаемый порядок колонок в селектах ниже:
    name, full_description, brand, price, url, score
    """
    return {
        "name": row[0],
        "description": row[1],
        "brand": row[2],
        "price": float(row[3]) if row[3] is not None else None,
        "url": row[4],
        "score": float(row[5]) if row[5] is not None else None,
    }


# -----------------------------------
# Инструменты MCP
# -----------------------------------
@mcp.tool()
async def health() -> Dict[str, str]:
    """Быстрая проверка, что сервер на месте."""
    return {"status": "ok", "service": "shoes-rag"}


@mcp.tool()
async def vector_search(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Поиск ОБУВИ по векторному сходству (эмбеддинги).
    Примеры:
        >>> await vector_search("мужские кроссовки nike черные 42", 3)
    """
    logger.info(f"[vector] query='{query}' limit={limit}")
    if not model:
        return [{"error": "Модель эмбеддингов не инициализирована"}]

    try:
        query_emb = model.encode([query], prompt_name="query")[0]
        with connect_db() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT name, full_description, brand, price, url,
                       1 - (embedding <=> %s) AS similarity
                FROM shoes
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (query_emb, query_emb, limit),
            )
            rows = cur.fetchall()
        return [format_product_row(r) for r in rows]
    except Exception as e:
        logger.exception("Ошибка в vector_search")
        return [{"error": f"vector_search failed: {e}"}]


@mcp.tool()
async def fulltext_search(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Полнотекстовый поиск ОБУВИ по описанию/характеристикам (russian FTS).
    Примеры:
        >>> await fulltext_search("женские зимние ботинки кожа 37 черные", 3)
    """
    logger.info(f"[fts] query='{query}' limit={limit}")
    try:
        with connect_db() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT name, full_description, brand, price, url,
                       ts_rank(fts_vector, websearch_to_tsquery('russian', %s)) AS rank
                FROM shoes
                WHERE fts_vector @@ websearch_to_tsquery('russian', %s)
                ORDER BY rank DESC
                LIMIT %s
                """,
                (query, query, limit),
            )
            rows = cur.fetchall()
        return [format_product_row(r) for r in rows]
    except Exception as e:
        logger.exception("Ошибка в fulltext_search")
        return [{"error": f"fulltext_search failed: {e}"}]


@mcp.tool()
async def hybrid_search(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Гибридный поиск ОБУВИ (векторный + полнотекстовый).
    Примеры:
        >>> await hybrid_search("кеды converse белые мужские 43", 5)
    """
    logger.info(f"[hybrid] query='{query}' limit={limit}")
    if not model:
        return [{"error": "Модель эмбеддингов не инициализирована"}]

    try:
        query_emb = model.encode([query], prompt_name="query")[0]
        with connect_db() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT name, full_description, brand, price, url,
                       (0.7 * (1 - (embedding <=> %s))
                        + 0.3 * ts_rank(fts_vector, websearch_to_tsquery('russian', %s))) AS score
                FROM shoes
                ORDER BY score DESC
                LIMIT %s
                """,
                (query_emb, query, limit),
            )
            rows = cur.fetchall()
        return [format_product_row(r) for r in rows]
    except Exception as e:
        logger.exception("Ошибка в hybrid_search")
        return [{"error": f"hybrid_search failed: {e}"}]


@mcp.tool()
async def search_methods() -> List[str]:
    """Возвращает список доступных методов поиска."""
    return ["vector_search", "fulltext_search", "hybrid_search", "health"]


# -----------------------------------
# Точка входа
# -----------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="MCP Shoes RAG Server")
    parser.add_argument("--debug", action="store_true", help="Подробное логирование")

    # Подключение к БД (оставляем совместимые дефолты)
    parser.add_argument("--db-host", type=str, default=os.getenv("DB_HOST", "vector-db"))
    parser.add_argument("--db-port", type=int, default=int(os.getenv("DB_PORT", "5432")))
    parser.add_argument("--db-name", type=str, default=os.getenv("DB_NAME", "shoes"))
    parser.add_argument("--db-user", type=str, default=os.getenv("DB_USER", "postgres"))
    parser.add_argument("--db-password", type=str, default=os.getenv("DB_PASSWORD", ""))

    # Модель/устройство
    parser.add_argument("--use-gpu", action="store_true", help="Использовать GPU для эмбеддингов")
    parser.add_argument("--model", type=str, default=os.getenv("EMBEDDING_MODEL", EMBEDDING_MODEL_NAME))

    # Сетевые параметры MCP
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--path", type=str, default="/")

    args = parser.parse_args()

    # Логи
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.debug("Режим отладки включен")

    # Обновляем конфиг по аргументам
    DB_CONNECTION_PARAMS.update(
        {
            "host": args.db_host,
            "port": args.db_port,
            "dbname": args.db_name,
            "user": args.db_user,
            "password": args.db_password,
        }
    )
    global EMBEDDING_MODEL_NAME, USE_GPU
    EMBEDDING_MODEL_NAME = args.model
    if args.use_gpu:
        USE_GPU = True

    # Инициализация
    try:
        initialize_model()
        with connect_db():
            logger.info("Проверка подключения к БД — OK")
    except Exception as e:
        logger.error(f"Инициализация не удалась: {e}")
        sys.exit(1)

    # Старт MCP
    try:
        logger.info(f"Запуск Shoes RAG Server на {args.host}:{args.port}{args.path}")
        mcp.run(transport="streamable-http", host=args.host, port=args.port, path=args.path)
    except KeyboardInterrupt:
        logger.info("Остановлено пользователем")
    except Exception as e:
        logger.exception(f"Критическая ошибка во время работы: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
