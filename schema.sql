CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE shoes (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    brand TEXT,
    category TEXT,
    price numeric(12,2),
    old_price TEXT,
    discount TEXT,
    in_stock TEXT,
    url TEXT,
    image_url TEXT,
    full_description TEXT,  -- Склеенное описание для индексов
    embedding VECTOR(1024),  -- Размерность модели Qwen3-Embedding-0.6B
    fts_vector tsvector  -- Для полнотекстового поиска
);

-- Векторный индекс (HNSW для оптимальной производительности)
CREATE INDEX ON shoes USING hnsw (embedding vector_cosine_ops);

-- Полнотекстовый индекс (GIN для русского языка)
CREATE INDEX ON shoes USING gin(fts_vector);