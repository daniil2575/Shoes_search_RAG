"""
MCP SQL Generator Server - Сервер для генерации SQL-запросов с использованием LLM
"""

# Стандартные библиотеки
import os
from dotenv import load_dotenv
import sys
import json
import logging
import argparse
from typing import Optional

# Сторонние библиотеки
from langchain_openai import ChatOpenAI
from fastmcp import FastMCP

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Инициализация FastMCP сервера
mcp = FastMCP("sql-generator")

# Глобальный экземпляр LLM
llm: Optional[ChatOpenAI] = None

def initialize_llm():
    """Инициализирует языковую модель для генерации SQL"""
    global llm
    
    try:
        llm = ChatOpenAI(
            model="deepseek-chat",
            max_tokens=10000,
            temperature=0.7,
            top_p=0.8,
            api_key=os.getenv("API_KEY"),
            base_url=os.getenv("BASE_URL"),
            extra_body={"response_format": {"type": "json_object"}}
        )
        logger.info("Языковая модель успешно инициализирована")
    except Exception as e:
        logger.error(f"Ошибка при инициализации языковой модели: {str(e)}")
        raise RuntimeError("Не удалось инициализировать языковую модель")

@mcp.tool()
async def generate_sql(schema: str, user_query: str) -> str:
    """
    Description:
    ---------------
        Генерирует SQL-запрос на основе схемы данных и пользовательского запроса
        
    Args:
    ---------------
        schema (str): Описание схемы данных (таблицы, колонки, отношения)
        user_query (str): Пользовательский запрос на естественном языке
        
    Returns:
    ---------------
        str: Сгенерированный SQL-запрос или сообщение об ошибке
        
    Examples:
    ---------------
        >>> await generate_sql(
        ...     schema="Таблица Users: id, name, email; Таблица Orders: id, user_id, amount",
        ...     user_query="Покажи всех пользователей с их заказами"
        ... )
        'SELECT Users.name, Orders.amount FROM Users JOIN Orders ON Users.id = Orders.user_id'
    """
    logger.info("Получен запрос на генерацию SQL")
    
    if not llm:
        return "Языковая модель не инициализирована. Проверьте настройки сервера."
    
    try:
        # Формируем системный промпт
        system_prompt = f"""
        Ты опытный SQL разработчик. Твоя задача - генерировать корректные SQL запросы 
        на основе предоставленной схемы базы данных и пользовательского запроса на естественном языке.
        
        Схема базы данных:
        {schema}
        
        Инструкции:
        1. Анализируй схему и пользовательский запрос
        2. Генерируй только SQL запрос без пояснений
        3. Используй стандартный SQL синтаксис (ANSI SQL)
        4. Если запрос требует JOIN, явно укажи условие соединения
        5. Для неоднозначных запросов делай разумные предположения
        6. Возвращай ответ в формате JSON: {{"sql": "сгенерированный запрос"}}
        
        Пользовательский запрос: {user_query}
        """
        
        # Вызываем языковую модель
        response = llm.invoke(system_prompt)
        
        # Парсим JSON ответ
        try:
            result = json.loads(response.content)
            return result.get("sql", "Ошибка: в ответе отсутствует SQL запрос")
        except json.JSONDecodeError:
            logger.error(f"Неверный формат ответа от LLM: {response.content}")
            return "Ошибка: неверный формат ответа от языковой модели"
            
    except Exception as e:
        logger.error(f"Ошибка при генерации SQL: {str(e)}")
        return f"Ошибка при генерации SQL: {str(e)}"

def main() -> None:
    """Основная точка входа"""
    load_dotenv()

    parser = argparse.ArgumentParser(description="MCP SQL Generator Server")
    parser.add_argument("--debug", action="store_true", help="Включить подробное логирование")
    
    # Параметры для LLM
    parser.add_argument("--api-key", type=str, default=os.getenv("API_KEY"), 
                       help="API ключ для LLM (по умолчанию из переменной окружения API_KEY)")
    parser.add_argument("--base-url", type=str, default=os.getenv("BASE_URL"), 
                       help="Базовый URL для LLM API (по умолчанию из переменной окружения BASE_URL)")
    
    args = parser.parse_args()

    # Установка уровня логирования
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.info("Режим отладки включен")

    # Установка переменных окружения для LLM
    if args.api_key:
        os.environ["API_KEY"] = args.api_key
    if args.base_url:
        os.environ["BASE_URL"] = args.base_url

    # Проверка обязательных параметров
    if not os.getenv("API_KEY"):
        logger.error("Не указан API_KEY! Используйте --api-key или установите переменную окружения API_KEY")
        sys.exit(1)
        
    if not os.getenv("BASE_URL"):
        logger.error("Не указан BASE_URL! Используйте --base-url или установите переменную окружения BASE_URL")
        sys.exit(1)

    # Инициализация языковой модели
    try:
        initialize_llm()
    except Exception as e:
        logger.error(f"Не удалось инициализировать сервер: {str(e)}")
        sys.exit(1)

    try:
        # Запускаем сервер
        logger.info("Запуск SQL Generator Server...")
        mcp.run(transport='streamable-http', host="0.0.0.0", port=8000, path="/")
    except KeyboardInterrupt:
        logger.info("Сервер остановлен пользователем")
    except Exception as e:
        logger.exception(f"Ошибка при работе сервера: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()