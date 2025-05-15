# utils/api_client.py
import requests
from typing import Optional, Dict, Any
import tempfile
import os

def get_categories(api_url=None):
    from config.settings import API_URL_CATEGORIES
    url = api_url or API_URL_CATEGORIES
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        return []
    except Exception as e:
        print(f"Ошибка подключения к API категорий: {e}")
        return []


def get_subcategories(category_id, api_url=None):
    from config.settings import API_URL_SUBCATEGORIES
    url = api_url or API_URL_SUBCATEGORIES  # например: '/api/subcategorys/'
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            # Фильтруем подкатегории по категории на стороне клиента
            filtered = [item for item in data if item.get('category') == category_id]
            return filtered
        return []
    except Exception as e:
        print(f"Ошибка подключения к API подкатегорий: {e}")
        return []


def get_subcategory_to_category_mapping():
    mapping = {}

    # Получаем все категории
    categories = get_categories()
    category_names = {cat["identifier"]: cat["name"] for cat in categories}

    # Для каждой категории получаем подкатегории
    for cat in categories:
        category_id = cat["identifier"]
        subcategories = get_subcategories(category_id)

        for sub in subcategories:
            # Ключ — имя подкатегории, значение — имя категории
            mapping[sub["name"]] = category_names.get(sub["category"], "Без категории")

    return mapping


def upload_original_file(file_bytes: bytes, filename: str, api_url: str) -> Optional[Dict[str, Any]]:
    """
    Загружает оригинальный CSV-файл на сервер.
    
    :param file_bytes: Содержимое файла в байтах.
    :param filename: Имя файла (например, 'data.csv').
    :param api_url: URL API для загрузки.
    :return: JSON-ответ от сервера или None в случае ошибки.
    """
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        tmpfile.write(file_bytes)
        tmpfile_path = tmpfile.name

    try:
        with open(tmpfile_path, 'rb') as f:
            files = {'file': (filename, f, 'text/csv')}
            response = requests.post(api_url, files=files)
        
        if response.status_code == 201:
            return response.json()
        else:
            print(f"Ошибка при загрузке файла: {response.status_code} — {response.text}")
            return None
    except Exception as e:
        print(f"Ошибка при отправке файла: {e}")
        return None
    finally:
        os.unlink(tmpfile_path)


def update_processed_file(file_id: int, processed_df, api_url: str) -> bool:
    """
    Отправляет обработанный DataFrame как CSV на сервер для указанного ID файла.

    :param file_id: ID файла на сервере.
    :param processed_df: pd.DataFrame — обработанный датафрейм.
    :param api_url: Базовый URL API (например, 'http://localhost:8000/api/up_fles/').
    :return: True, если успешно, иначе False.
    """
    csv_data = processed_df.to_csv(index=False).encode()

    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmpfile:
        tmpfile.write(csv_data)
        tmpfile_path = tmpfile.name

    try:
        with open(tmpfile_path, 'rb') as f:
            files = {'processed_file': (f"processed_{tmpfile.name.split('/')[-1]}", f, 'text/csv')}
            data = {}
            response = requests.patch(f"{api_url}{file_id}/", data=data, files=files)

        if response.status_code == 200:
            print("Обработанный файл успешно обновлён.")
            return True
        else:
            print(f"Ошибка при обновлении файла: {response.status_code} — {response.text}")
            return False
    except Exception as e:
        print(f"Ошибка при обновлении файла: {e}")
        return False
    finally:
        os.unlink(tmpfile_path)