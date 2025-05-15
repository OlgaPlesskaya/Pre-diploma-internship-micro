# utils/api_client.py
import requests

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
    url = f"{api_url or API_URL_SUBCATEGORIES}?category={category_id}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        return []
    except Exception as e:
        print(f"Ошибка подключения к API подкатегорий: {e}")
        return []