import streamlit as st
import requests
import pandas as pd

# === Настройки ===
API_URL_UPLOAD = "http://localhost:8000/api/up_fles/"  # должен быть реализован на Django
API_URL_CATEGORIES = "http://localhost:8000/api/categorys/"
API_URL_SUBCATEGORIES = "http://localhost:8000/api/subcategorys/"

# === UI ===
st.set_page_config(page_title="Сервис предобработки текстовых сообщений", layout="wide")
st.title("Сервис предобработки текстовых сообщений")

# === Боковая панель с категориями и подкатегориями ===
with st.sidebar:
    st.header("Образование")
    st.markdown("Информация о категориях:")
    search_term = st.text_input("Поиск по подкатегориям")

    try:
        response = requests.get(API_URL_CATEGORIES)
        if response.status_code == 200:
            categories = response.json()
        else:
            categories = []
            st.error("Не удалось загрузить категории")
    except Exception as e:
        st.error(f"Ошибка подключения к API: {e}")
        categories = []

    for category in categories:
        with st.expander(f"{category['emoji']} {category['name']}"):
            try:
                subcat_response = requests.get(
                    f"{API_URL_SUBCATEGORIES}?category={category['identifier']}"
                )
                if subcat_response.status_code == 200:
                    subcategories = subcat_response.json()
                else:
                    subcategories = []
            except:
                subcategories = []

            filtered_subcats = [
                s for s in subcategories
                if search_term.lower() in s["name"].lower() or not search_term
            ]

            if not filtered_subcats:
                st.markdown("*Ничего не найдено*")
            else:
                for subcat in filtered_subcats:
                    with st.popover(subcat["name"]):
                        st.markdown(subcat["description"])

# === Основной контент ===
st.markdown("## Как пользоваться сервисом:")
st.markdown("""
1. **Загрузите файл**  
   Нажмите кнопку «Выберите файл» и выберите подходящий файл в формате `.csv`. Убедитесь, что файл содержит текстовые данные.

2. **Дождитесь обработки**  
   После загрузки начнётся автоматическая обработка данных. Прогресс будет отображаться на экране.

3. **Просмотрите и скачайте результаты**  
   По завершении вы увидите таблицу с результатами анализа. Также станет доступен для скачивания обработанный CSV-файл.
""")

st.markdown("### Загрузите файл для обработки")
uploaded_file = st.file_uploader("Выберите CSV-файл", type=["csv"], label_visibility="collapsed")

if uploaded_file is not None:
    if st.button("Начать обработку"):
        files = {"file": uploaded_file.getvalue()}
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            status_text.info("Отправка файла на сервер...")
            response = requests.post(API_URL_UPLOAD, files=files)

            if response.status_code == 200:
                result = response.json()
                for percent_complete in range(0, 100, 10):
                    time.sleep(0.2)
                    progress_bar.progress(percent_complete + 10)
                status_text.success("Файл успешно обработан!")

                # === Таблица с первыми 10 строками ===
                st.markdown("## Результаты обработки (первые 10 строк):")
                df = pd.DataFrame(result["preview"])
                st.dataframe(df.head(10))

                # === Ссылка для скачивания ===
                download_url = result.get("download_url")
                if download_url:
                    st.download_button(
                        label="Скачать обработанный файл",
                        data=requests.get(download_url).content,
                        file_name="processed_data.csv",
                        mime="text/csv"
                    )

                # === Графики ===
                st.markdown("## Графики")
                cols = st.columns(3)

                if result.get("wordcloud_url"):
                    with cols[0]:
                        st.markdown("#### ☁️ Облако слов")
                        st.image(result["wordcloud_url"], use_column_width=True)

                if result.get("graph1_url"):
                    with cols[1]:
                        st.markdown("#### 📊 Распределение по категориям")
                        st.image(result["graph1_url"], use_column_width=True)

                if result.get("graph2_url"):
                    with cols[2]:
                        st.markdown("#### 📈 Распределение по подкатегориям")
                        st.image(result["graph2_url"], use_column_width=True)

            else:
                st.error(f"Ошибка сервера: {response.status_code}")
                st.json(response.text)

        except Exception as e:
            st.error(f"Не удалось связаться с API: {str(e)}")