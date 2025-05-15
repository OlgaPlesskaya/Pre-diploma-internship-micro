# streamlit_app.py

import streamlit as st
import pandas as pd
import time

# Импорт модулей
from utils.data_processor import process_uploaded_file
from utils.visualization import generate_wordcloud, generate_graphs
from utils.api_client import get_categories, get_subcategories
from config.settings import API_URL_UPLOAD
from utils.api_client import upload_original_file, update_processed_file

# === UI ===
st.set_page_config(page_title="Сервис предобработки текстовых сообщений", layout="wide")
st.title("Сервис предобработки текстовых сообщений")



# === Боковая панель ===
with st.sidebar:
    st.header("Образование")
    search_term = st.text_input("Поиск по подкатегориям")
    categories = get_categories()
    for category in categories:
        with st.expander(f"{category['emoji']} {category['name']}"):
            subcategories = get_subcategories(category['identifier'])
            filtered = [s for s in subcategories if search_term.lower() in s["name"].lower() or not search_term]
            
            # Стиль для выравнивания заголовка popover по левому краю
            st.markdown("""
                <style>
                p {

                        text-align: left !important;
                    }
                    .st-emotion-cache-qm7g72 {
                        font-size: 0rem !important;
                    }
                </style>
            """, unsafe_allow_html=True)
                        
            for subcat in filtered:
                with st.popover(subcat["name"]):
                    st.markdown(subcat["description"])

# === Основной интерфейс ===
st.markdown("## Как пользоваться сервисом:")
st.markdown("""
1️⃣ **Загрузите файл**  
   Нажмите кнопку «Выберите файл» и выберите подходящий файл в формате `.csv`. Убедитесь, что файл содержит текстовые данные.

2️⃣ **Дождитесь обработки**  
   После загрузки начнётся автоматическая обработка данных. Прогресс будет отображаться на экране.

3️⃣ **Просмотрите и скачайте результаты**  
   По завершении вы увидите таблицу с результатами анализа. Также станет доступен для скачивания обработанный CSV-файл.
""")

if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None

uploaded_file = st.file_uploader("Выберите CSV-файл", type=["csv"], label_visibility="collapsed")

if uploaded_file is not None:
    if st.button("Начать обработку"):
        try:
            df = pd.read_csv(uploaded_file)

            file_info = upload_original_file(
                uploaded_file.getvalue(),
                uploaded_file.name,
                API_URL_UPLOAD
            )

            file_id = file_info['id'] if file_info else None

            status_text = st.empty()
            progress_bar = st.progress(0)

            status_text.info("Запуск модели...")
            time.sleep(0.5)
            processed_df = process_uploaded_file(df)
            progress_bar.progress(33)

            status_text.info("Генерация графиков...")

            
            wordcloud_fig = generate_wordcloud(processed_df)

            fig1, fig2 = generate_graphs(processed_df)
            progress_bar.progress(66)


            status_text.info("Сохранение данных...")
            time.sleep(0.5)
            progress_bar.progress(100)

            if file_id:
                update_processed_file(file_id, processed_df, API_URL_UPLOAD)

            st.session_state.update({
                'processed_df': processed_df,
                'wordcloud_fig': wordcloud_fig,
                'fig1': fig1,
                'fig2': fig2
            })

            status_text.empty()
            progress_bar.empty()
            st.success("Файл успешно обработан!")

        except Exception as e:
            st.error(f"Ошибка: {e}")

# Показываем результаты, если они уже есть в session_state
if st.session_state.processed_df is not None:
    processed_df = st.session_state.processed_df

    # Кнопка для скачивания
    csv_ready_df = processed_df[['text', 'predicted_labels']]
    csv_data = csv_ready_df.to_csv(index=False).encode()
    st.download_button(
        label="Скачать обработанный файл",
        data=csv_data,
        file_name="processed_data.csv",
        mime="text/csv",
    )

    st.markdown("#### Результаты обработки (первые 10 строк):")

    
    #st.dataframe(processed_df[['text', 'predicted_labels']].head(10))
    
    # Пример преобразования списка меток в HTML-бэйджи
    def labels_to_badges(labels):
        if isinstance(labels, str):
            labels = labels.split(';')  # если метки записаны в одну строку через запятую
        badges = ''.join(
            [f'<span style="margin:2px 4px; padding:4px 8px; background-color:#f0f2f6; border-radius:12px; display:inline-block; min-width:100px; text-align:left;">{label}</span>' 
            for label in labels]
        )
        return badges

    # Подготовка DataFrame для отображения
    display_df = processed_df[['text', 'predicted_labels']].head(10).copy()
    display_df['predicted_labels'] = display_df['predicted_labels'].apply(labels_to_badges)
    display_df.columns = ['Текст', 'Метки']

    # Стили для таблицы
    st.html('''
        <style>
        .badge-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 0px !important;
        }
        .badge-table th, .badge-table td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        .badge-table tr:hover {
            background-color: #f9f9f9;
        }

        .st-emotion-cache-16tyu1 th {
            text-align: left;
        }

        td {
            vertical-align: top;
        }


        </style>
    ''')

    # Конвертация DataFrame в HTML и вывод
    html_table = display_df.to_html(index=False, escape=False)
    st.markdown(f'<table class="badge-table">{html_table}</table>', unsafe_allow_html=True)



    # Графики
    if st.session_state.wordcloud_fig:
        st.markdown("#### ☁️ Облако слов")
        st.plotly_chart(wordcloud_fig, use_container_width=True)

    if st.session_state.fig1:
        st.markdown("#### 📊 Распределение по категориям")
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown("#### 📈 Топ-10 подкатегорий")
        st.plotly_chart(fig2, use_container_width=True)
        