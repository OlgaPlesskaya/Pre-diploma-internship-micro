import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import requests
import re
import time
import os
import io


# === Настройки ===
API_URL_UPLOAD = "http://localhost:8000/api/up_fles/"  # должен быть реализован на Django
API_URL_CATEGORIES = "http://localhost:8000/api/categorys/"
API_URL_SUBCATEGORIES = "http://localhost:8000/api/subcategorys/"

MAX_LENGTH = 150
SCALING_FACTOR = 0.7

# --- Стоп-слова ---
STOP_WORDS = {
        'c', 'а', 'алло', 'без', 'белый', 'близко', 'более', 'больше',
        'большой', 'будем', 'будет', 'будете', 'будешь', 'будто', 'буду',
        'будут', 'будь', 'бы', 'бывает', 'бывь', 'был', 'была', 'были',
        'было', 'быть', 'в', 'важная', 'важное', 'важные', 'важный', 'вам',
        'вами', 'вас', 'ваш', 'ваша', 'ваше', 'ваши', 'вверх', 'вдали',
        'вдруг', 'ведь', 'везде', 'вернуться', 'весь', 'вечер', 'взгляд',
        'взять', 'вид', 'видел', 'видеть', 'вместе', 'вне', 'вниз', 'внизу',
        'во', 'вода', 'война', 'вокруг', 'вон', 'вообще', 'вопрос',
        'восемнадцатый', 'восемнадцать', 'восемь', 'восьмой', 'вот',
        'впрочем', 'времени', 'время', 'все', 'все еще', 'всегда', 'всего',
        'всем', 'всеми', 'всему', 'всех', 'всею', 'всю', 'всюду', 'вся',
        'всё', 'второй', 'вы', 'выйти', 'г', 'где', 'главный', 'глаз',
        'говорил', 'говорит', 'говорить', 'год', 'года', 'году', 'голова',
        'голос', 'город', 'да', 'давать', 'давно', 'даже', 'далекий',
        'далеко', 'дальше', 'даром', 'дать', 'два', 'двадцатый', 'двадцать',
        'две', 'двенадцатый', 'двенадцать', 'дверь', 'двух', 'девятнадцатый',
        'девятнадцать', 'девятый', 'девять', 'действительно', 'дел', 'делал',
        'делать', 'делаю', 'дело', 'день', 'деньги', 'десятый', 'десять',
        'для', 'до', 'довольно', 'долго', 'должен', 'должно', 'должный',
        'дом', 'дорога', 'друг', 'другая', 'другие', 'других', 'друго',
        'другое', 'другой', 'думать', 'душа', 'е', 'его', 'ее', 'ей', 'ему',
        'если', 'есть', 'еще', 'ещё', 'ею', 'её', 'ж', 'ждать', 'же', 'жена',
        'женщина', 'жизнь', 'жить', 'за', 'занят', 'занята', 'занято',
        'заняты', 'затем', 'зато', 'зачем', 'здесь', 'земля', 'знать',
        'значит', 'значить', 'и', 'иди', 'идти', 'из', 'или', 'им', 'имел',
        'именно', 'иметь', 'ими', 'имя', 'иногда', 'их', 'к', 'каждая',
        'каждое', 'каждые', 'каждый', 'кажется', 'казаться', 'как', 'какая',
        'какой', 'кем', 'книга', 'когда', 'кого', 'ком', 'комната', 'кому',
        'конец', 'конечно', 'которая', 'которого', 'которой', 'которые',
        'который', 'которых', 'кроме', 'кругом', 'кто', 'куда', 'лежать',
        'лет', 'ли', 'лицо', 'лишь', 'лучше', 'любить', 'люди', 'м',
        'маленький', 'мало', 'мать', 'машина', 'между', 'меля', 'менее',
        'меньше', 'меня', 'место', 'миллионов', 'мимо', 'минута', 'мир',
        'мира', 'мне', 'много', 'многочисленная', 'многочисленное',
        'многочисленные', 'многочисленный', 'мной', 'мною', 'мог', 'могу',
        'могут', 'мож', 'может', 'может быть', 'можно', 'можхо', 'мои', 'мой',
        'мор', 'москва', 'мочь', 'моя', 'моё', 'мы', 'на', 'наверху', 'над',
        'надо', 'назад', 'наиболее', 'найти', 'наконец', 'нам', 'нами',
        'народ', 'нас', 'начала', 'начать', 'наш', 'наша', 'наше', 'наши',
        'не', 'него', 'недавно', 'недалеко', 'нее', 'ней', 'некоторый',
        'нельзя', 'нем', 'немного', 'нему', 'непрерывно', 'нередко',
        'несколько', 'нет', 'нею', 'неё', 'ни', 'нибудь', 'ниже', 'низко',
        'никакой', 'никогда', 'никто', 'никуда', 'ним', 'ними', 'них',
        'ничего', 'ничто', 'но', 'новый', 'нога', 'ночь', 'ну', 'нужно',
        'нужный', 'нх', 'о', 'об', 'оба', 'обычно', 'один', 'одиннадцатый',
        'одиннадцать', 'однажды', 'однако', 'одного', 'одной', 'оказаться',
        'окно', 'около', 'он', 'она', 'они', 'оно', 'опять', 'особенно',
        'остаться', 'от', 'ответить', 'отец', 'откуда', 'отовсюду', 'отсюда',
        'очень', 'первый', 'перед', 'писать', 'плечо', 'по', 'под', 'подойди',
        'подумать', 'пожалуйста', 'позже', 'пойти', 'пока', 'пол', 'получить',
        'помнить', 'понимать', 'понять', 'пор', 'пора', 'после', 'последний',
        'посмотреть', 'посреди', 'потом', 'потому', 'почему', 'почти',
        'правда', 'прекрасно', 'при', 'про', 'просто', 'против', 'процентов',
        'путь', 'пятнадцатый', 'пятнадцать', 'пятый', 'пять', 'работа',
        'работать', 'раз', 'разве', 'рано', 'раньше', 'ребенок', 'решить',
        'россия', 'рука', 'русский', 'ряд', 'рядом', 'с', 'с кем', 'сам',
        'сама', 'сами', 'самим', 'самими', 'самих', 'само', 'самого',
        'самой', 'самом', 'самому', 'саму', 'самый', 'свет', 'свое', 'своего',
        'своей', 'свои', 'своих', 'свой', 'свою', 'сделать', 'сеаой', 'себе',
        'себя', 'сегодня', 'седьмой', 'сейчас', 'семнадцатый', 'семнадцать',
        'семь', 'сидеть', 'сила', 'сих', 'сказал', 'сказала', 'сказать',
        'сколько', 'слишком', 'слово', 'случай', 'смотреть', 'сначала',
        'снова', 'со', 'собой', 'собою', 'советский', 'совсем', 'спасибо',
        'спросить', 'сразу', 'стал', 'старый', 'стать', 'стол', 'сторона',
        'стоять', 'страна', 'суть', 'считать', 'т', 'та', 'так', 'такая',
        'также', 'таки', 'такие', 'такое', 'такой', 'там', 'твои', 'твой',
        'твоя', 'твоё', 'те', 'тебе', 'тебя', 'тем', 'теми', 'теперь',
        'тех', 'то', 'тобой', 'тобою', 'товарищ', 'тогда', 'того', 'тоже',
        'только', 'том', 'тому', 'тот', 'тою', 'третий', 'три', 'тринадцатый',
        'тринадцать', 'ту', 'туда', 'тут', 'ты', 'тысяч', 'у', 'увидеть',
        'уж', 'уже', 'улица', 'уметь', 'утро', 'хороший', 'хорошо',
        'хотел бы', 'хотеть', 'хоть', 'хотя', 'хочешь', 'час', 'часто',
        'часть', 'чаще', 'чего', 'человек', 'чем', 'чему', 'через', 'четвертый',
        'четыре', 'четырнадцатый', 'четырнадцать', 'что', 'чтоб', 'чтобы',
        'чуть', 'шестнадцатый', 'шестнадцать', 'шестой', 'шесть', 'эта',
        'эти', 'этим', 'этими', 'этих', 'это', 'этого', 'этой', 'этом',
        'этому', 'этот', 'эту', 'я', 'являюсь'
    }

# --- Модели и токенизатор ---
@st.cache_resource
def load_model_and_tokenizer(model_path):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

class PredictionDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        if not isinstance(text, str):
            text = str(text)
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }

# --- Предсказание меток ---
def predict_labels(texts, model_path="/workspaces/Pre-diploma-internship-micro/best_model"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = load_model_and_tokenizer(model_path)
    model.to(device)
    model.eval()

    dataset = PredictionDataset(texts=texts, tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    all_active_labels = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs_batch = torch.sigmoid(outputs.logits).cpu().numpy()
            probs_batch = probs_batch.squeeze(1) if probs_batch.ndim == 3 else probs_batch
            adaptive_thresholds = np.max(probs_batch, axis=1) * SCALING_FACTOR
            active_labels_batch = (probs_batch > adaptive_thresholds[:, np.newaxis]).astype(int)
            active_indices_batch = [
                ",".join(map(str, (np.where(active_labels)[0] + 1)))
                if np.any(active_labels) else ""
                for active_labels in active_labels_batch
            ]
        all_active_labels.extend(active_indices_batch)

    return all_active_labels

#  Функция для преобразования графика в байты
def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', dpi=100)
    plt.close(fig)
    buf.seek(0)
    return buf

# --- Генерация графиков ---
def generate_graphs(output_df):
    labels_path = "Классификации.xlsx"
    labels_df = pd.read_excel(labels_path)
    label_to_category = dict(zip(labels_df['level_3'], labels_df['level_2']))

    def map_to_category(labels_str):
        if not isinstance(labels_str, str) or not labels_str.strip():
            return []
        labels = [label.strip() for label in labels_str.split(";")]
        categories = set(label_to_category[label] for label in labels if label in label_to_category)
        return list(categories)

    output_df['categories'] = output_df['predicted_labels'].apply(map_to_category)
    exploded_categories = output_df.explode('categories')
    category_counts = exploded_categories['categories'].value_counts().sort_values(ascending=False)
    sorted_categories = category_counts.sort_values(ascending=True)

    colors = ['#fd7e14', '#6c757d', '#17a2b8', '#dc3545', '#ffc107', '#28a745', '#007bff']

    # === График 1: Распределение по категориям ===
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    bars = ax1.barh(sorted_categories.index, sorted_categories.values, color=colors[:len(sorted_categories)], height=0.5)
    ax1.grid(axis='x', color='gray', linestyle='--', linewidth=0.7, zorder=1)
    ax1.set_facecolor('none')

    graph1_buffer = fig_to_bytes(fig1)

    # === График 2: Распределение по подкатегориям ===
    subcategory_counts = output_df['predicted_labels'].str.split('; ').explode().value_counts().head(10)

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    bars = ax2.barh(subcategory_counts.index[::-1], subcategory_counts.values[::-1], color='#BA68C8', height=0.5)
    ax2.grid(axis='x', color='gray', linestyle='--', linewidth=0.7, zorder=1)
    ax2.set_facecolor('none')

    graph2_buffer = fig_to_bytes(fig2)

    return graph1_buffer, graph2_buffer

# --- Генерация облака слов ---
def generate_wordcloud(output_df):
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'[^а-яА-Яa-zA-Z\s]', '', text)
        words = text.split()
        filtered_words = [word for word in words if word not in STOP_WORDS and len(word) > 2]
        return ' '.join(filtered_words)

    output_df['cleaned_text'] = output_df['text'].apply(clean_text)
    all_cleaned_text = ' '.join(output_df['cleaned_text'])

    if not all_cleaned_text.strip():
        all_cleaned_text = "Нет данных"

    wordcloud = WordCloud(width=1500, height=800, background_color='white', max_words=100).generate(all_cleaned_text)

    # Преобразуем WordCloud в BytesIO
    wordcloud_buffer = io.BytesIO()
    wordcloud.to_image().save(wordcloud_buffer, format="PNG")
    wordcloud_buffer.seek(0)

    return wordcloud_buffer

# --- Обработка файла ---
def process_uploaded_file(df):
    texts = df.iloc[:, 0].values
    predicted_indices = predict_labels(texts)

    output_df = pd.DataFrame({"text": texts, "predicted_labels": predicted_indices})

    labels_df = pd.read_excel("Классификации.xlsx")
    label_dict = dict(zip(labels_df['id'], labels_df['level_3']))

    def map_labels(label_str, label_map):
        try:
            ids = [int(x.strip()) for x in str(label_str).split(",")]
            return "; ".join([label_map[id] for id in ids if id in label_map])
        except Exception as e:
            return ""

    output_df['predicted_labels'] = output_df['predicted_labels'].apply(lambda x: map_labels(x, label_dict))

    output_df = output_df[['text', 'predicted_labels']]
    return output_df



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
1️⃣ **Загрузите файл**  
   Нажмите кнопку «Выберите файл» и выберите подходящий файл в формате `.csv`. Убедитесь, что файл содержит текстовые данные.

2️⃣ **Дождитесь обработки**  
   После загрузки начнётся автоматическая обработка данных. Прогресс будет отображаться на экране.

3️⃣ **Просмотрите и скачайте результаты**  
   По завершении вы увидите таблицу с результатами анализа. Также станет доступен для скачивания обработанный CSV-файл.
""")

# Инициализация session_state
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None
    
st.markdown("### Загрузите файл для обработки")
uploaded_file = st.file_uploader("Выберите CSV-файл", type=["csv"], label_visibility="collapsed")

if uploaded_file is not None:
    if st.button("Начать обработку"):
        try:
            df = pd.read_csv(uploaded_file)
            status_text = st.empty()
            progress_bar = st.progress(0)

            status_text.info("Запуск модели...")
            processed_df = process_uploaded_file(df)

            status_text.info("Генерация графиков...")
            wordcloud_buffer = generate_wordcloud(processed_df)
            graph1_buffer, graph2_buffer = generate_graphs(processed_df)

            # Сохраняем всё в session_state, чтобы не пересчитывать при скачивании
            st.session_state.processed_df = processed_df
            st.session_state.wordcloud_buffer = wordcloud_buffer
            st.session_state.graph1_buffer = graph1_buffer
            st.session_state.graph2_buffer = graph2_buffer

            status_text.success("Файл успешно обработан!")
            progress_bar.progress(100)

        except Exception as e:
            st.error(f"Ошибка: {e}")

# Показываем результаты, если они уже есть в session_state
if st.session_state.processed_df is not None:
    processed_df = st.session_state.processed_df
    


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
    st.markdown("""
        <style>
        .badge-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }
        .badge-table th, .badge-table td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        .badge-table tr:hover {
            background-color: #f9f9f9;
        }

        .st-emotion-cache-16tyu1 th{
            text-align: left;
        }

        td {
            vertical-align: top;
        }
        </style>
    """, unsafe_allow_html=True)

    # Конвертация DataFrame в HTML и вывод
    html_table = display_df.to_html(index=False, escape=False)
    st.markdown(f'<table class="badge-table">{html_table}</table>', unsafe_allow_html=True)


    st.markdown("## Результаты обработки (первые 10 строк):")
    st.dataframe(processed_df[['text', 'predicted_labels']].head(10))

    # Скачивание CSV
    csv_ready_df = processed_df[['text', 'predicted_labels']]
    csv_data = csv_ready_df.to_csv(index=False).encode()
    st.download_button(
        label="Скачать обработанный файл",
        data=csv_data,
        file_name="processed_data.csv",
        mime="text/csv"
    )

    # Графики
    if st.session_state.wordcloud_buffer:
        st.markdown("#### ☁️ Облако слов")
        st.image(st.session_state.wordcloud_buffer, use_container_width=True)

    if st.session_state.graph1_buffer:
        st.markdown("#### 📊 Распределение по категориям")
        st.image(st.session_state.graph1_buffer, use_container_width=True)

    if st.session_state.graph2_buffer:
        st.markdown("#### 📈 Распределение по подкатегориям")
        st.image(st.session_state.graph2_buffer, use_container_width=True)