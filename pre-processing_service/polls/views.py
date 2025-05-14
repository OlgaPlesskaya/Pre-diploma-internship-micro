# polls/views.py
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from .models import Category, Up_Fle
from .forms import UploadFileForm
import numpy as np
from tqdm import tqdm
import tempfile
from django.core.files import File
from io import StringIO
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.core.cache import cache
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import time

# Параметры модели
MAX_LENGTH = 150
SCALING_FACTOR = 0.7
PROGRESS_KEY = 'file_processing_progress'

def set_progress(percent):
    cache.set(PROGRESS_KEY, percent, timeout=600)  # Храним до 10 минут

def check_model_files():
    model_dir = '/workspaces/Pre-diploma-internship/pre-processing_service/polls/best_model'
    required_files = ['config.json', 'vocab.txt', 'model.safetensors', 'tokenizer_config.json', 'special_tokens_map.json']
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_dir, f))]
    if missing_files:
        raise FileNotFoundError(f"Отсутствуют файлы модели: {', '.join(missing_files)}")
    print("✅ Все файлы модели найдены")

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

def load_model_and_tokenizer():
    try:
        check_model_files()
        tokenizer = BertTokenizer.from_pretrained('/workspaces/Pre-diploma-internship/pre-processing_service/polls/best_model')
        model = BertForSequenceClassification.from_pretrained('/workspaces/Pre-diploma-internship/pre-processing_service/polls/best_model')
        return tokenizer, model
    except Exception as e:
        print(f"⚠️ Ошибка: {str(e)}. Убедитесь, что все файлы модели находятся в папке best_model.")
        raise

def generate_graphs(output_df):
    """
    Генерирует два графика и возвращает их URL-адреса.
    """
    static_dir = '/workspaces/Pre-diploma-internship/pre-processing_service/polls/static/images'
    
    
    labels_path = os.path.join(os.path.dirname(__file__), 'Классификации.xlsx')
    labels_df = pd.read_excel(labels_path)
    required_columns = ['level_3', 'level_2']
    label_to_category = dict(zip(labels_df['level_3'], labels_df['level_2']))
    
    def map_to_category(labels_str):
        if not isinstance(labels_str, str) or not labels_str.strip():
            return []
        labels = [label.strip() for label in labels_str.split(";")]
        categories = set(label_to_category[label] for label in labels if label in label_to_category)
        return list(categories)

    output_df['categories'] = output_df['predicted_labels'].apply(map_to_category)
    
    
    
    # Считаем частоты
    exploded_categories = output_df.explode('categories')
    category_counts = exploded_categories['categories'].value_counts().sort_values(ascending=False)
    
    # Сортируем категории по убыванию
    sorted_categories = category_counts.sort_values(ascending=True)
    
    # Цветовая гамма
    colors = ['#fd7e14', '#6c757d', '#17a2b8', '#dc3545', '#ffc107', '#28a745', '#007bff']
    
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='none')  # Устанавливаем прозрачный фон для всей фигуры
    
    # Рисуем бары с zorder=2 (они будут выше сетки)
    bars = ax.barh(sorted_categories.index, sorted_categories.values,
                   color=colors[:len(sorted_categories)], zorder=2, height=0.5)
    
    # Настраиваем сетку с zorder=1 (она будет под барами)
    ax.grid(axis='x', color='gray', linestyle='--', linewidth=0.7, zorder=1)
    
    # Дополнительные настройки
    ax.tick_params(axis='both', colors='black')
    
    # Прозрачный фон
    ax.set_facecolor('none')
    
    graph1_path = os.path.join(static_dir, 'graph1.png')
    
    plt.savefig(str(graph1_path), transparent=True, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    # График 2: Распределение по подкатегориям (топ-10)
    subcategory_counts = output_df['predicted_labels'].str.split('; ').explode().value_counts().head(10)
    
    # Цвет всех баров
    bar_color = '#007bff'
    
    # Рисуем график
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='none')

    bars = ax.barh(subcategory_counts.index[::-1], subcategory_counts.values[::-1],
                   color=bar_color, zorder=2, height=0.5)
                   
                   

    # Настраиваем сетку
    ax.grid(axis='x', color='gray', linestyle='--', linewidth=0.7, zorder=1)
    
    # Только целые числа на оси X
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.tick_params(axis='both', colors='black')

    # Прозрачный фон
    ax.set_facecolor('none')
    
    graph2_path = os.path.join(static_dir, 'graph2.png')
    plt.savefig(str(graph2_path), transparent=True, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    # Возвращаем URL-адреса графиков
    
    # 🔥 Добавляем временную метку к URL, чтобы отключить кэширование
    timestamp = int(time.time())
    
    # Возвращаем URL с параметром v={timestamp}
    graph1_url = '/static/images/graph1.png?v=' + str(timestamp)
    graph2_url = '/static/images/graph2.png?v=' + str(timestamp)
    
    return graph1_url, graph2_url


def generate_wordcloud(output_df):
    """
    Генерирует облако слов на основе текстов из DataFrame,
    исключая предлоги и служебные слова.
    """
    # Список стоп-слов
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

    # Функция для очистки и фильтрации текста
    def clean_text(text):
        text = str(text).lower()  # Приводим к нижнему регистру
        text = re.sub(r'[^а-яА-Яa-zA-Z\s]', '', text)  # Убираем символы, кроме букв
        words = text.split()
        filtered_words = [word for word in words if word not in STOP_WORDS and len(word) > 2]
        return ' '.join(filtered_words)

    # Очищаем все тексты
    output_df['cleaned_text'] = output_df['text'].apply(clean_text)
    all_cleaned_text = ' '.join(output_df['cleaned_text'])

    # Проверяем, есть ли текст для облака
    if not all_cleaned_text.strip():
        # Если текст пустой, создаём заглушку
        all_cleaned_text = "Нет_данных"

    # Создаем объект WordCloud
    wordcloud = WordCloud(
        width=1000,
        height=600,
        background_color='white',
        max_words=100
    ).generate(all_cleaned_text)

    # Сохраняем облако слов в файл
    static_dir = '/workspaces/Pre-diploma-internship/pre-processing_service/polls/static/images'
    wordcloud_path = os.path.join(static_dir, 'wordcloud.png')
    wordcloud.to_file(wordcloud_path)
    
    # 🔥 Добавляем временную метку к URL, чтобы отключить кэширование
    timestamp = int(time.time())
    
    # Возвращаем URL с параметром v={timestamp}
    wordcloud = '/static/images/wordcloud.png?v=' + str(timestamp)
    
    # Возвращаем URL-адрес облака слов
    return wordcloud

@csrf_exempt
def upload_file(request):
    html_table = ''
    headers = []  # <-- добавляем здесь
    table_data = []
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            # Сохраняем файл
            up_file = form.save()

            # Загружаем модель и токенизатор
            tokenizer, model = load_model_and_tokenizer()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()

            # Читаем CSV без заголовка
            file_path = default_storage.path(up_file.file.name)
            data = pd.read_csv(file_path, header=None)
            texts = data.iloc[:, 0].values

            # Подготавливаем данные
            dataset = PredictionDataset(texts=texts, tokenizer=tokenizer)
            dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

            all_active_labels = []
            for batch in tqdm(dataloader, desc="Предсказание"):
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
            
          
            # Формируем DataFrame
            output_df = pd.DataFrame({
                "text": texts,
                "predicted_labels": all_active_labels
            })
            
            
            # Путь к Excel
            labels_path = os.path.join(os.path.dirname(__file__), 'Классификации.xlsx')

            if not os.path.exists(labels_path):
                raise FileNotFoundError(f"Файл {labels_path} не найден.")

            labels_df = pd.read_excel(labels_path)

            # Проверяем наличие нужных колонок
            required_columns = ['id', 'level_3']
            missing_cols = [col for col in required_columns if col not in labels_df.columns]
            if missing_cols:
                raise KeyError(f"В Excel-файле отсутствуют колонки: {missing_cols}")

            label_dict = dict(zip(labels_df['id'], labels_df['level_3']))

            def map_labels(label_str, label_map):
                try:
                    ids = [int(x.strip()) for x in str(label_str).split(",")]
                    return "; ".join([label_map[id] for id in ids if id in label_map])
                except Exception as e:
                    print("Ошибка при маппинге меток:", e)
                    return ""

            output_df['predicted_labels'] = output_df['predicted_labels'].apply(lambda x: map_labels(x, label_dict))
            
            

            # === Сохраняем в CSV в памяти ===
            csv_buffer = StringIO()
            output_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
            
            # Создаем ContentFile из строки
            csv_content = ContentFile(csv_buffer.getvalue().encode('utf-8'))
            
            # Указываем имя файла (можно любое, главное — с расширением .csv)
            file_name = f"processed_{up_file.file.name.split('/')[-1]}.csv"
            
            # Сохраняем файл в поле processed_file
            up_file.processed_file.save(file_name, csv_content)
            
            # Сохраняем изменения в БД
            up_file.save()
            
            
            # Формируем preview (первые 10 строк)
            preview = output_df.head(10).to_dict(orient='records')
    
            # Получаем URL файла
            download_url = up_file.processed_file.url
            
            # Генерируем графики
            graph1_url, graph2_url = generate_graphs(output_df)
            
            
            # Генерируем облако слов
            wordcloud_url = generate_wordcloud(output_df)
    
            return JsonResponse({
            'preview': preview,
            'download_url': download_url,
            'graph1_url': graph1_url,
            'graph2_url': graph2_url,
            'wordcloud_url': wordcloud_url
            
            })
            
        else:
            return JsonResponse({"error": form.errors}, status=400)
    else:
        form = UploadFileForm()
    categories = Category.objects.all()
    return render(request, 'polls/index.html', {'categories': categories, 'form': form, })