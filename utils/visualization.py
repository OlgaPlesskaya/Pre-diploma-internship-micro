# utils/visualization.py
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import io
import pandas as pd
import re
from .api_client import get_subcategory_to_category_mapping

import plotly.express as px
import plotly.graph_objects as go


from config.settings import STOP_WORDS


def generate_graphs(output_df):
    label_to_category = get_subcategory_to_category_mapping()

    def map_to_category(labels_str):
        if not isinstance(labels_str, str) or not labels_str.strip():
            return []
        labels = [label.strip() for label in labels_str.split(";")]
        categories = set(label_to_category[label] for label in labels if label in label_to_category)
        return list(categories)

    # Категории
    output_df['categories'] = output_df['predicted_labels'].apply(map_to_category)
    exploded_categories = output_df.explode('categories')
    category_counts = exploded_categories['categories'].value_counts().sort_values(ascending=False)
    sorted_categories = category_counts.sort_values(ascending=True)
    
    # График 1: категории
    fig1 = px.bar(
        x=sorted_categories.values,
        y=sorted_categories.index,
        orientation='h',
        #title="Распределение по категориям",
        #labels={'x': 'Количество', 'y': 'Категории'},
        color_discrete_sequence=['#fd7e14', '#6c757d', '#17a2b8', '#dc3545', '#ffc107', '#28a745', '#007bff']
    )
    fig1.update_layout(height=400, margin=dict(l=100))
    fig1.update_xaxes(gridcolor='lightgray', gridwidth=0.7)
    fig1.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    
    # График 2: подкатегории
    subcategory_counts = output_df['predicted_labels'].str.split('; ').explode().value_counts().head(10)
    fig2 = px.bar(
        x=subcategory_counts.values[::-1],
        y=subcategory_counts.index[::-1],
        orientation='h',
        #title="Топ-10 подкатегорий",
        #labels={'x': 'Количество', 'y': 'Подкатегории'},
        color_discrete_sequence=['#BA68C8']
    )
    fig2.update_layout(height=400, margin=dict(l=100))
    fig2.update_xaxes(gridcolor='lightgray', gridwidth=0.7)
    fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')

    return fig1, fig2




def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^а-яА-Яa-zA-Z\s]', '', text)
    words = text.split()
    filtered_words = [word for word in words if word not in STOP_WORDS and len(word) > 2]
    return ' '.join(filtered_words)

def generate_wordcloud(output_df):
    output_df['cleaned_text'] = output_df['text'].apply(clean_text)
    all_cleaned_text = ' '.join(output_df['cleaned_text'])

    if not all_cleaned_text.strip():
        all_cleaned_text = "Нет данных"

    # Генерация облака слов
    wordcloud = WordCloud(
        width=1200,
        height=500,
        background_color='white',
        max_words=100,
    ).generate(all_cleaned_text)

    # Преобразуем изображение в объект PIL и сохраняем в байты
    image = wordcloud.to_image()

    # Создаем Plotly-график с изображением
    fig = go.Figure()

    fig.add_layout_image(
        dict(
            source=image,
            x=0,
            y=1,
            xref="paper",
            yref="paper",
            sizex=1,
            sizey=1,
            sizing="contain",
            opacity=1
        )
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        height=400
    )

    return fig