# utils/visualization.py
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import io
import pandas as pd
import re

def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', dpi=100)
    plt.close(fig)
    buf.seek(0)
    return buf


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

    fig1, ax1 = plt.subplots(figsize=(12, 6))
    bars = ax1.barh(sorted_categories.index, sorted_categories.values, color=colors[:len(sorted_categories)], height=0.5)
    ax1.grid(axis='x', color='gray', linestyle='--', linewidth=0.7, zorder=1)
    ax1.set_facecolor('none')
    graph1_buffer = fig_to_bytes(fig1)

    subcategory_counts = output_df['predicted_labels'].str.split('; ').explode().value_counts().head(10)
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    bars = ax2.barh(subcategory_counts.index[::-1], subcategory_counts.values[::-1], color='#BA68C8', height=0.5)
    ax2.grid(axis='x', color='gray', linestyle='--', linewidth=0.7, zorder=1)
    ax2.set_facecolor('none')
    graph2_buffer = fig_to_bytes(fig2)

    return graph1_buffer, graph2_buffer


def generate_wordcloud(output_df):
    from config.settings import STOP_WORDS

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
    wordcloud_buffer = io.BytesIO()
    wordcloud.to_image().save(wordcloud_buffer, format="PNG")
    wordcloud_buffer.seek(0)
    return wordcloud_buffer