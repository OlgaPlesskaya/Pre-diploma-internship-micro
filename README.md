# 🧠 Сервис предобработки текстовых сообщений  
> Преддипломная практика: Реализация микросервиса с интеграцией модели машинного обучения

## Описание проекта

Это Django-проект, реализующий **микросервисную архитектуру** для обработки текстовых данных. Проект включает:
- Загрузку CSV-файлов.
- Обработку текста с помощью ML-модели (на основе BERT).
- Классификацию по категориям и подкатегориям.
- Визуализацию результатов (облако слов, графики).
- API на основе DRF с документацией Swagger / ReDoc.
- Streamlit-интерфейс для взаимодействия с пользователем.
- Django-админка для управления данными.

---

## 📁 Структура проекта

```
Pre-diploma-internship-micro/
├── best_model/                # ML-модель
│   ├── config.json
│   ├── model.safetensors
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── vocab.txt
├── config/
│   └── settings.py            # Конфигурационные параметры проекта
├── pre-processing_service/    # Django-приложение
│   ├── mysite/
│   ├── polls/
│   │   ├── models.py          # Модели БД: Category, Subcategory, Up_Fle
│   │   ├── views.py          
│   │   └── urls.py
│   ├── uploads/               # Папка для загрузки файлов
│   ├── manage.py
│   └── db.sqlite3             # SQLite база данных
├── utils/
│   ├── api_client.py          # Работа с API (загрузка файлов, категории)
│   ├── data_processor.py      # Обработка данных и предсказания от модели
│   ├── model_loader.py        # Загрузка модели и токенизатора
│   └── visualization.py       # Генерация графиков и облака слов
├── streamlit_app.py           # Интерфейс пользователя (Streamlit)
├── requirements.txt           # Зависимости Python
├── LICENSE
└── README.md
```

---

## 🔧 Технологии

- **Python 3.8+**
- **Django** – веб-фреймворк
- **Django REST Framework (DRF)** – API
- **drf-yasg** – генерация Swagger UI / ReDoc
- **Transformers** – модель BERT
- **PyTorch / TorchVision** – работа с моделью
- **Pandas / NumPy** – обработка данных
- **Streamlit** – интерфейс пользователя
- **Plotly / WordCloud** – визуализация
- **SQLite** – локальная БД (по умолчанию)

---

## 🚀 Как запустить проект

### 1. Клонировать репозиторий

```bash
git clone https://github.com/OlgaPlesskaya/Pre-diploma-internship-micro.git
cd Pre-diploma-internship-micro
```

### 2. Установить зависимости

```bash
pip install -r requirements.txt
```

### 3. Настроить Django

Перейдите в папку `pre-processing_service` и выполните миграции:

```bash
cd pre-processing_service
python manage.py migrate
```

Запустите сервер:

```bash
python manage.py runserver
```

Откройте [http://localhost:8000](http://localhost:8000) — доступны:
- API: `/api/up_fles/`, `/api/categorys/`, `/api/subcategorys/`
- Документация API: `/swagger/`, `/redoc/`
- Админка: `/admin/` (логин: admin, пароль: admin)

### 4. Запустить Streamlit

В новом терминале (в корне проекта):

```bash
streamlit run streamlit_app.py
```

Откройте [http://localhost:8501](http://localhost:8501) и начните использовать приложение.

---

## 📈 Функциональность

- **Загрузка CSV-файла** с текстами.
- **Автоматическая классификация** текстов по категориям и подкатегориям.
- **Генерация графиков**:
  - Распределение по категориям.
  - Топ-10 подкатегорий.
  - Облако слов из текстов.
- **Сохранение и скачивание** обработанного файла.
- **Полная интеграция с Django API**, что позволяет легко масштабировать систему.
- **Админка Django** для:
  - Управления категориями.
  - Подкатегориями.
  - Загруженными файлами.

---

## 📦 API Endpoints

| Эндпоинт | Метод | Описание |
|---------|-------|----------|
| `/api/up_fles/` | GET, POST, PATCH | Работа с загруженными файлами |
| `/api/categorys/` | GET | Получение списка категорий |
| `/api/subcategorys/` | GET | Получение списка подкатегорий |

API документирован через Swagger: [http://localhost:8000/swagger/](http://localhost:8000/swagger/)  
И через ReDoc: [http://localhost:8000/redoc/](http://localhost:8000/redoc/)

---

## 📷 Скрины

![screencapture-humble-space-pancake-4xv9pj5g9rfj5pv-8501-app-github-dev-2025-05-15-18_56_13](https://github.com/user-attachments/assets/57d410fa-9d88-4daf-a4bd-9324ea2adc1a)

![screencapture-humble-space-pancake-4xv9pj5g9rfj5pv-8501-app-github-dev-2025-05-15-19_43_54](https://github.com/user-attachments/assets/ca576d68-f4b4-4be7-85c7-d715b553ce06)


