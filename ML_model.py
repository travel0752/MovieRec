import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pymorphy3
from movies_data import all_movies, _unique_movies

morph = pymorphy3.MorphAnalyzer()

vectorizer = None
movies_tfidf_matrix = None

STOP_WORDS = set([
    "хочу", "посмотреть", "фильм", "кино", "сериал", "про", "чтобы", "был", "очень",
    "и", "в", "во", "на", "под", "из", "с", "по", "а", "но", "или", "да", "как",
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
    "about", "movie", "film", "watch", "want", "like", "see", "good", "best"
])


def advanced_clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zа-яё0-9\s]', ' ', text)
    words = text.split()
    cleaned = []
    for w in words:
        if w in STOP_WORDS or len(w) < 2:
            continue
        if re.match(r'[а-яё]', w):
            try:
                cleaned.append(morph.parse(w)[0].normal_form)
            except:
                cleaned.append(w)
        else:
            cleaned.append(w)
    return " ".join(cleaned)


def init_ml():
    global vectorizer, movies_tfidf_matrix

    if vectorizer is not None:
        return

    print(f"Загружено {len(all_movies)} фильмов, {len(_unique_movies)} уникальных")

    descriptions = [advanced_clean_text(m['description']) for m in all_movies]

    print("Строим TF-IDF матрицу...")
    vectorizer = TfidfVectorizer(max_features=5000, min_df=1, max_df=0.9)
    movies_tfidf_matrix = vectorizer.fit_transform(descriptions)
    print(f"Готово! Матрица: {movies_tfidf_matrix.shape}")


def get_recommendations(user_query, top_n=10):
    global vectorizer, movies_tfidf_matrix

    init_ml()

    cleaned_query = advanced_clean_text(user_query)
    if not cleaned_query.strip():
        return get_random_movies(top_n)

    query_vector = vectorizer.transform([cleaned_query])
    scores = cosine_similarity(query_vector, movies_tfidf_matrix).flatten()

    # Сортируем по релевантности
    top_indices = np.argsort(scores)[::-1]

    result = []
    seen_originals = set()
    for idx in top_indices:
        if len(result) >= top_n:
            break
        movie = all_movies[idx]
        # Проверяем по original_title чтобы не было дублей
        if movie['original_title'] not in seen_originals:
            seen_originals.add(movie['original_title'])
            result.append({
                'id': movie['id'],
                'title': movie['title'],
                'original_title': movie['original_title'],
                'description': movie['description']
            })

    return result


def get_random_movies(top_n=10):
    init_ml()

    import random
    # Берем случайные из уникальных фильмов
    shuffled = random.sample(_unique_movies, len(_unique_movies))
    return shuffled[:top_n]


init_ml()