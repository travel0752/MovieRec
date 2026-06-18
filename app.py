from flask import Flask, render_template, request, jsonify
import requests
import random
from ML_model import get_recommendations, get_random_movies

app = Flask(__name__)

TMDB_API_KEY = "f68b4f2826209c8254c32463c31ea7e8"
TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG = "https://image.tmdb.org/t/p/w400"

_poster_cache = {}


def get_tmdb_poster(title):
    if not TMDB_API_KEY:
        return None
    if title in _poster_cache:
        return _poster_cache[title]

    try:
        url = f"{TMDB_BASE}/search/movie"
        params = {
            "api_key": TMDB_API_KEY,
            "query": title,
            "language": "en-US"
        }
        res = requests.get(url, params=params, timeout=5).json()
        if res.get('results'):
            path = res['results'][0].get('poster_path')
            if path:
                full_url = f"{TMDB_IMG}{path}"
                _poster_cache[title] = full_url
                return full_url
    except Exception as e:
        print(f"Ошибка TMDB для {title}: {e}")

    return None


def enrich_movies(movies_list):
    for m in movies_list:
        poster = get_tmdb_poster(m['original_title'])
        m['poster'] = poster
    return movies_list


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/random', methods=['GET'])
def api_random_movies():
    movies = get_random_movies(top_n=10)
    if not movies:
        return jsonify([])
    return jsonify(enrich_movies(movies))


@app.route('/api/recommend', methods=['POST'])
def recommend_api():
    data = request.get_json() or {}
    query = data.get('query', '')
    is_random = data.get('random', False)

    if not query or not query.strip():
        return jsonify([])

    if is_random:
        movies = get_random_movies(top_n=10)
    else:
        movies = get_recommendations(query, top_n=10)

    return jsonify(enrich_movies(movies))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=54321)