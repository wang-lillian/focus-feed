from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer, util
import json
import urllib.request
from urllib.parse import urlencode
from datetime import datetime, timezone, timedelta
import time

load_dotenv()
gnews_api_key = os.getenv("GNEWS_API_KEY")
now_datetime = datetime.now(timezone.utc)
from_date_datetime = now_datetime - timedelta(weeks=4)
from_date_string = from_date_datetime.strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_articles(user_interest: str) -> list:
    # fetching articles from the top 3 categories related to user_interest
    interest_categories = get_interest_categories(user_interest)
    articles_to_process = []
    for category in interest_categories:
        articles = fetch_top_headlines(category)
        articles_to_process.extend(articles)
        time.sleep(1)  # GNews API: 1 request per second limit

    # fetching articles from direct search
    articles = fetch_top_search_results(user_interest)
    articles_to_process.extend(articles)

    return articles_to_process


def get_interest_categories(user_interest: str) -> list:
    interest = [user_interest]
    categories = [
        "general",
        "world",
        "nation",
        "business",
        "technology",
        "entertainment",
        "sports",
        "science",
        "health",
    ]

    model = SentenceTransformer("all-MiniLM-L6-v2")
    categories_embeddings = model.encode(categories, normalize_embeddings=True)
    interest_embedding = model.encode(interest, normalize_embeddings=True)

    similarities = util.cos_sim(categories_embeddings, interest_embedding).tolist()
    top_similarities_indices = sorted(
        range(len(categories)), key=lambda i: similarities[i], reverse=True
    )[:3]
    interest_categories = [categories[i] for i in top_similarities_indices]
    return interest_categories


def fetch_top_headlines(category: str) -> list:
    params = {
        "category": category,
        "lang": "en",
        "country": "us",
        "max": "10",
        "from": from_date_string,
        "apikey": gnews_api_key,
    }
    gnews_url = f"https://gnews.io/api/v4/top-headlines?{urlencode(params)}"

    try:
        with urllib.request.urlopen(gnews_url) as response:
            data = json.loads(response.read().decode("utf-8"))
            articles = data.get("articles", [])
            return articles
    except Exception as e:
        print(f"Error fetching {category} articles: {str(e)}")
        return []


def fetch_top_search_results(user_interest: str) -> list:
    params = {
        "q": user_interest,
        "lang": "en",
        "country": "us",
        "max": "10",
        "in": "title,description,content",
        "from": from_date_string,
        "sortby": "relevance",
        "apikey": gnews_api_key,
    }
    gnews_url = f"https://gnews.io/api/v4/search?{urlencode(params)}"

    try:
        with urllib.request.urlopen(gnews_url) as response:
            data = json.loads(response.read().decode("utf-8"))
            articles = data.get("articles", [])
            return articles
    except Exception as e:
        print(f"Error searching for articles about {user_interest}: {str(e)}")
        return []