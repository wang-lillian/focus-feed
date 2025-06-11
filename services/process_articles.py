from fetch_articles import fetch_articles
from newspaper import Article
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from hashlib import sha256
from dotenv import load_dotenv
import os

load_dotenv()
model = SentenceTransformer("msmarco-MiniLM-L6-cos-v5")
elastic_index_name = "articles_index"


def index_articles(user_interest: str) -> None:
    articles_to_process = fetch_articles(user_interest)
    docs_to_index = []
    for article in articles_to_process:
        docs = process_article(article)
        docs_to_index.extend(docs)

    es = prepare_index()
    bulk(
        es,
        bulk_actions(docs_to_index),
        stats_only=True,
        raise_on_error=False,
        ignore_status=[409],
    )


def prepare_index():
    elastic_cloud_id = os.getenv("ELASTIC_CLOUD_ID")
    elastic_api_key = os.getenv("ELASTIC_API_KEY")

    try:
        es = Elasticsearch(cloud_id=elastic_cloud_id, api_key=elastic_api_key)
    except Exception as e:
        raise Exception("Failed to connect to Elasticsearch: " + str(e))

    mappings = {
        "dynamic": "false",
        "properties": {
            "title": {"type": "text", "index": "false"},
            "description": {"type": "text", "index": "false"},
            "url": {"type": "text", "index": "false"},
            "title_desc": {"type": "text", "index": "true"},
            "content_vector": {
                "type": "dense_vector",
                "dims": 384,
                "index": "true",
                "similarity": "cosine",
                "index_options": {"type": "int4_hnsw"},
            },
        },
    }

    if not es.indices.exists(index=elastic_index_name):
        es.indices.create(index=elastic_index_name, mappings=mappings)

    return es


def bulk_actions(docs_to_index: list, index_name=elastic_index_name):
    for doc in docs_to_index:
        raw_id = f"{doc["url"]}_{doc["chunk_index"]}"
        doc_id = sha256(raw_id.encode()).hexdigest()

        doc_for_indexing = {
            "_index": index_name,
            "_id": doc_id,
            "_op_type": "create",
            "_source": doc,
        }

        yield doc_for_indexing


def process_article(article: dict) -> list[dict]:
    content = extract_content(article)
    chunks = chunk_content(content)
    docs = create_documents(article, chunks)
    return docs


def extract_content(article: dict) -> str:
    url = article["url"]
    article_obj = Article(url=url)
    article_obj.download()
    article_obj.parse()
    return article_obj.text


def chunk_content(content: str) -> list[str]:
    words = content.split()
    chunks = []
    chunk_size = 300
    overlap = 60
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks


def create_documents(article: dict, chunks: list[str]) -> list[dict]:
    docs = []
    for index, chunk in enumerate(chunks):
        docs.append(
            {
                "title": article["title"],
                "description": article["description"],
                "title_desc": f"{article["title"]} {article["description"]}",
                "url": article["url"],
                "chunk_index": index,
                "content_vector": model.encode(chunk).tolist(),
            }
        )
    return docs
