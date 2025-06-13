from process_articles import es, model, elastic_index_name, index_articles


def search(user_interest: str) -> list[dict]:
    query_vector = model.encode(user_interest).tolist()
    response = es.search(
        index=elastic_index_name,
        body={
            "size": 5,
            "collapse": {"field": "url"},
            "retriever": {
                "rrf": {
                    "retrievers": [
                        {
                            "standard": {
                                "query": {
                                    "multi_match": {
                                        "query": user_interest,
                                        "fields": ["title^0.3", "description^0.3"],
                                    }
                                }
                            }
                        },
                        {
                            "knn": {
                                "field": "content_vector",
                                "query_vector": query_vector,
                                "k": 20,
                                "num_candidates": 60,
                            }
                        },
                    ],
                    "rank_window_size": 20,
                }
            },
            "_source": ["title", "description", "url"],
        },
    )

    hits = response["hits"]["hits"]
    results = [
        {
            "title": hit["_source"]["title"],
            "description": hit["_source"]["description"],
            "url": hit["_source"]["url"],
        }
        for hit in hits
    ]

    return results
