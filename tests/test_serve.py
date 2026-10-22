import lightgbm as lgb
import numpy as np
from fastapi.testclient import TestClient

from hmrec.recommender import Recommender
from hmrec.serve import MAX_BATCH, create_app
from hmrec.store import Store


def train_model():
    rng = np.random.default_rng(0)
    features = rng.random((500, 3)).astype("float32")
    labels = features[:, 0].copy()
    params = {
        "objective": "regression",
        "num_leaves": 16,
        "min_data_in_leaf": 5,
        "verbose": -1,
        "seed": 0,
    }
    return lgb.train(params, lgb.Dataset(features, labels), num_boost_round=50)


def make_store():
    features = np.zeros((6, 3), dtype="float32")
    features[:, 0] = [0.1, 0.9, 0.5, 0.3, 0.2, 0.8]
    return Store(
        customer_ids=np.array(["c1", "c2"]),
        offsets=np.array([0, 4, 6]),
        article_ids=np.array([11, 12, 13, 14, 21, 22], dtype="int64"),
        features=features,
        feature_names=["a", "b", "c"],
        fallback=np.array([901, 902, 903], dtype="int64"),
    )


def make_client():
    recommender = Recommender(train_model(), make_store(), k=4)
    return TestClient(create_app(recommender))


def test_health_reports_store_size_and_k():
    with make_client() as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "customers_in_store": 2, "k": 4}


def test_known_customer_gets_a_ranked_list():
    with make_client() as client:
        response = client.get("/recommend/c1")
    assert response.status_code == 200
    assert response.json() == {
        "customer_id": "c1",
        "source": "ranker",
        "article_ids": [12, 13, 14, 11],
    }


def test_unknown_customer_gets_the_fallback_not_an_error():
    with make_client() as client:
        response = client.get("/recommend/nobody")
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "fallback_popular"
    assert body["article_ids"] == [901, 902, 903]


def test_batch_returns_one_result_per_customer_in_order():
    with make_client() as client:
        response = client.post("/recommend", json={"customer_ids": ["c2", "nobody", "c1"]})
    assert response.status_code == 200
    results = response.json()
    assert [r["customer_id"] for r in results] == ["c2", "nobody", "c1"]
    assert [r["source"] for r in results] == ["ranker", "fallback_popular", "ranker"]


def test_batch_rejects_empty_oversized_and_missing_bodies():
    with make_client() as client:
        empty = client.post("/recommend", json={"customer_ids": []})
        too_many = client.post(
            "/recommend", json={"customer_ids": ["c1"] * (MAX_BATCH + 1)}
        )
        missing = client.post("/recommend")
    assert empty.status_code == 422
    assert too_many.status_code == 422
    assert missing.status_code == 422


def test_response_article_ids_are_integers():
    with make_client() as client:
        body = client.get("/recommend/c2").json()
    assert all(isinstance(article, int) for article in body["article_ids"])
    assert set(body) == {"customer_id", "source", "article_ids"}