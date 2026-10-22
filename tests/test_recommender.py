import lightgbm as lgb
import numpy as np
import pandas as pd

from hmrec.ranker import top_k_lists
from hmrec.recommender import FALLBACK, RANKED, Recommender
from hmrec.store import Store

NAMES = ["a", "b", "c"]


def train_model(constant=False):
    rng = np.random.default_rng(0)
    features = rng.random((500, 3)).astype("float32")
    labels = np.ones(500) if constant else features[:, 0].copy()
    params = {
        "objective": "regression",
        "num_leaves": 16,
        "min_data_in_leaf": 5,
        "verbose": -1,
        "seed": 0,
    }
    return lgb.train(params, lgb.Dataset(features, labels), num_boost_round=50)


def make_store():
    first = [0.1, 0.9, 0.5, 0.3, 0.2, 0.8]
    features = np.zeros((6, 3), dtype="float32")
    features[:, 0] = first
    return Store(
        customer_ids=np.array(["c1", "c2"]),
        offsets=np.array([0, 4, 6]),
        article_ids=np.array([11, 12, 13, 14, 21, 22], dtype="int64"),
        features=features,
        feature_names=NAMES,
        fallback=np.array([901, 902, 903], dtype="int64"),
    )


def test_recommendations_follow_the_model_score():
    result = Recommender(train_model(), make_store(), k=4).recommend("c1")
    assert result["source"] == RANKED
    assert result["article_ids"] == [12, 13, 14, 11]


def test_unknown_customer_gets_the_popular_fallback():
    result = Recommender(train_model(), make_store(), k=2).recommend("nobody")
    assert result["source"] == FALLBACK
    assert result["article_ids"] == [901, 902]


def test_list_is_cut_to_k():
    result = Recommender(train_model(), make_store(), k=2).recommend("c1")
    assert result["article_ids"] == [12, 13]


def test_equal_scores_keep_candidate_order():
    result = Recommender(train_model(constant=True), make_store(), k=4).recommend("c1")
    assert result["article_ids"] == [11, 12, 13, 14]


def test_recommend_many_matches_single_requests():
    recommender = Recommender(train_model(), make_store(), k=3)
    many = recommender.recommend_many(["c2", "nobody", "c1"])
    assert many == [recommender.recommend(c) for c in ["c2", "nobody", "c1"]]


def test_response_has_the_expected_fields():
    result = Recommender(train_model(), make_store()).recommend("c2")
    assert set(result) == {"customer_id", "source", "article_ids"}
    assert result["customer_id"] == "c2"


def test_online_lists_match_the_offline_top_k_function():
    model = train_model()
    store = make_store()
    frame = pd.DataFrame(
        {
            "customer_idx": [0, 0, 0, 0, 1, 1],
            "article_id": store.article_ids,
        }
    )
    offline = top_k_lists(frame, model.predict(store.features), k=3)
    recommender = Recommender(model, store, k=3)
    assert recommender.recommend("c1")["article_ids"] == offline[0]
    assert recommender.recommend("c2")["article_ids"] == offline[1]