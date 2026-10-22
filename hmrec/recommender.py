import lightgbm as lgb
import numpy as np

from hmrec import config
from hmrec.store import load_store

RANKED = "ranker"
FALLBACK = "fallback_popular"


class Recommender:
    def __init__(self, model, store, k=config.K, threads=1):
        self.model = model
        self.store = store
        self.k = k
        self.threads = threads

    def recommend(self, customer_id):
        if customer_id not in self.store:
            return {
                "customer_id": customer_id,
                "source": FALLBACK,
                "article_ids": self.store.fallback[: self.k].tolist(),
            }
        articles, features = self.store.rows(customer_id)
        scores = self.model.predict(features, num_threads=self.threads)
        order = np.argsort(-scores, kind="stable")[: self.k]
        return {
            "customer_id": customer_id,
            "source": RANKED,
            "article_ids": articles[order].tolist(),
        }

    def recommend_many(self, customer_ids):
        return [self.recommend(customer_id) for customer_id in customer_ids]


def load_recommender(model_path, store_dir, k=config.K, threads=1):
    model = lgb.Booster(model_file=str(model_path))
    return Recommender(model, load_store(store_dir), k, threads)