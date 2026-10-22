import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from hmrec.recommender import load_recommender

DEFAULT_MODEL = "data/ranker/processed_200k_16w/ranker.txt"
DEFAULT_STORE = "data/store/processed_200k_16w"
MAX_BATCH = 100


class RecommendResponse(BaseModel):
    customer_id: str
    source: str
    article_ids: list[int]


class BatchRequest(BaseModel):
    customer_ids: list[str] = Field(min_length=1, max_length=MAX_BATCH)


def create_app(recommender=None):
    @asynccontextmanager
    async def lifespan(app):
        if recommender is None:
            app.state.recommender = load_recommender(
                os.environ.get("HMREC_MODEL", DEFAULT_MODEL),
                os.environ.get("HMREC_STORE", DEFAULT_STORE),
            )
        else:
            app.state.recommender = recommender
        yield

    app = FastAPI(title="H&M two-stage recommender", lifespan=lifespan)

    @app.get("/health")
    def health():
        current = app.state.recommender
        return {"status": "ok", "customers_in_store": len(current.store), "k": current.k}

    @app.get("/recommend/{customer_id}", response_model=RecommendResponse)
    def recommend(customer_id: str):
        return app.state.recommender.recommend(customer_id)

    @app.post("/recommend", response_model=list[RecommendResponse])
    def recommend_batch(request: BatchRequest):
        return app.state.recommender.recommend_many(request.customer_ids)

    return app


app = create_app()