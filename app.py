from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import logging
import time

from recommender import MovieRecommender

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("movie-rec-api")

app = FastAPI(
    title="Movie Recommendation Service",
    description="Simple hybrid movie recommender (collaborative + content-based) with in-memory demo data.",
    version="1.0.0",
)

recommender = MovieRecommender()


class Recommendation(BaseModel):
    movie_id: int
    title: str
    score: float


class RecommendResponse(BaseModel):
    user_id: int
    recommendations: List[Recommendation]
    processing_time_ms: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/recommend", response_model=RecommendResponse)
def recommend(user_id: int, top_k: int = 5):
    start = time.perf_counter()

    # validate user_id (for demo, we require it to exist in user_ratings)
    if user_id not in recommender.user_ratings:
        raise HTTPException(
            status_code=404,
            detail=f"User {user_id} not found in demo data. Available users: {list(recommender.user_ratings.keys())}",
        )

    recs = recommender.recommend(user_id=user_id, top_k=top_k)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    return RecommendResponse(
        user_id=user_id,
        recommendations=recs,
        processing_time_ms=round(elapsed_ms, 2),
    )
