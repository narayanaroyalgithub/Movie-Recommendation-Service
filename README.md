# Movie Recommendation Service

A simple **movie recommendation API** built to demonstrate:

- Collaborative-style recommendations (user–user similarity)
- Content-based recommendations (genre similarity)
- A small hybrid scoring logic
- A clean REST API using FastAPI

This is a learning / demo project, not a production system.

---

## 🎯 What It Does

Given a `user_id`, the service returns a ranked list of movies that user is likely to enjoy.

Under the hood it:

1. Uses a tiny in-memory dataset of:
   - movies with genres
   - users with past ratings
2. Computes:
   - **Collaborative scores**: based on similar users’ ratings
   - **Content-based scores**: based on genre similarity
3. Combines both signals into a final score and returns the **top‑K** movies.

Example use case:
- User 1 has liked several action/sci‑fi movies (e.g. Nolan films).
- The API recommends other action/sci‑fi movies such as *The Matrix* or *Mad Max: Fury Road* with high scores.

---

## 🧠 Recommendation Logic

### Data (in-memory, for demo)

- `movies`: a small dict of movie IDs → {title, genres}
- `user_ratings`: user IDs → {movie_id → rating}

### Content-based

- Build a **movie–genre matrix** (one‑hot for each genre).
- Compute cosine similarity between movies based on their genre vectors.
- For a user, build a “profile” from movies they rated highly and find similar movies by genre.

### Collaborative-style

- Represent each user as a vector over movie IDs (ratings, 0 for unrated).
- Compute cosine similarity between users.
- For a target user:
  - find similar users,
  - aggregate their ratings for movies the target user hasn’t seen,
  - produce a predicted score for those movies.

### Hybrid scoring

- If user has history:
  - Final score ≈ `0.6 * collaborative_score + 0.4 * content_score`
- If user has no history (cold start):
  - Rely more on content-based / genre similarity.

The API then sorts movies by the final score and returns the top‑K.

---

## 📂 Project Structure

```text
movie-recommendation-service/
├── app.py          # FastAPI app – API endpoints
├── recommender.py  # Hybrid recommender logic
└── requirements.txt
```


## ⚡ Quick Setup (Summary)

```bash
# 1. Clone repo
git clone https://github.com/<your-username>/movie-recommendation-service.git
cd movie-recommendation-service

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the API
uvicorn app:app --reload

# 4. Open in browser
# Swagger docs: http://127.0.0.1:8000/docs
# Sample:      http://127.0.0.1:8000/recommend?user_id=1&top_k=3
```
