"""
Simple hybrid movie recommender:
- tiny in-memory "ratings" for collaborative filtering style similarity
- simple content-based similarity using genres
"""

from typing import List, Dict, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class MovieRecommender:
    def __init__(self):
        # --- tiny in-memory dataset for demo purposes ---

        # Movies with IDs and genres
        # In a real system, these come from a DB or CSV
        self.movies: Dict[int, Dict] = {
            1: {"title": "The Dark Knight", "genres": ["Action", "Crime", "Drama"]},
            2: {"title": "Inception", "genres": ["Action", "Sci-Fi", "Thriller"]},
            3: {"title": "Interstellar", "genres": ["Adventure", "Drama", "Sci-Fi"]},
            4: {"title": "The Matrix", "genres": ["Action", "Sci-Fi"]},
            5: {"title": "Mad Max: Fury Road", "genres": ["Action", "Adventure", "Sci-Fi"]},
            6: {"title": "The Social Network", "genres": ["Biography", "Drama"]},
            7: {"title": "The Big Short", "genres": ["Biography", "Comedy", "Drama"]},
            8: {"title": "Toy Story", "genres": ["Animation", "Adventure", "Comedy"]},
        }

        # Fake user–movie ratings matrix
        # user_id -> {movie_id: rating}
        self.user_ratings: Dict[int, Dict[int, float]] = {
            # User 1 likes Nolan/sci-fi/action
            1: {1: 5.0, 2: 4.5, 3: 4.5},
            # User 2 also likes action/sci-fi
            2: {1: 4.5, 2: 5.0, 4: 4.5, 5: 4.0},
            # User 3 likes drama/biography
            3: {6: 4.5, 7: 4.0},
            # User 4 likes family/animation
            4: {8: 5.0},
        }

        # Precompute data structures for content-based similarity
        self._build_genre_matrix()

    # ---------- Content-based part ----------

    def _build_genre_matrix(self):
        # Build a vocabulary of all genres
        genres_set = set()
        for m in self.movies.values():
            genres_set.update(m["genres"])
        self.genre_list: List[str] = sorted(genres_set)

        # Movie-genre matrix: rows=movies, cols=genres (one-hot)
        self.movie_ids: List[int] = sorted(self.movies.keys())
        self.movie_genre_matrix = np.zeros((len(self.movie_ids), len(self.genre_list)), dtype=float)

        for i, mid in enumerate(self.movie_ids):
            genres = self.movies[mid]["genres"]
            for g in genres:
                j = self.genre_list.index(g)
                self.movie_genre_matrix[i, j] = 1.0

        # Precompute cosine similarity between movies (content-based)
        self.content_sim = cosine_similarity(self.movie_genre_matrix)

    def _similar_movies_content(self, movie_id: int, top_k: int = 5) -> List[Tuple[int, float]]:
        if movie_id not in self.movie_ids:
            return []

        idx = self.movie_ids.index(movie_id)
        sims = self.content_sim[idx]
        # sort by similarity (descending), skip itself
        similar_idx = np.argsort(sims)[::-1]
        results = []
        for i in similar_idx:
            if i == idx:
                continue
            mid = self.movie_ids[i]
            results.append((mid, float(sims[i])))
            if len(results) >= top_k:
                break
        return results

    # ---------- Collaborative-like part ----------

    def _get_user_vector(self, user_id: int) -> np.ndarray:
        """
        Build a sparse-like rating vector for the user across all movies.
        Unrated movies are 0.
        """
        vec = np.zeros(len(self.movie_ids), dtype=float)
        ratings = self.user_ratings.get(user_id, {})
        for i, mid in enumerate(self.movie_ids):
            if mid in ratings:
                vec[i] = ratings[mid]
        return vec

    def _user_similarity(self, user_id: int, other_id: int) -> float:
        u = self._get_user_vector(user_id)
        v = self._get_user_vector(other_id)
        if np.linalg.norm(u) == 0 or np.linalg.norm(v) == 0:
            return 0.0
        sim = cosine_similarity(u.reshape(1, -1), v.reshape(1, -1))[0, 0]
        return float(sim)

    def _collaborative_scores(self, user_id: int) -> Dict[int, float]:
        """
        Simple user-based collaborative filtering:
        - find similar users
        - for each movie they rated, accumulate weighted scores
        """
        scores: Dict[int, float] = {}
        weights: Dict[int, float] = {}

        for other_id in self.user_ratings:
            if other_id == user_id:
                continue
            sim = self._user_similarity(user_id, other_id)
            if sim <= 0:
                continue

            for mid, rating in self.user_ratings[other_id].items():
                if mid in self.user_ratings.get(user_id, {}):
                    continue  # skip movies user already rated
                scores[mid] = scores.get(mid, 0.0) + sim * rating
                weights[mid] = weights.get(mid, 0.0) + sim

        # normalize
        for mid in list(scores.keys()):
            if weights[mid] > 0:
                scores[mid] /= weights[mid]

        return scores

    # ---------- Hybrid recommendation ----------

    def recommend(self, user_id: int, top_k: int = 5) -> List[Dict]:
        """
        Hybrid logic:
        - get collaborative scores
        - if user has no history or few scores, fall back more on content-based
        - combine signals and return top_k movies
        """
        user_history = self.user_ratings.get(user_id, {})
        has_history = len(user_history) > 0

        collab_scores = self._collaborative_scores(user_id) if has_history else {}

        # Content-based "profile": average genre vector of movies user liked
        content_scores: Dict[int, float] = {}
        if has_history:
            liked_vectors = []
            for mid, rating in user_history.items():
                if rating >= 4.0 and mid in self.movie_ids:
                    idx = self.movie_ids.index(mid)
                    liked_vectors.append(self.movie_genre_matrix[idx])
            if liked_vectors:
                user_profile = np.mean(liked_vectors, axis=0, keepdims=True)
                sims = cosine_similarity(user_profile, self.movie_genre_matrix)[0]
                for i, mid in enumerate(self.movie_ids):
                    if mid in user_history:
                        continue
                    content_scores[mid] = float(sims[i])

        # Combine scores: simple weighted sum
        final_scores: Dict[int, float] = {}
        for mid in self.movie_ids:
            if mid in user_history:
                continue
            cs = collab_scores.get(mid, 0.0)
            cb = content_scores.get(mid, 0.0)
            if not has_history:
                # cold start: rely mainly on content (or popularity)
                final_scores[mid] = cb
            else:
                # simple hybrid: 0.6 * collab + 0.4 * content
                final_scores[mid] = 0.6 * cs + 0.4 * cb

        # sort and build result
        ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        recs = []
        for mid, score in ranked[:top_k]:
            recs.append(
                {
                    "movie_id": mid,
                    "title": self.movies[mid]["title"],
                    "score": round(score, 4),
                }
            )
        return recs
