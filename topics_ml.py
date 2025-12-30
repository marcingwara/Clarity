from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np

TOPIC_SEEDS: Dict[str, List[str]] = {
    "career": ["zmiana pracy", "awans", "rekrutacja", "rozmowa kwalifikacyjna", "CV", "firma", "szef"],
    "money": ["kredyt", "budżet domowy", "oszczędzanie", "inwestowanie", "wynagrodzenie", "pensja", "długi"],
    "relationships": ["związek", "relacja", "partner", "małżeństwo", "rodzina", "przyjaźń", "rozstanie"],
    "health": ["zdrowie", "trening", "dieta", "sen", "lekarz", "terapia", "psycholog", "stres"],
    "education": ["kurs", "studia", "nauka", "bootcamp", "python", "ai", "ml", "certyfikat"],
    "life": ["wyjazd", "wakacje", "przeprowadzka", "mieszkanie", "dom", "samochód", "kupno auta"],
}

DEFAULT_TOPIC = "other"


def cosine(a: List[float], b: List[float]) -> float:
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def build_topic_centroids(client, embed_fn, seeds: Dict[str, List[str]] = TOPIC_SEEDS) -> Dict[str, List[float]]:
    """
    Buduje centroidy: wymaga API (embed_fn), więc wywołuj TYLKO po kliknięciu przycisku.
    """
    centroids: Dict[str, List[float]] = {}
    for topic, phrases in seeds.items():
        vecs = [embed_fn(client, p) for p in phrases]
        m = np.mean(np.asarray(vecs, dtype=np.float32), axis=0)
        centroids[topic] = m.tolist()
    return centroids


def classify_from_embedding(
        emb: Optional[List[float]],
        centroids: Optional[Dict[str, List[float]]],
        min_score: float = 0.35,
) -> Tuple[str, float]:
    """
    Zero API: działa tylko na embeddingach już zapisanych w decisions.jsonl
    Zwraca: (topic, best_score)
    """
    if not emb or not centroids:
        return DEFAULT_TOPIC, 0.0

    best_topic = DEFAULT_TOPIC
    best_score = -1.0
    for t, c in centroids.items():
        s = cosine(emb, c)
        if s > best_score:
            best_score = s
            best_topic = t

    if best_score < min_score:
        return DEFAULT_TOPIC, best_score
    return best_topic, best_score