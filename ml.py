from typing import List, Dict, Any, Tuple
import numpy as np


EMBED_MODEL = "models/text-embedding-004"


def embed_text(client, text: str) -> List[float]:
    """
    Creates an embedding vector for 'text' using Gemini embeddings model.
    Returns list[float].
    """
    # IMPORTANT: normalize input + force string
    clean = (text or "").strip()
    if not clean:
        raise ValueError("embed_text got empty text")

    # Some SDK versions expect contents as a list
    resp = client.models.embed_content(
        model=EMBED_MODEL,
        contents=[clean],
    )

    # --- Most common shape: resp.embeddings[0].values ---
    if hasattr(resp, "embeddings") and resp.embeddings:
        emb0 = resp.embeddings[0]
        if hasattr(emb0, "values") and emb0.values:
            return list(emb0.values)

    # --- Alternate shape: resp.embedding.values ---
    if hasattr(resp, "embedding") and hasattr(resp.embedding, "values"):
        return list(resp.embedding.values)

    # --- Last resort: print/debug-friendly error ---
    raise ValueError(f"Could not extract embedding vector. Response type: {type(resp)}")


def cosine_similarity(a: List[float], b: List[float]) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    denom = (np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def top_k_similar(
        query_embedding: List[float],
        decisions: List[Dict[str, Any]],
        k: int = 3
) -> List[Tuple[float, Dict[str, Any]]]:
    """
    Returns top-k decisions sorted by cosine similarity.
    Each decision must have decision["embedding"].
    """
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for d in decisions:
        emb = d.get("embedding")
        if not emb:
            continue
        score = cosine_similarity(query_embedding, emb)
        scored.append((score, d))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:k]