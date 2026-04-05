"""Shared app state (avoids circular imports between api.main and route modules)."""

_state = {
    "biencoder": None,
    "colbert": None,
    "user_model": None,
    "bm25": None,
    "docs": None,
    "embeddings": None,
}


def get_state():
    return _state
