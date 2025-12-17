import os
import json
import joblib


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "model", "model.joblib"))
VECTORIZER_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "model", "tfidf_vectorizer.joblib"))
METADATA_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "model", "model_metadata.json"))

_model = None
_vectorizer = None
_metadata = None


def get_model():
    """
    Charge le modèle joblib une seule fois (cache en mémoire).
    """
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
    return _model


def get_vectorizer():
    """
    Charge le TF-IDF vectorizer joblib une seule fois (cache en mémoire).
    """
    global _vectorizer
    if _vectorizer is None:
        if not os.path.exists(VECTORIZER_PATH):
            raise FileNotFoundError(f"Vectorizer file not found: {VECTORIZER_PATH}")
        _vectorizer = joblib.load(VECTORIZER_PATH)
    return _vectorizer


def get_metadata():
    """
    Charge les métadonnées JSON si disponibles.
    - gère UTF-8 BOM via utf-8-sig
    - si absent ou invalide, renvoie None sans faire planter l'API
    """
    global _metadata
    if _metadata is not None:
        return _metadata

    if not os.path.exists(METADATA_PATH):
        _metadata = None
        return _metadata

    try:
        # utf-8-sig => enlève automatiquement le BOM si présent
        with open(METADATA_PATH, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        _metadata = data if isinstance(data, dict) else None
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"[WARN] Could not parse metadata JSON at {METADATA_PATH}: {e}")
        _metadata = None

    return _metadata
