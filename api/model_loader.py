import os
import joblib

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "model.joblib")
VECTORIZER_PATH = os.path.join(BASE_DIR, "..", "model", "tfidf_vectorizer.joblib")

_model = None
_vectorizer = None

def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model

def get_vectorizer():
    global _vectorizer
    if _vectorizer is None:
        _vectorizer = joblib.load(VECTORIZER_PATH)
    return _vectorizer
