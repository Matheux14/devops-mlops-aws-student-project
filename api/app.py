from flask import Flask, request, jsonify
from .model_loader import get_model, get_vectorizer

app = Flask(__name__)

model = get_model()
vectorizer = get_vectorizer()

@app.route("/")
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    """
    JSON attendu:
    {"text": "ton texte ici"}
    """
    data = request.get_json(silent=True) or {}
    text = data.get("text")

    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Missing or empty 'text'"}), 400

    X = vectorizer.transform([text])
    y_pred = model.predict(X)[0]

    response = {"prediction": str(y_pred)}

    # optionnel: proba si dispo
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        response["probabilities"] = [float(p) for p in proba]

    return jsonify(response)
