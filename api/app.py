from flask import Flask, request, jsonify
from .model_loader import get_model, get_vectorizer, get_metadata

app = Flask(__name__)

# Chargement au démarrage (plus simple et plus rapide ensuite)
model = get_model()
vectorizer = get_vectorizer()
metadata = get_metadata()  # dict ou None

LABELS = {0: "ham", 1: "spam"}


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/metadata", methods=["GET"])
def metadata_route():
    if not metadata:
        return jsonify({"error": "metadata not available"}), 404
    return jsonify(metadata), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    JSON attendu:
    {"text": "ton texte ici", "threshold": 0.5 (optionnel)}

    Réponse:
    - prediction: 0/1
    - label: ham/spam
    - spam_probability: proba classe 1 (si dispo)
    - threshold: seuil utilisé (si fourni)
    - probabilities: [p(ham), p(spam)] (si dispo)
    """
    data = request.get_json(silent=True) or {}

    text = data.get("text", "")
    threshold = data.get("threshold", None)

    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Missing or empty 'text'"}), 400

    # seuil optionnel (si fourni)
    if threshold is not None:
        try:
            threshold = float(threshold)
        except (TypeError, ValueError):
            return jsonify({"error": "Invalid 'threshold' (must be a number)"}), 400
        if not (0.0 <= threshold <= 1.0):
            return jsonify({"error": "Invalid 'threshold' (must be between 0 and 1)"}), 400

    X = vectorizer.transform([text])

    # base prediction (0/1)
    y_hat = int(model.predict(X)[0])

    response = {
        "prediction": y_hat,
        "label": LABELS.get(y_hat, str(y_hat)),
    }

    # Probabilités si dispo
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        p_ham = float(proba[0])
        p_spam = float(proba[1])

        response["probabilities"] = [p_ham, p_spam]
        response["spam_probability"] = p_spam

        # Si threshold fourni, on décide avec p_spam plutôt que predict()
        if threshold is not None:
            y_thr = 1 if p_spam >= threshold else 0
            response["prediction"] = int(y_thr)
            response["label"] = LABELS.get(int(y_thr), str(y_thr))
            response["threshold"] = float(threshold)

    else:
        # Si pas de predict_proba, on renvoie quand même threshold si fourni
        if threshold is not None:
            response["threshold"] = float(threshold)

    return jsonify(response), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
