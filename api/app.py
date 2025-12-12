from flask import Flask, request, jsonify, render_template
from model_loader import ModelLoader

app = Flask(__name__)

# Initialisation du chargeur de modèle
model_loader = ModelLoader()

@app.route('/')
def home():
    """Page d'accueil avec formulaire HTML"""
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    """Vérifie l'état de l'API"""
    status = {
        'status': 'healthy',
        'model_loaded': model_loader.loaded,
        'api': 'spam-detection-api',
        'version': '1.0.0'
    }
    return jsonify(status)

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint pour une prédiction unique"""
    try:
        if not model_loader.loaded:
            if not model_loader.load_model():
                return jsonify({'error': 'Modèle non disponible'}), 503
        
        data = request.get_json()

        if not data or 'text' not in data:
            return jsonify({
                'error': 'Le champ "text" est requis',
                'example': {'text': 'Win a free iPhone now!'}
            }), 400
        
        text = data['text']

        if not text.strip():
            return jsonify({'error': 'Le texte ne peut pas être vide'}), 400
        
        result = model_loader.predict(text)

        response = {
            'text_preview': text[:100] + '...' if len(text) > 100 else text,
            'prediction': result['prediction'],
            'label': result['label'],
            'is_spam': result['label'] == 'spam'
        }

        if 'probabilities' in result:
            response['probabilities'] = result['probabilities']
            response['confidence'] = result['confidence']

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Prédiction pour plusieurs textes"""
    try:
        if not model_loader.loaded:
            model_loader.load_model()
        
        data = request.get_json()

        if not data or 'texts' not in data:
            return jsonify({'error': 'Le champ "texts" est requis'}), 400
        
        texts = data['texts']

        if not isinstance(texts, list):
            return jsonify({'error': '"texts" doit être une liste'}), 400
        
        if len(texts) > 100:
            return jsonify({'error': 'Maximum 100 textes par requête'}), 400
        
        results = []
        spam_count = 0

        for text in texts:
            result = model_loader.predict(text)
            results.append({
                'text': text[:50] + '...' if len(text) > 50 else text,
                'prediction': result['prediction'],
                'label': result['label']
            })
            if result['label'] == 'spam':
                spam_count += 1
        
        return jsonify({
            'count': len(results),
            'spam_count': spam_count,
            'ham_count': len(results) - spam_count,
            'spam_ratio': spam_count / len(results),
            'results': results
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Infos sur le modèle"""
    if not model_loader.loaded:
        return jsonify({'error': 'Modèle non chargé'}), 503
    
    return jsonify({
        'model_type': type(model_loader.model).__name__,
        'features': model_loader.vectorizer.get_feature_names_out().shape[0],
        'has_probabilities': hasattr(model_loader.model, 'predict_proba'),
        'has_decision_function': hasattr(model_loader.model, 'decision_function')
    })

if __name__ == '__main__':
    print("Démarrage de l'API de détection de spam...")
    model_loader.load_model()
    app.run(host='0.0.0.0', port=5000, debug=True)
