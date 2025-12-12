import joblib
import os

class ModelLoader:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.loaded = False
    
    def load_model(self):
        try:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, 'model', 'model.joblib')
            vectorizer_path = os.path.join(base_dir, 'model', 'tfidf_vectorizer.joblib')

            print(f"Chargement depuis: {model_path}")

            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)

            self.loaded = True
            print("Modèle et vectorizer chargés avec joblib")
            return True
        
        except Exception as e:
            print(f"Erreur: {e}")
            import traceback
            traceback.print_exc()
            return False

    def predict(self, text):
        if not self.loaded:
            raise Exception("Modèle non chargé")

        vector = self.vectorizer.transform([text])
        prediction = self.model.predict(vector)[0]

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(vector)[0]
            return {
                'prediction': int(prediction),
                'label': 'spam' if prediction == 1 else 'ham',
                'probabilities': {
                    'ham': float(probs[0]),
                    'spam': float(probs[1])
                },
                'confidence': float(max(probs))
            }

        return {
            'prediction': int(prediction),
            'label': 'spam' if prediction == 1 else 'ham'
        }
