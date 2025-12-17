FROM python:3.12-slim

WORKDIR /app

# Copier d'abord requirements pour profiter du cache Docker
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copier le code + modèles
COPY api ./api
COPY model ./model

EXPOSE 5000

# Lancer l'API via gunicorn (prod-friendly)
CMD ["gunicorn", "-b", "0.0.0.0:5000", "api.app:app"]
