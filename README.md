# DevOps–MLOps AWS Student Project — Spam Email Classifier (Flask + Docker + CI + EC2)

Ce projet met en place un mini pipeline **DevOps–MLOps** complet :
- Entraînement d’un modèle de classification **spam vs ham** (NLP) dans un notebook
- Sérialisation **model + vectorizer** (Joblib)
- API **Flask** exposant `/predict` + endpoints de diagnostic
- Conteneurisation **Docker** + exécution via **Gunicorn**
- **CI GitHub Actions** (pytest)
- Déploiement sur **AWS EC2** (Docker)

---

## Démo (endpoints)

- `GET /` : healthcheck
- `GET /metadata` : métadonnées du modèle (si `model/model_metadata.json` présent)
- `POST /predict` : prédiction + probabilités + seuil

Exemple (PowerShell) :

```powershell
$payload = @{ text = "Win a free iPhone now!"; threshold = 0.5 } | ConvertTo-Json -Compress

Invoke-RestMethod -Uri "http://127.0.0.1:5000/predict" `
  -Method Post -ContentType "application/json; charset=utf-8" -Body $payload
```

Exemple (curl Linux) :

```bash
curl -X POST http://127.0.0.1:5000/predict   -H "Content-Type: application/json"   -d '{"text":"Win a free iPhone now!","threshold":0.5}'
```

---

## Dataset & Modèle

- Sujet : **Détection de Spam dans les Emails**
- Dataset : Kaggle — Email Spam Classification Dataset  
  https://www.kaggle.com/datasets/purusinghvi/email-spam-classification-dataset
- Vectorisation : `TfidfVectorizer(stop_words="english", max_features=5000)`
- Modèle : `LogisticRegression` (classification binaire)

---

## Structure du dépôt

```
devops-mlops-aws-student-project/
├── README.md
├── requirements.txt
├── notebooks/
│   └── train_model.ipynb
├── model/
│   ├── model.joblib
│   ├── tfidf_vectorizer.joblib
│   └── model_metadata.json
├── api/
│   ├── app.py
│   ├── model_loader.py
│   └── __init__.py
├── docker/
│   └── Dockerfile
├── tests/
│   └── test_api.py
├── .github/
│   └── workflows/
│       └── ci.yml
└── docs/
    ├── rapport_final.tex
    └── screenshots/
        ├── 01_notebook_train.png
        ├── 02_model_saved.png
        ├── 03_app_flask.png
        ├── dockerfile1.png
        ├── test1_api.png
        └── 08_ec2_running.png
```

> **Note** : vous avez aussi un dossier “Instance EC2 Ubuntu + Images Rapport/Images Rapport/” contenant d’autres captures.  
> Le rapport LaTeX fourni (docs/rapport_final.tex) inclut **toutes** ces images via des chemins qui contiennent des espaces (gérés via `\detokenize{...}`).

---

## Installation (local)

### 1) Créer un environnement virtuel
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
# source .venv/bin/activate
```

### 2) Installer les dépendances
```bash
pip install -r requirements.txt
```

---

## Lancer l’API en local (mode dev)

```bash
python -m api.app
```

Tester :
- http://127.0.0.1:5000/
- http://127.0.0.1:5000/metadata

---

## Tests unitaires

```bash
python -m pytest -q
```

---

## Docker (build & run)

### Build
```bash
docker build -t ml-api -f docker/Dockerfile .
```

### Run
```bash
docker run --rm -p 5000:5000 ml-api
```

---

## CI GitHub Actions

Le workflow `.github/workflows/ci.yml` :
- installe Python
- installe les dépendances
- exécute `pytest`

Quand tout est OK, l’exécution apparaît en **vert** dans l’onglet **Actions**.

---

## Déploiement AWS EC2 (Docker)

### Connexion SSH
Exemple :
```bash
ssh -i "devops-mlops.pem" ubuntu@ec2-XXX.compute-1.amazonaws.com
```

### Déploiement (sur EC2)
```bash
sudo systemctl enable --now docker
sudo usermod -aG docker $USER
# Reconnexion SSH ensuite (logout/login)

git clone https://github.com/Matheux14/devops-mlops-aws-student-project.git
cd devops-mlops-aws-student-project

docker build -t ml-api -f docker/Dockerfile .

docker run -d --name ml-api   -p 5000:5000   --restart unless-stopped   ml-api
```

### Ouvrir le port 5000
Dans le **Security Group** de l’instance EC2, ajouter une règle **Inbound** :
- Type : Custom TCP
- Port : 5000
- Source : *My IP* (recommandé) ou 0.0.0.0/0 (moins sûr)

Test depuis votre PC :
```powershell
Invoke-RestMethod "http://ec2-XXX.compute-1.amazonaws.com:5000/metadata"
```

---

## Mise à jour (EC2)

```bash
cd ~/devops-mlops-aws-student-project
git pull

docker build -t ml-api -f docker/Dockerfile .

docker stop ml-api
docker rm ml-api

docker run -d --name ml-api   -p 5000:5000   --restart unless-stopped   ml-api
```

---

## Dépannage rapide

### PowerShell et `curl`
Dans PowerShell, `curl` peut être un alias de `Invoke-WebRequest`.  
Solutions :
- utiliser `curl.exe`
- ou utiliser `Invoke-RestMethod` (recommandé)

### “Permission denied (publickey)”
- mauvais chemin du `.pem`
- mauvais user (`ubuntu` pour Ubuntu)
- clé non associée à l’instance

### API EC2 inaccessible
- port 5000 non ouvert dans le Security Group
- mauvaise adresse DNS/IP
- container non démarré : `docker ps`, `docker logs -n 50 ml-api`

---

## Auteurs
- KOUADIO Konan
- DIMITRI Mintsa