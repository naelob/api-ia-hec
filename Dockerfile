# Utiliser une image de base officielle Python
FROM python:3.10

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier le fichier requirements.txt dans le conteneur
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Installer libgomp1 pour LightGBM
RUN apt-get update && apt-get install -y libgomp1

# Copier le code source dans le conteneur
COPY . .

# Exposer le port sur lequel FastAPI s'exécutera
EXPOSE 8000

# Commande pour lancer l'application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
