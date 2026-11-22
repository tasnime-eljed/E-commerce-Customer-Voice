"""
Configuration centrale du projet : tous les chemins au même endroit.

On calcule les chemins à partir de l'emplacement de ce fichier, ce qui permet
de lancer les scripts depuis n'importe quel dossier.
"""
from pathlib import Path

# Dossier racine du projet (le dossier parent de "sentiment/")
RACINE = Path(__file__).resolve().parent.parent

# Jeu de données fourni
DOSSIER_DATASET = RACINE / "dataset_sentiment_ecommerce"
FICHIER_TEXTES = DOSSIER_DATASET / "textes.csv"
FICHIER_IMAGES_CSV = DOSSIER_DATASET / "images.csv"
FICHIER_AUDIOS_CSV = DOSSIER_DATASET / "audios.csv"
DOSSIER_IMAGES = DOSSIER_DATASET / "images"
DOSSIER_AUDIOS = DOSSIER_DATASET / "audios"

# Artefacts produits par l'entraînement (train.py)
DOSSIER_MODELES = RACINE / "models"
FICHIER_MODELE = DOSSIER_MODELES / "model.pkl"
FICHIER_VECTORIZER = DOSSIER_MODELES / "vectorizer.pkl"

# Dossier contenant la langue française pour Tesseract (fra.traineddata)
DOSSIER_TESSDATA = RACINE / "tessdata"

# Les 3 classes que le modèle sait prédire (mêmes libellés que dans le CSV)
CLASSES = ["positif", "negatif", "neutre"]
