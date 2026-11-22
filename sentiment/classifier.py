"""
Classifieur de sentiment texte : c'est le "cerveau" commun aux 3 sources.

Il charge les artefacts produits par train.py (model.pkl + vectorizer.pkl)
et prédit le sentiment d'un texte, quelle que soit son origine :
  - texte saisi directement
  - texte extrait d'une image (OCR)
  - texte transcrit d'un audio (ASR)
"""
from functools import lru_cache

import joblib

from sentiment import config
from sentiment.preprocessing import nettoyer_texte


class SentimentClassifier:
    """Enveloppe simple autour du modèle scikit-learn entraîné."""

    def __init__(self):
        if not config.FICHIER_MODELE.exists():
            raise FileNotFoundError(
                "Modèle introuvable. Lancez d'abord l'entraînement : python train.py"
            )
        self.modele = joblib.load(config.FICHIER_MODELE)
        self.vectorizer = joblib.load(config.FICHIER_VECTORIZER)

    def predire(self, texte: str) -> tuple[str, float]:
        """Renvoie (sentiment, confiance) pour un texte brut.

        La confiance est la probabilité de la classe prédite (entre 0 et 1).
        """
        texte_propre = nettoyer_texte(texte)                 # même nettoyage qu'à l'entraînement
        vecteur = self.vectorizer.transform([texte_propre])  # texte -> vecteur TF-IDF
        probabilites = self.modele.predict_proba(vecteur)[0] # une probabilité par classe

        indice = probabilites.argmax()                       # classe la plus probable
        sentiment = self.modele.classes_[indice]
        confiance = float(probabilites[indice])
        return sentiment, confiance


@lru_cache(maxsize=1)
def get_classifier() -> SentimentClassifier:
    """Charge le modèle UNE seule fois (mis en cache) puis le réutilise."""
    return SentimentClassifier()


def analyser_texte(texte: str, source: str = "texte") -> dict:
    """Point de sortie commun de la cascade : texte -> réponse JSON unifiée.

    Toutes les routes de l'API (texte, image, audio) finissent par appeler
    cette fonction, ce qui garantit une réponse au même format.
    """
    sentiment, confiance = get_classifier().predire(texte)
    return {
        "source": source,
        "texte_extrait": texte,
        "sentiment_predit": sentiment,
        "confiance": round(confiance, 4),
    }
