"""
Schémas Pydantic : ils décrivent et VALIDENT les données qui entrent et sortent de l'API.

Si une donnée ne respecte pas un schéma, Pydantic lève une erreur et FastAPI
renvoie automatiquement une réponse 422 (Unprocessable Entity).
"""
from typing import Literal

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Règles de validation des fichiers
# ---------------------------------------------------------------------------
TAILLE_MAX_IMAGE = 5 * 1024 * 1024     # 5 Mo
TAILLE_MAX_AUDIO = 10 * 1024 * 1024    # 10 Mo

TYPES_IMAGE = {"image/png", "image/jpeg", "image/jpg", "image/bmp", "image/tiff"}
TYPES_AUDIO = {"audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave",
               "audio/mpeg", "audio/mp3"}


# ---------------------------------------------------------------------------
# Entrée de la route /predict/text
# ---------------------------------------------------------------------------
class TexteRequete(BaseModel):
    """Corps JSON attendu par POST /predict/text, ex : {"texte": "Super produit !"}"""

    texte: str = Field(..., min_length=3, max_length=5000,
                       examples=["Produit conforme, livraison rapide, je recommande !"])

    @field_validator("texte")
    @classmethod
    def pas_vide(cls, valeur: str) -> str:
        # Refuse un texte composé uniquement d'espaces
        if not valeur.strip():
            raise ValueError("le texte ne peut pas être vide")
        return valeur.strip()


# ---------------------------------------------------------------------------
# Validation des fichiers envoyés (image / audio)
# ---------------------------------------------------------------------------
class FichierImage(BaseModel):
    """Vérifie le type MIME et la taille d'une image reçue."""

    nom: str
    type_mime: str
    taille: int = Field(..., gt=0, le=TAILLE_MAX_IMAGE)  # > 0 octet et <= 5 Mo

    @field_validator("type_mime")
    @classmethod
    def type_autorise(cls, valeur: str) -> str:
        if valeur not in TYPES_IMAGE:
            raise ValueError(f"type d'image non supporté ({valeur}). Autorisés : {sorted(TYPES_IMAGE)}")
        return valeur


class FichierAudio(BaseModel):
    """Vérifie le type MIME et la taille d'un fichier audio reçu."""

    nom: str
    type_mime: str
    taille: int = Field(..., gt=0, le=TAILLE_MAX_AUDIO)  # > 0 octet et <= 10 Mo

    @field_validator("type_mime")
    @classmethod
    def type_autorise(cls, valeur: str) -> str:
        if valeur not in TYPES_AUDIO:
            raise ValueError(f"type audio non supporté ({valeur}). Autorisés : {sorted(TYPES_AUDIO)}")
        return valeur


# ---------------------------------------------------------------------------
# Réponse unifiée (identique pour toutes les sources)
# ---------------------------------------------------------------------------
class PredictionReponse(BaseModel):
    """Format de réponse commun à toutes les routes de prédiction."""

    source: Literal["texte", "image", "audio"]
    texte_extrait: str
    sentiment_predit: Literal["positif", "negatif", "neutre"]
    confiance: float = Field(..., ge=0, le=1)
