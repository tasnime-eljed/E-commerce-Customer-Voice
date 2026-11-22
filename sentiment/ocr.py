"""
ÉTAPE 2 (module Image) : Image -> texte (OCR avec pytesseract) -> classifieur.

pytesseract est une simple "passerelle" Python vers le logiciel Tesseract,
qui doit être installé sur la machine (voir README.md).
"""
import io
import os
import shutil
from pathlib import Path

import pytesseract
from PIL import Image, ImageOps

from sentiment import config
from sentiment.classifier import analyser_texte

# --- Localisation du programme Tesseract -----------------------------------
# 1) variable d'environnement TESSERACT_CMD si elle est définie
# 2) sinon "tesseract" s'il est dans le PATH
# 3) sinon le chemin d'installation par défaut sous Windows
CHEMIN_WINDOWS = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = (
    os.getenv("TESSERACT_CMD") or shutil.which("tesseract") or CHEMIN_WINDOWS
)


def _choisir_langue() -> str:
    """Choisit la langue : français si fra.traineddata est présent, sinon anglais."""
    if (config.DOSSIER_TESSDATA / "fra.traineddata").exists():
        # TESSDATA_PREFIX indique à Tesseract où trouver le fichier de langue français
        # (plus fiable que l'option --tessdata-dir quand le chemin contient des espaces)
        os.environ["TESSDATA_PREFIX"] = str(config.DOSSIER_TESSDATA)
        return "fra"
    return "eng"


def extraire_texte_image(image: str | Path | bytes | Image.Image) -> str:
    """Extrait le texte brut d'une image (chemin, octets ou objet PIL)."""
    # Ouverture de l'image avec Pillow selon le type reçu
    if isinstance(image, bytes):
        image = Image.open(io.BytesIO(image))
    elif isinstance(image, (str, Path)):
        image = Image.open(image)

    # Petit prétraitement : passage en niveaux de gris (améliore souvent l'OCR)
    image = ImageOps.grayscale(image)

    texte = pytesseract.image_to_string(image, lang=_choisir_langue())

    # On remplace les retours à la ligne par des espaces et on enlève les espaces en trop
    return " ".join(texte.split())


def analyser_image(image: str | Path | bytes | Image.Image) -> dict:
    """Cascade complète : image -> OCR -> classifieur texte -> réponse unifiée."""
    texte = extraire_texte_image(image)
    return analyser_texte(texte, source="image")


if __name__ == "__main__":
    # Test rapide : python -m sentiment.ocr
    for fichier in sorted(config.DOSSIER_IMAGES.glob("*.png")):
        print(fichier.name, "->", analyser_image(fichier))
