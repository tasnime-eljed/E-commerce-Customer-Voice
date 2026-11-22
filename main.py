"""
ÉTAPE 3 : API REST avec FastAPI.

Routes :
    GET  /                -> mini-application web de test (static/index.html)
    GET  /health          -> vérifie que l'API fonctionne
    POST /predict/text    -> JSON {"texte": "..."}
    POST /predict/image   -> fichier image (multipart/form-data)
    POST /predict/audio   -> fichier audio (multipart/form-data)
    POST /predict         -> détection automatique du type (texte OU fichier)

Documentation interactive générée automatiquement : /docs (Swagger) et /redoc

Lancement :
    uvicorn main:app --reload
"""
import mimetypes
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import ValidationError

from schemas import FichierAudio, FichierImage, PredictionReponse, TexteRequete
from sentiment.asr import analyser_audio
from sentiment.classifier import analyser_texte
from sentiment.ocr import analyser_image

app = FastAPI(
    title="E-commerce Customer Voice - Classifieur de sentiment multi-source",
    description=(
        "Prédit le sentiment (positif / negatif / neutre) d'un avis client "
        "reçu sous forme de **texte**, d'**image** (OCR) ou d'**audio** (transcription). "
        "Architecture en cascade : image et audio sont convertis en texte, "
        "puis un classifieur texte unique prend la décision."
    ),
    version="1.0.0",
)

DOSSIER_STATIC = Path(__file__).parent / "static"


# ---------------------------------------------------------------------------
# Fonctions utilitaires
# ---------------------------------------------------------------------------
def type_mime(fichier: UploadFile) -> str:
    """Renvoie le type MIME du fichier.

    Si le client n'envoie pas de type précis (ex : "application/octet-stream"),
    on le devine à partir de l'extension du nom de fichier (.png, .wav, .mp3...).
    """
    if fichier.content_type and fichier.content_type != "application/octet-stream":
        return fichier.content_type
    devine, _ = mimetypes.guess_type(fichier.filename or "")
    return devine or "application/octet-stream"


def valider_fichier(schema, fichier: UploadFile, contenu: bytes):
    """Valide un fichier avec un schéma Pydantic ; renvoie une erreur 422 si invalide."""
    try:
        schema(nom=fichier.filename or "", type_mime=type_mime(fichier), taille=len(contenu))
    except ValidationError as erreur:
        # On renvoie seulement les messages lisibles (pas les objets internes)
        messages = [f"{'.'.join(map(str, e['loc']))} : {e['msg']}" for e in erreur.errors()]
        raise HTTPException(status_code=422, detail=messages)


def verifier_texte_extrait(resultat: dict) -> dict:
    """Si l'OCR / la transcription n'a trouvé aucun texte, on renvoie une erreur 422."""
    if not resultat["texte_extrait"].strip():
        raise HTTPException(status_code=422, detail="Aucun texte n'a pu être extrait du fichier.")
    return resultat


def traiter_image(fichier: UploadFile) -> dict:
    """Lit, valide puis analyse une image (OCR -> classifieur)."""
    contenu = fichier.file.read()
    valider_fichier(FichierImage, fichier, contenu)
    try:
        resultat = analyser_image(contenu)
    except Exception:  # fichier corrompu, faux PNG, etc.
        raise HTTPException(status_code=422, detail="Image illisible ou corrompue.")
    return verifier_texte_extrait(resultat)


def traiter_audio(fichier: UploadFile) -> dict:
    """Lit, valide puis analyse un fichier audio (transcription -> classifieur)."""
    contenu = fichier.file.read()
    valider_fichier(FichierAudio, fichier, contenu)
    try:
        resultat = analyser_audio(contenu)
    except Exception:  # fichier corrompu, mauvais format, etc.
        raise HTTPException(status_code=422, detail="Fichier audio illisible ou corrompu.")
    return verifier_texte_extrait(resultat)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def accueil():
    """Affiche la mini-application web de test."""
    return FileResponse(DOSSIER_STATIC / "index.html")


@app.get("/health", tags=["Système"])
def health():
    """Permet de vérifier rapidement que l'API est en ligne."""
    return {"status": "ok"}


@app.post("/predict/text", response_model=PredictionReponse, tags=["Prédiction"])
def predict_text(requete: TexteRequete):
    """Prédit le sentiment d'un texte envoyé en JSON : `{"texte": "..."}`."""
    return analyser_texte(requete.texte, source="texte")


@app.post("/predict/image", response_model=PredictionReponse, tags=["Prédiction"])
def predict_image(fichier: UploadFile = File(..., description="Image PNG / JPEG (5 Mo max)")):
    """Extrait le texte d'une image avec l'OCR (pytesseract) puis prédit le sentiment."""
    return traiter_image(fichier)


@app.post("/predict/audio", response_model=PredictionReponse, tags=["Prédiction"])
def predict_audio(fichier: UploadFile = File(..., description="Audio WAV / MP3 (10 Mo max)")):
    """Transcrit un fichier audio (faster-whisper) puis prédit le sentiment."""
    return traiter_audio(fichier)


@app.post("/predict", response_model=PredictionReponse, tags=["Prédiction"])
def predict_auto(
    texte: str | None = Form(None, description="Texte de l'avis (si pas de fichier)"),
    fichier: UploadFile | None = File(None, description="Image ou audio"),
):
    """Détection automatique du type d'entrée.

    - si un fichier est envoyé : son type MIME décide (image/... ou audio/...)
    - sinon, si un texte est envoyé : on utilise le classifieur texte
    """
    if fichier is not None and fichier.filename:
        mime = type_mime(fichier)
        if mime.startswith("image/"):
            return traiter_image(fichier)
        if mime.startswith("audio/"):
            return traiter_audio(fichier)
        raise HTTPException(status_code=422, detail=f"Type de fichier non reconnu : {mime}")

    if texte is not None:
        try:
            requete = TexteRequete(texte=texte)  # même validation que /predict/text
        except ValidationError:
            raise HTTPException(status_code=422, detail="Texte invalide (3 à 5000 caractères).")
        return analyser_texte(requete.texte, source="texte")

    raise HTTPException(status_code=422, detail="Envoyez un champ 'texte' ou un 'fichier'.")
