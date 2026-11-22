"""
ÉTAPE 5 : Tests automatisés de l'API (unitaires + intégration).

On utilise le TestClient de FastAPI (basé sur httpx) : il appelle l'API
directement en mémoire, sans avoir besoin de lancer le serveur uvicorn.

Lancement (depuis la racine du projet) :
    pytest -v
"""
import pytest
from fastapi.testclient import TestClient

from main import app
from sentiment import config
from sentiment.preprocessing import nettoyer_texte

client = TestClient(app)

# Fichiers réels du dataset utilisés pour les tests d'intégration
IMAGE_POSITIVE = config.DOSSIER_IMAGES / "I04.png"
AUDIO_POSITIF = config.DOSSIER_AUDIOS / "A04.mp3"

CLES_ATTENDUES = {"source", "texte_extrait", "sentiment_predit", "confiance"}


def verifier_structure(donnees: dict, source: str):
    """Vérifie que la réponse JSON a bien le format unifié attendu."""
    assert set(donnees.keys()) == CLES_ATTENDUES
    assert donnees["source"] == source
    assert donnees["sentiment_predit"] in config.CLASSES
    assert 0 <= donnees["confiance"] <= 1
    assert donnees["texte_extrait"].strip() != ""


# ---------------------------------------------------------------------------
# Tests unitaires : nettoyage du texte
# ---------------------------------------------------------------------------
def test_nettoyage_minuscules_accents_stopwords():
    assert nettoyer_texte("Article reçu CASSÉ, je suis déçu !") == "article recu casse decu"


def test_nettoyage_garde_la_negation():
    # "ne" et "pas" ne doivent PAS être supprimés : ils inversent le sentiment
    assert "ne" in nettoyer_texte("Ne fonctionne pas").split()
    assert "pas" in nettoyer_texte("Ne fonctionne pas").split()


# ---------------------------------------------------------------------------
# Routes simples
# ---------------------------------------------------------------------------
def test_health():
    reponse = client.get("/health")
    assert reponse.status_code == 200
    assert reponse.json() == {"status": "ok"}


def test_page_web_de_test():
    reponse = client.get("/")
    assert reponse.status_code == 200
    assert "text/html" in reponse.headers["content-type"]


# ---------------------------------------------------------------------------
# POST /predict/text
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("texte, attendu", [
    ("Excellent produit, livraison rapide, je recommande vivement !", "positif"),
    ("Article reçu cassé, remboursement toujours en attente, très déçu.", "negatif"),
])
def test_predict_text_ok(texte, attendu):
    reponse = client.post("/predict/text", json={"texte": texte})
    assert reponse.status_code == 200
    verifier_structure(reponse.json(), "texte")
    assert reponse.json()["sentiment_predit"] == attendu


@pytest.mark.parametrize("corps", [
    {},                      # champ "texte" manquant
    {"texte": ""},           # texte vide
    {"texte": "   "},        # que des espaces
    {"texte": 123},          # mauvais type
    {"mauvais_champ": "x"},  # mauvais nom de champ
])
def test_predict_text_invalide_422(corps):
    reponse = client.post("/predict/text", json=corps)
    assert reponse.status_code == 422


# ---------------------------------------------------------------------------
# POST /predict/image
# ---------------------------------------------------------------------------
def test_predict_image_ok():
    with open(IMAGE_POSITIVE, "rb") as f:
        reponse = client.post("/predict/image", files={"fichier": ("I04.png", f, "image/png")})
    assert reponse.status_code == 200
    verifier_structure(reponse.json(), "image")


def test_predict_image_sans_fichier_422():
    assert client.post("/predict/image").status_code == 422


def test_predict_image_mauvais_type_422():
    # Un fichier texte envoyé à la place d'une image doit être refusé
    reponse = client.post("/predict/image", files={"fichier": ("note.txt", b"bonjour", "text/plain")})
    assert reponse.status_code == 422


def test_predict_image_trop_grande_422():
    gros_fichier = b"0" * (5 * 1024 * 1024 + 1)  # 5 Mo + 1 octet
    reponse = client.post("/predict/image", files={"fichier": ("grosse.png", gros_fichier, "image/png")})
    assert reponse.status_code == 422


def test_predict_image_corrompue_422():
    reponse = client.post("/predict/image", files={"fichier": ("faux.png", b"pas une image", "image/png")})
    assert reponse.status_code == 422


# ---------------------------------------------------------------------------
# POST /predict/audio
# ---------------------------------------------------------------------------
def test_predict_audio_ok():
    with open(AUDIO_POSITIF, "rb") as f:
        reponse = client.post("/predict/audio", files={"fichier": ("A04.mp3", f, "audio/mpeg")})
    assert reponse.status_code == 200
    verifier_structure(reponse.json(), "audio")
    assert reponse.json()["sentiment_predit"] == "positif"


def test_predict_audio_mauvais_type_422():
    with open(IMAGE_POSITIVE, "rb") as f:
        reponse = client.post("/predict/audio", files={"fichier": ("I04.png", f, "image/png")})
    assert reponse.status_code == 422


# ---------------------------------------------------------------------------
# POST /predict (détection automatique)
# ---------------------------------------------------------------------------
def test_predict_auto_texte():
    reponse = client.post("/predict", data={"texte": "Service client exemplaire, bravo !"})
    assert reponse.status_code == 200
    verifier_structure(reponse.json(), "texte")


def test_predict_auto_image():
    with open(IMAGE_POSITIVE, "rb") as f:
        reponse = client.post("/predict", files={"fichier": ("I04.png", f, "image/png")})
    assert reponse.status_code == 200
    assert reponse.json()["source"] == "image"


def test_predict_auto_audio():
    with open(AUDIO_POSITIF, "rb") as f:
        reponse = client.post("/predict", files={"fichier": ("A04.mp3", f, "audio/mpeg")})
    assert reponse.status_code == 200
    assert reponse.json()["source"] == "audio"


def test_predict_auto_vide_422():
    assert client.post("/predict").status_code == 422


def test_predict_auto_type_inconnu_422():
    reponse = client.post("/predict", files={"fichier": ("doc.pdf", b"%PDF", "application/pdf")})
    assert reponse.status_code == 422
