"""
ÉTAPE 2 (module Audio) : Audio -> texte (transcription ASR) -> classifieur.

On utilise `faster-whisper`, une version allégée et plus rapide du modèle
Whisper d'OpenAI. Avantages :
  - fonctionne sur CPU (pas besoin de carte graphique)
  - lit directement les .wav et .mp3 (pas besoin d'installer ffmpeg)
Le modèle (~145 Mo pour "base") est téléchargé automatiquement au 1er lancement.
"""
import io
import os
from functools import lru_cache
from pathlib import Path

from faster_whisper import WhisperModel

from sentiment import config
from sentiment.classifier import analyser_texte

# Taille du modèle Whisper : "tiny" (rapide) < "base" < "small" (plus précis)
TAILLE_MODELE = os.getenv("WHISPER_MODEL", "base")


@lru_cache(maxsize=1)
def charger_modele_whisper() -> WhisperModel:
    """Charge le modèle Whisper une seule fois (opération lente) puis le garde en mémoire."""
    # int8 = calculs compressés : plus rapide et moins de mémoire sur CPU
    return WhisperModel(TAILLE_MODELE, device="cpu", compute_type="int8")


def transcrire_audio(audio: str | Path | bytes) -> str:
    """Transcrit un fichier audio (chemin ou octets) en texte français."""
    if isinstance(audio, bytes):
        audio = io.BytesIO(audio)      # faster-whisper accepte un "fichier en mémoire"
    elif isinstance(audio, Path):
        audio = str(audio)

    modele = charger_modele_whisper()
    # language="fr" : on force le français (évite une mauvaise détection de langue)
    segments, _infos = modele.transcribe(audio, language="fr", beam_size=5)

    # Le modèle renvoie le texte découpé en segments : on les recolle
    return " ".join(segment.text.strip() for segment in segments).strip()


def analyser_audio(audio: str | Path | bytes) -> dict:
    """Cascade complète : audio -> transcription -> classifieur texte -> réponse unifiée."""
    texte = transcrire_audio(audio)
    return analyser_texte(texte, source="audio")


if __name__ == "__main__":
    # Test rapide : python -m sentiment.asr
    for fichier in sorted(config.DOSSIER_AUDIOS.glob("*.*")):
        print(fichier.name, "->", analyser_audio(fichier))
