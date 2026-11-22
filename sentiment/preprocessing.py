"""
Étape 1 (partie nettoyage) : prétraitement minimal du texte.

Le MÊME nettoyage est utilisé :
  - pendant l'entraînement (train.py)
  - pendant la prédiction (API), y compris pour le texte issu de l'OCR / de l'audio.
C'est indispensable : le modèle doit voir des textes préparés de la même façon.
"""
import re
import unicodedata

# Liste de stop-words français (mots très fréquents qui n'apportent pas de sens).
# Elle est écrite SANS accents car on supprime les accents avant de filtrer.
# ATTENTION : on garde volontairement les mots de négation et d'intensité
# ("ne", "pas", "plus", "sans", "aucun", "rien", "tres", "trop", "jamais"...)
# car ils changent complètement le sentiment ("fonctionne" vs "ne fonctionne pas").
STOP_WORDS = {
    "le", "la", "les", "un", "une", "des", "du", "de", "au", "aux",
    "et", "ou", "en", "a", "est", "sont", "ete", "etre", "etait",
    "ce", "cet", "cette", "ces", "se", "sa", "son", "ses",
    "mon", "ma", "mes", "ton", "ta", "tes", "notre", "nos", "votre", "vos",
    "leur", "leurs", "je", "tu", "il", "elle", "on", "nous", "vous", "ils",
    "elles", "me", "te", "lui", "y", "que", "qui", "quoi", "dont",
    "par", "pour", "sur", "dans", "avec", "chez", "ai", "as", "avons",
    "avez", "ont", "suis", "es", "comme", "lors", "ici",
}


def supprimer_accents(texte: str) -> str:
    """Remplace les lettres accentuées par leur version simple (é -> e, à -> a).

    Utile car l'OCR et la transcription audio ne restituent pas toujours
    correctement les accents : on rend ainsi le modèle plus robuste.
    """
    # NFD sépare la lettre de son accent ("é" -> "e" + "´"), puis on retire les accents
    texte_decompose = unicodedata.normalize("NFD", texte)
    return "".join(c for c in texte_decompose if unicodedata.category(c) != "Mn")


def nettoyer_texte(texte: str) -> str:
    """Nettoie un avis client et renvoie une chaîne prête pour la vectorisation.

    Étapes :
      1. minuscules
      2. suppression des accents
      3. suppression de la ponctuation / chiffres / symboles (on garde les lettres)
      4. suppression des stop-words et des lettres isolées ("l'", "d'", "c'")

    Exemple :
        "Article reçu cassé, je suis TRÈS déçu !"  ->  "article recu casse tres decu"
    """
    if not texte:
        return ""

    texte = texte.lower()                         # 1. minuscules
    texte = supprimer_accents(texte)              # 2. accents
    texte = re.sub(r"[^a-z]+", " ", texte)        # 3. tout ce qui n'est pas une lettre -> espace

    mots = texte.split()
    mots = [m for m in mots if m not in STOP_WORDS and len(m) > 1]  # 4. stop-words
    return " ".join(mots)
