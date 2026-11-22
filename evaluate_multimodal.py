"""
ÉTAPE 2 (vérification) : évalue les pipelines Image (OCR) et Audio (ASR)
sur les fichiers du dossier dataset_sentiment_ecommerce/.

Pour chaque fichier on affiche :
  - la similarité entre le texte extrait et le texte de référence (0 à 100 %)
  - le sentiment prédit et le sentiment attendu

Lancement :
    python evaluate_multimodal.py
"""
from difflib import SequenceMatcher

import pandas as pd

from sentiment import config
from sentiment.asr import analyser_audio
from sentiment.ocr import analyser_image
from sentiment.preprocessing import nettoyer_texte


def similarite(texte_a: str, texte_b: str) -> float:
    """Pourcentage de ressemblance entre deux textes (après nettoyage)."""
    return SequenceMatcher(None, nettoyer_texte(texte_a), nettoyer_texte(texte_b)).ratio() * 100


def trouver_fichier(dossier, nom):
    """Retrouve un fichier même si l'extension diffère.

    Exemple : audios.csv mentionne "A01.wav" alors que le fichier réel est "A01.mp3".
    """
    chemin = dossier / nom
    if chemin.exists():
        return chemin
    candidats = list(dossier.glob(chemin.stem + ".*"))
    return candidats[0] if candidats else None


def evaluer(source, fichier_csv, dossier, colonne_reference, fonction_analyse):
    """Lance la cascade sur chaque fichier listé dans le CSV et affiche les résultats."""
    df = pd.read_csv(fichier_csv)
    lignes = []
    for _, ligne in df.iterrows():
        chemin = trouver_fichier(dossier, ligne["fichier"])
        resultat = fonction_analyse(chemin)
        lignes.append({
            "fichier": chemin.name,
            "similarite_%": round(similarite(resultat["texte_extrait"], ligne[colonne_reference]), 1),
            "attendu": ligne["sentiment_ref"],
            "predit": resultat["sentiment_predit"],
            "confiance": resultat["confiance"],
            "texte_extrait": resultat["texte_extrait"],
        })

    resultats = pd.DataFrame(lignes)
    exactitude = (resultats["attendu"] == resultats["predit"]).mean() * 100
    print(f"\n===== {source} =====")
    print(resultats.drop(columns="texte_extrait").to_string(index=False))
    print(f"-> Similarité moyenne du texte extrait : {resultats['similarite_%'].mean():.1f} %")
    print(f"-> Sentiments correctement prédits     : {exactitude:.1f} %")
    return resultats


def main():
    images = evaluer("IMAGES (OCR pytesseract)", config.FICHIER_IMAGES_CSV,
                     config.DOSSIER_IMAGES, "texte_attendu_ocr", analyser_image)
    audios = evaluer("AUDIOS (ASR faster-whisper)", config.FICHIER_AUDIOS_CSV,
                     config.DOSSIER_AUDIOS, "transcription_reference", analyser_audio)

    # Sauvegarde des résultats détaillés (utile pour le rapport / la démo)
    sortie = config.RACINE / "resultats_multimodal.csv"
    pd.concat([images.assign(source="image"), audios.assign(source="audio")]).to_csv(
        sortie, index=False, encoding="utf-8-sig")
    print(f"\nRésultats détaillés enregistrés dans {sortie.name}")


if __name__ == "__main__":
    main()
