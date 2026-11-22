"""
ÉTAPE 1 : Entraînement du classifieur de sentiment texte.

Ce script :
  1. charge le jeu de données textuel `textes.csv`
  2. nettoie les textes (minuscules, accents, stop-words)
  3. vectorise les textes avec TF-IDF
  4. entraîne une Régression Logistique (3 classes : positif / negatif / neutre)
  5. évalue le modèle (rapport de classification + matrice de confusion)
  6. sauvegarde les artefacts : models/model.pkl et models/vectorizer.pkl

Lancement :
    python train.py
"""
import joblib
import matplotlib

matplotlib.use("Agg")  # pas de fenêtre graphique : on enregistre juste l'image
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from sentiment import config
from sentiment.preprocessing import nettoyer_texte


# ---------------------------------------------------------------------------
# 1. Chargement des données
# ---------------------------------------------------------------------------
def charger_donnees():
    """Charge textes.csv (entraînement) et un jeu de test externe.

    Le jeu de test externe est composé des textes de référence des images et
    des audios (images.csv / audios.csv) : ce sont des avis que le modèle n'a
    JAMAIS vus pendant l'entraînement -> bonne mesure de la généralisation.
    """
    textes = pd.read_csv(config.FICHIER_TEXTES)

    images = pd.read_csv(config.FICHIER_IMAGES_CSV).rename(columns={"texte_attendu_ocr": "texte"})
    audios = pd.read_csv(config.FICHIER_AUDIOS_CSV).rename(columns={"transcription_reference": "texte"})
    test_externe = pd.concat([images, audios], ignore_index=True)

    print(f"Textes d'entraînement : {len(textes)}")
    print(f"Répartition des classes :\n{textes['sentiment_ref'].value_counts()}\n")
    return textes, test_externe


# ---------------------------------------------------------------------------
# 2. + 3. Nettoyage et vectorisation
# ---------------------------------------------------------------------------
def creer_vectorizer():
    """Crée le vectoriseur TF-IDF.

    Choix technique : TF-IDF sur des n-grammes de CARACTÈRES (3 à 5 lettres)
    plutôt que sur des mots entiers. Avec seulement 24 exemples, beaucoup de
    mots du test n'apparaissent jamais dans l'entraînement. Les morceaux de mots
    permettent de rapprocher "recommande" / "recommanderai" ou "déçu" / "décevant",
    et résistent mieux aux petites erreurs de l'OCR et de la transcription audio.
    """
    return TfidfVectorizer(
        analyzer="char_wb",   # n-grammes de caractères à l'intérieur des mots
        ngram_range=(3, 5),   # morceaux de 3 à 5 caractères
        sublinear_tf=True,    # atténue l'effet des mots répétés
    )


# ---------------------------------------------------------------------------
# 4. + 5. Entraînement et évaluation
# ---------------------------------------------------------------------------
def comparer_modeles(X, y):
    """Compare rapidement Régression Logistique et SVM par validation croisée."""
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    modeles = {
        "Régression Logistique": LogisticRegression(C=10, max_iter=1000),
        "SVM linéaire": LinearSVC(),
    }
    print("=== Comparaison des modèles (validation croisée 4 plis sur textes.csv) ===")
    for nom, modele in modeles.items():
        pipeline = make_pipeline(creer_vectorizer(), modele)
        predictions = cross_val_predict(pipeline, X, y, cv=cv)
        print(f"  {nom:<22} accuracy = {accuracy_score(y, predictions):.2f}")
    print("  -> On garde la Régression Logistique : elle fournit des probabilités,")
    print("     ce qui permet de renvoyer un score de 'confiance' dans l'API.\n")


def evaluer(modele, vectorizer, X_test, y_test, titre, fichier_image):
    """Affiche le rapport de classification et enregistre la matrice de confusion."""
    predictions = modele.predict(vectorizer.transform(X_test))

    print(f"=== Évaluation : {titre} ===")
    print(classification_report(y_test, predictions, labels=config.CLASSES, zero_division=0))

    ConfusionMatrixDisplay.from_predictions(y_test, predictions, labels=config.CLASSES, cmap="Blues")
    plt.title(f"Matrice de confusion - {titre}")
    plt.tight_layout()
    plt.savefig(config.DOSSIER_MODELES / fichier_image)
    plt.close()
    print(f"Matrice de confusion enregistrée dans models/{fichier_image}\n")


def main():
    config.DOSSIER_MODELES.mkdir(exist_ok=True)

    # 1. Chargement
    textes, test_externe = charger_donnees()

    # 2. Nettoyage (la même fonction sera utilisée par l'API)
    X_train = textes["texte"].apply(nettoyer_texte)
    y_train = textes["sentiment_ref"]
    X_test = test_externe["texte"].apply(nettoyer_texte)
    y_test = test_externe["sentiment_ref"]
    print("Exemple de nettoyage :")
    print(f"  avant : {textes['texte'][1]}")
    print(f"  après : {X_train[1]}\n")

    # Comparaison LogReg / SVM (petit jeu de données -> résultats indicatifs)
    comparer_modeles(X_train, y_train)

    # 3. Vectorisation TF-IDF (apprise UNIQUEMENT sur les données d'entraînement)
    vectorizer = creer_vectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)

    # 4. Entraînement du modèle final sur tout textes.csv
    modele = LogisticRegression(C=10, max_iter=1000)
    modele.fit(X_train_vec, y_train)

    # 5. Évaluation sur les 18 avis jamais vus (textes de référence images + audios)
    evaluer(modele, vectorizer, X_test, y_test,
            titre="test externe (18 avis jamais vus)",
            fichier_image="confusion_matrix.png")

    # 6. Sauvegarde des artefacts
    joblib.dump(modele, config.FICHIER_MODELE)
    joblib.dump(vectorizer, config.FICHIER_VECTORIZER)
    print(f"Modèle sauvegardé     : {config.FICHIER_MODELE}")
    print(f"Vectoriseur sauvegardé : {config.FICHIER_VECTORIZER}")


if __name__ == "__main__":
    main()
