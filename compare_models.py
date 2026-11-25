"""
Comparaison courte : Régression Logistique contre SVM linéaire.

Les deux modèles sont comparés dans les mêmes conditions :
  - mêmes données (24 avis de textes.csv), même nettoyage, même TF-IDF
  - validation croisée stratifiée à 4 plis, répétée 20 fois (résultat plus stable)
  - test externe : 18 avis (textes de référence des images et des audios)

Lancement : python compare_models.py
Résultat  : docs/comparaison_modeles.csv et docs/comparaison_modeles.png
"""
import matplotlib

matplotlib.use("Agg")  # on enregistre le graphe sans ouvrir de fenêtre
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from sentiment import config
from sentiment.preprocessing import nettoyer_texte
from train import charger_donnees, creer_vectorizer


def main():
    # Mêmes données et même nettoyage que train.py
    textes, test = charger_donnees()
    X, y = textes["texte"].apply(nettoyer_texte), textes["sentiment_ref"]
    X_test, y_test = test["texte"].apply(nettoyer_texte), test["sentiment_ref"]

    # Les deux modèles à comparer (mêmes réglages que train.py)
    modeles = {
        "Régression Logistique": LogisticRegression(C=10, max_iter=1000),
        "SVM linéaire": LinearSVC(),
    }
    cv = RepeatedStratifiedKFold(n_splits=4, n_repeats=20, random_state=42)

    lignes, predictions = [], {}
    for nom, modele in modeles.items():
        # Le TF-IDF est dans le pipeline : il est réappris sur chaque pli (pas de fuite)
        pipeline = make_pipeline(creer_vectorizer(), modele)
        scores = cross_val_score(pipeline, X, y, cv=cv)
        pipeline.fit(X, y)
        predictions[nom] = pipeline.predict(X_test)
        lignes.append({
            "Modèle": nom,
            "Accuracy validation croisée": scores.mean(),
            "Accuracy test externe": accuracy_score(y_test, predictions[nom]),
            "F1 macro test externe": f1_score(y_test, predictions[nom], average="macro"),
            "Bonnes réponses (sur 18)": int((predictions[nom] == y_test).sum()),
            "Probabilités (predict_proba)": "Oui" if hasattr(modele, "predict_proba") else "Non",
        })
    resultats = pd.DataFrame(lignes)
    print(resultats.round(3).T.to_string())
    a, b = predictions.values()
    print("Prédictions identiques sur le test :", int((a == b).sum()), "/ 18")

    dossier = config.RACINE / "docs"
    resultats.round(3).to_csv(dossier / "comparaison_modeles.csv", index=False, encoding="utf-8-sig")

    # Graphe en barres : une couleur par modèle
    mesures = ["Accuracy validation croisée", "Accuracy test externe", "F1 macro test externe"]
    fig, ax = plt.subplots(figsize=(8, 4))
    for k, (couleur, (_, r)) in enumerate(zip(["#0f766e", "#d97706"], resultats.iterrows())):
        barres = ax.bar([i + (k - 0.5) * 0.36 for i in range(3)], [r[m] for m in mesures],
                        width=0.34, color=couleur, label=r["Modèle"])
        ax.bar_label(barres, fmt="%.2f", padding=3)
    ax.axhline(1 / 3, color="gray", linestyle="--", linewidth=1, label="Hasard (1/3)")
    ax.set_xticks(range(3), ["Accuracy\nvalidation croisée", "Accuracy\ntest externe", "F1 macro\ntest externe"])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score (0 à 1)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    plt.tight_layout()
    plt.savefig(dossier / "comparaison_modeles.png", dpi=150)


if __name__ == "__main__":
    main()
