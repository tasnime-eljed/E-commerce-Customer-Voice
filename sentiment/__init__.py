"""
Package `sentiment` : toute la logique "métier" du projet.

    preprocessing.py -> nettoyage du texte (minuscules, accents, stop-words)
    classifier.py    -> chargement du modèle entraîné + prédiction sur un texte
    ocr.py           -> Image  -> texte (pytesseract)  -> classifieur
    asr.py           -> Audio  -> texte (faster-whisper) -> classifieur

Architecture en cascade : l'image et l'audio sont d'abord convertis en texte,
puis c'est TOUJOURS le même classifieur texte qui prend la décision finale.
"""
