"""
Studio Voix — mini logiciel local :
  1. enregistre / importe un échantillon de TA voix
  2. génère une chanson avec ACE-Step 1.5 à partir d'un « preprompt »
     (genre, style, instruments, ambiance) + tes paroles
  3. sépare la voix de l'instrumental (Demucs)
  4. remplace la voix chantée par la tienne (Seed-VC, conversion de voix chantée)
  5. remixe le tout
Il sait aussi générer de la musique seule (voix d'ACE-Step ou instrumental) et lire un texte
avec une voix de la bibliothèque (synthèse vocale Chatterbox).
Un onglet « Modèles » indique où sont stockés les modèles et permet de les télécharger.

Point d'entrée lancé par lancer.bat ; le code est dans le paquet studiovoix/.
Prérequis : voir README.md (serveur ACE-Step lancé + Seed-VC installé).
"""
from studiovoix.interface import build_ui

if __name__ == "__main__":
    build_ui().queue().launch(inbrowser=True)
