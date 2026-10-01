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
Le serveur ACE-Step est démarré ici, en arrière-plan (il charge ses modèles pendant que l'interface s'ouvre), et
arrêté avec l'application ; il est aussi arrêté pendant les moteurs gourmands en mémoire graphique.
"""
import threading

from studiovoix import serveur_acestep
from studiovoix.interface import build_ui


def _demarrer_acestep():
    try:
        serveur_acestep.demarrer()
    except Exception as e:  # noqa: BLE001 - l'interface s'ouvre quand même ; l'erreur réapparaîtra à la génération
        print(f"Serveur ACE-Step non démarré : {e}")


if __name__ == "__main__":
    threading.Thread(target=_demarrer_acestep, daemon=True).start()
    build_ui().queue().launch(inbrowser=True)
