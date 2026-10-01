"""Interface Gradio : assemble les onglets (un module par onglet dans studiovoix/onglets/).

Chaque module d'onglet a deux fonctions : construire() crée ses composants dans l'onglet ouvert ici et les renvoie
(accès par c.nom), brancher(c, demo, o) déclare ses événements ; o donne accès aux composants des autres onglets
(o.chanson.voix…), pour les listes à tenir à jour d'un onglet à l'autre.
"""
from types import SimpleNamespace

import gradio as gr

from .onglets import (bande_son, bibliotheque, bruitages, chanson, entrainement, galerie, illustrations, modeles,
                      modeles_3d, synthese)
from .onglets.chanson import apercu_description, maj_mode
from .onglets.synthese import synthese_puis_rvc

# apercu_description, maj_mode et synthese_puis_rvc sont réexportés pour les tests
__all__ = ["ONGLETS", "build_ui", "apercu_description", "maj_mode", "synthese_puis_rvc"]

# (clé dans o, module, titre de l'onglet), dans l'ordre d'affichage
ONGLETS = [
    ("bibliotheque", bibliotheque, "1. Bibliothèque de voix"),
    ("chanson", chanson, "2. Créer une chanson"),
    ("bande_son", bande_son, "3. Bande-son de jeu"),
    ("synthese", synthese, "4. Synthèse vocale"),
    ("galerie", galerie, "5. Galerie"),
    ("rvc", entrainement, "6. Entraîner ma voix (RVC)"),
    ("bruitages", bruitages, "7. Bruitages"),
    ("modeles_3d", modeles_3d, "8. Modèles 3D"),
    ("illustrations", illustrations, "9. Illustrations de cartes"),
    ("modeles", modeles, "10. Modèles"),
]


def build_ui():
    with gr.Blocks(title="Studio Voix") as demo:
        gr.Markdown("# 🎤 Studio Voix\nClone ta voix, écris tes paroles, choisis le style : la chanson est générée en local.")
        o = SimpleNamespace()
        for cle, module, titre in ONGLETS:
            with gr.Tab(titre) as onglet:
                composants = module.construire()
            composants.onglet = onglet
            setattr(o, cle, composants)
        for cle, module, _ in ONGLETS:
            module.brancher(getattr(o, cle), demo, o)
    return demo
