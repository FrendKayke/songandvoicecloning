"""Interface Gradio : assemble les onglets (un module par onglet dans studiovoix/onglets/).

Chaque module d'onglet a deux fonctions : construire() crée ses composants dans l'onglet ouvert ici et les renvoie
(accès par c.nom), brancher(c, demo, o) déclare ses événements ; o donne accès aux composants des autres onglets
(o.chanson.voix…), pour les listes à tenir à jour d'un onglet à l'autre.
"""
from types import SimpleNamespace

import gradio as gr

from .onglets import (bande_son, bibliotheque, bruitages, chanson, entrainement, espace, galerie, histoire,
                      illustrations, images, modeles, modeles_3d, photos, plans, remix, serie, serie_musique,
                      synthese, videos)
from .onglets.chanson import apercu_description, maj_mode
from .onglets.synthese import synthese_puis_rvc

# apercu_description, maj_mode et synthese_puis_rvc sont réexportés pour les tests
__all__ = ["GROUPES", "ONGLETS", "build_ui", "apercu_description", "maj_mode", "synthese_puis_rvc"]

# Cinq groupes, chacun avec ses onglets : (titre du groupe, [(clé dans o, module, titre de l'onglet)]). Un onglet =
# une chose à faire, nommée par ce qu'on y fait (demande de l'utilisateur : « un onglet avec ce que je peux faire »).
GROUPES = [
    ("🎤 Voix", [
        ("bibliotheque", bibliotheque, "🎙️ Mes voix (enregistrer, nettoyer)"),
        ("synthese", synthese, "🗣️ Lire un texte avec une voix"),
        ("rvc", entrainement, "🧠 Entraîner un modèle de ma voix"),
    ]),
    ("🎵 Musique", [
        ("chanson", chanson, "🎵 Créer une chanson ou une musique"),
        ("remix", remix, "🎛️ Remixer une musique (autre style)"),
        ("serie_musique", serie_musique, "🗂️ Musiques en série (Excel, liste)"),
    ]),
    ("🖼️ Image et vidéo", [
        ("images", images, "🖼️ Une image (texte ou photo)"),
        ("serie", serie, "🗂️ Images en série (Excel, liste)"),
        ("histoire", histoire, "📖 Histoire en images"),
        ("videos", videos, "🎬 Une vidéo"),
        ("plans", plans, "🎞️ Vidéo en plusieurs plans"),
        ("photos", photos, "📷 Retoucher une photo"),
    ]),
    ("🎮 Jeu", [
        ("bande_son", bande_son, "🎼 Musiques de jeu"),
        ("bruitages", bruitages, "🔊 Bruitages"),
        ("illustrations", illustrations, "🃏 Illustrations de cartes"),
        ("modeles_3d", modeles_3d, "🧊 Modèles 3D"),
    ]),
    ("🧰 Outils", [
        ("galerie", galerie, "🗃️ Galerie (toutes mes créations)"),
        ("modeles", modeles, "⚙️ Modèles et diagnostic"),
        ("espace", espace, "💽 Espace disque"),
    ]),
]
ONGLETS = [onglet for _, onglets in GROUPES for onglet in onglets]


def build_ui():
    with gr.Blocks(title="Studio Voix") as demo:
        gr.Markdown("# 🎤 Studio Voix\nClone ta voix, écris tes paroles, choisis le style : la chanson est générée en local.")
        o = SimpleNamespace()
        with gr.Tabs():
            for titre_groupe, onglets in GROUPES:
                with gr.Tab(titre_groupe) as groupe:
                    # Un groupe d'un seul onglet affiche son contenu directement, sans seconde rangée d'onglets
                    with (gr.Tabs() if len(onglets) > 1 else gr.Column()):
                        for cle, module, titre in onglets:
                            with (gr.Tab(titre) if len(onglets) > 1 else gr.Column()) as onglet:
                                composants = module.construire()
                            composants.onglet = onglet if len(onglets) > 1 else groupe
                            composants.groupe = groupe
                            setattr(o, cle, composants)
        for cle, module, _ in ONGLETS:
            module.brancher(getattr(o, cle), demo, o)
    return demo
