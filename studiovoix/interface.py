"""Interface Gradio : assemble les onglets (un module par onglet dans studiovoix/onglets/).

Chaque module d'onglet a deux fonctions : construire() crée ses composants dans l'onglet ouvert ici et les renvoie
(accès par c.nom), brancher(c, demo, o) déclare ses événements ; o donne accès aux composants des autres onglets
(o.chanson.voix…), pour les listes à tenir à jour d'un onglet à l'autre.
"""
from types import SimpleNamespace

import gradio as gr

from .onglets import (alleger_3d, bande_son, bibliotheque, bruitages, chanson, composer, conv_couper, conv_format,
                      conv_montage, conv_recoller, conv_reduire, conv_son, costumes, diagnostic, doublage, entrainement,
                      espace, fluidite, fond_video, galerie, hd2d, histoire, illustrations, images, karaoke, kit3d,
                      lot_3d, mini_videos, modeles, modeles_3d, nettoyer, pack, parlante, passage, pdf, pdf_compresser,
                      pdf_filigrane, pdf_fusion, pdf_images, pdf_numeroter, pdf_pages, pdf_pivoter, pdf_proteger,
                      pdf_signer, pdf_texte, personnages_cartes,
                      photos, photos_lot, pixel_animer, pixel_convertir, pixel_personnages, dresseur, plans, remix,
                      revision_3d, rigging_3d, serie, serie_musique, sous_titres, sprites_lora, synthese, tileset3d,
                      videos, voix_off)
from .onglets.chanson import apercu_description, maj_mode
from .onglets.synthese import synthese_puis_rvc

# apercu_description, maj_mode et synthese_puis_rvc sont réexportés pour les tests
__all__ = ["GROUPES", "ONGLETS", "build_ui", "apercu_description", "maj_mode", "synthese_puis_rvc"]

# Onze groupes (huit onglets au plus chacun, pour qu'ils restent tous visibles), chacun avec ses onglets : (titre du groupe, [(clé dans o, module, titre de l'onglet)]). Un onglet =
# une seule chose à faire, nommée par ce qu'on y fait (demandes de l'utilisateur : « un onglet avec ce que je peux
# faire », puis « une page : une fonction », 03/11).
GROUPES = [
    ("🎤 Voix", [
        ("bibliotheque", bibliotheque, "🎙️ Mes voix (ajouter, écouter, renommer)"),
        ("nettoyer", nettoyer, "🧽 Nettoyer une voix"),
        ("synthese", synthese, "🗣️ Lire un texte avec une voix"),
        ("rvc", entrainement, "🧠 Entraîner un modèle de ma voix"),
        ("doublage", doublage, "🌍 Traduire et doubler"),
    ]),
    ("🎵 Musique", [
        ("chanson", chanson, "🎵 Créer une chanson ou une musique"),
        ("remix", remix, "🎛️ Remixer une musique (autre style)"),
        ("serie_musique", serie_musique, "🗂️ Musiques en série (Excel, liste)"),
        ("passage", passage, "✏️ Refaire un passage d'une chanson"),
        ("karaoke", karaoke, "🎤 Karaoké"),
    ]),
    ("🖼️ Image", [
        ("images", images, "🖼️ Une image (texte ou photo)"),
        ("serie", serie, "🗂️ Images en série (Excel, liste)"),
        ("histoire", histoire, "📖 Histoire en images"),
        ("photos", photos, "📷 Retoucher une photo"),
        ("photos_lot", photos_lot, "📦 Photos en lot"),
        ("costumes", costumes, "👗 Changer les vêtements"),
    ]),
    ("👾 Pixel art", [
        ("pixel_convertir", pixel_convertir, "👾 Convertir en pixel art"),
        ("pixel_personnages", pixel_personnages, "🧙 Personnages pixel art d'après une photo"),
        ("pixel_animer", pixel_animer, "🎞️ Animer un sprite"),
        ("dresseur", dresseur, "🎒 Kit de dresseur"),
        ("sprites_lora", sprites_lora, "🧪 Entraîner un modèle de sprites"),
    ]),
    ("🎬 Vidéo", [
        ("videos", videos, "🎬 Une vidéo"),
        ("plans", plans, "🎞️ Plusieurs plans"),
        ("mini_videos", mini_videos, "🎞️ Mini-vidéos (histoire)"),
        ("parlante", parlante, "🗣️ Faire parler un visage"),
        ("voix_off", voix_off, "🎙️ Voix off"),
        ("fluidite", fluidite, "⏩ Fluidifier / ralentir"),
        ("fond_video", fond_video, "✂️ Retirer le fond"),
        ("sous_titres", sous_titres, "📝 Sous-titrer (transcrire)"),
    ]),
    ("🔧 Audio / vidéo", [
        ("conv_format", conv_format, "🔁 Changer le format"),
        ("conv_son", conv_son, "🎵 Extraire le son"),
        ("conv_couper", conv_couper, "✂️ Couper"),
        ("conv_recoller", conv_recoller, "🔗 Recoller"),
        ("conv_montage", conv_montage, "🎬 Montage (fondus, musique, titre)"),
        ("conv_reduire", conv_reduire, "🪶 Réduire le poids"),
    ]),
    ("🎮 Jeu", [
        ("bande_son", bande_son, "🎼 Musiques de jeu"),
        ("bruitages", bruitages, "🔊 Bruitages"),
        ("illustrations", illustrations, "🃏 Illustrations de cartes"),
        ("personnages_cartes", personnages_cartes, "👤 Personnages récurrents"),
        ("composer", composer, "🃏 Composer une carte"),
        ("pack", pack, "📦 Pack du jeu (export)"),
    ]),
    ("🧊 3D", [
        ("modeles_3d", modeles_3d, "🧊 Image → modèle 3D"),
        ("revision_3d", revision_3d, "✏️ Corriger"),
        ("rigging_3d", rigging_3d, "🦴 Squelette"),
        ("lot_3d", lot_3d, "📚 En lot"),
        ("alleger_3d", alleger_3d, "🪶 Alléger / facetter"),
        ("hd2d", hd2d, "🏔️ HD-2D"),
        ("tileset3d", tileset3d, "🧱 Tileset → objets 3D"),
        ("kit3d", kit3d, "🏙️ Kit d'objets"),
    ]),
    ("📄 PDF", [
        ("pdf", pdf, "📄 Convertir en PDF"),
        ("pdf_fusion", pdf_fusion, "➕ Fusionner"),
        ("pdf_pages", pdf_pages, "✂️ Pages (extraire, diviser)"),
        ("pdf_pivoter", pdf_pivoter, "🔄 Pivoter"),
        ("pdf_compresser", pdf_compresser, "🗜️ Compresser"),
        ("pdf_images", pdf_images, "🖼️ PDF → images"),
        ("pdf_texte", pdf_texte, "📝 PDF → texte / Word"),
    ]),
    ("🖋️ PDF : marquer", [
        ("pdf_numeroter", pdf_numeroter, "🔢 Numéroter les pages"),
        ("pdf_filigrane", pdf_filigrane, "💧 Filigrane"),
        ("pdf_signer", pdf_signer, "✍️ Signer"),
        ("pdf_proteger", pdf_proteger, "🔒 Protéger / déverrouiller"),
    ]),
    ("🧰 Outils", [
        ("galerie", galerie, "🗃️ Galerie (toutes mes créations)"),
        ("modeles", modeles, "⚙️ Modèles (télécharger, importer)"),
        ("diagnostic", diagnostic, "🩺 Diagnostic et carte graphique"),
        ("espace", espace, "💽 Espace disque"),
    ]),
]
ONGLETS = [onglet for _, onglets in GROUPES for onglet in onglets]


def build_ui():
    with gr.Blocks(title="Studio Voix") as demo:
        gr.Markdown("# 🎤 Studio Voix\nVoix, musique, images, vidéos, jeu, PDF : tout est généré en local. Un onglet = une chose à faire.")
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
