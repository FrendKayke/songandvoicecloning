"""Onglet « Mes voix » : import ou micro avec contrôle de qualité, écoute, renommage, suppression. Le nettoyage a son
onglet (« Nettoyer une voix »)."""
import gradio as gr

from ..voix import delete_voice, infos_voix, list_voices, rename_voice, save_voice
from .commun import CONFIRMER_SUPPRESSION, espace_de_noms, synchro_toutes_les_voix, synchro_voix


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    gr.Markdown(
        "### Ajouter une voix\n"
        "Importe un fichier (**wav, mp3, flac ou m4a**) ou enregistre-toi au micro : **10 à 25 secondes**, "
        "dans une pièce calme, sans musique ni écho (idéalement en chantant, sinon en parlant). "
        "La qualité est vérifiée à l'import (durée, volume, saturation). Micro bruyant, pièce qui résonne : "
        "passe d'abord par l'onglet « 🧽 Nettoyer une voix »."
    )
    with gr.Row():
        audio_in = gr.Audio(sources=["upload", "microphone"], type="filepath",
                            label="Fichier audio ou enregistrement au micro")
        with gr.Column():
            nom = gr.Textbox(label="Nom de la voix", value="laurent")
            btn_save = gr.Button("Vérifier et enregistrer cette voix", variant="primary")
            msg_voice = gr.Markdown()
    gr.Markdown("### Mes voix")
    with gr.Row():
        with gr.Column():
            biblio = gr.Dropdown(choices=list_voices(), value=(list_voices() or [None])[0],
                                 label="Voix enregistrées")
            desc_voix = gr.Markdown()
            ecoute = gr.Audio(label="Écouter", type="filepath", interactive=False)
        with gr.Column():
            nouveau_nom = gr.Textbox(label="Nouveau nom")
            btn_ren = gr.Button("✏️ Renommer")
            btn_del = gr.Button("🗑️ Supprimer", variant="stop")
            msg_biblio = gr.Markdown()
    return espace_de_noms(locals())


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    def synchro_autres(evt):
        """Après une opération sur la bibliothèque : listes de voix des autres onglets à jour."""
        return synchro_toutes_les_voix(evt, o, sauf=c.biblio)

    synchro_autres(c.btn_save.click(save_voice, [c.audio_in, c.nom], [c.msg_voice, c.biblio]))
    synchro_autres(c.btn_ren.click(rename_voice, [c.biblio, c.nouveau_nom], [c.msg_biblio, c.biblio]))
    synchro_autres(c.btn_del.click(delete_voice, c.biblio, [c.msg_biblio, c.biblio], js=CONFIRMER_SUPPRESSION))
    c.biblio.change(infos_voix, c.biblio, [c.ecoute, c.desc_voix])
    # Voix ajoutées hors de l'application : listes à jour à chaque ouverture de la page
    synchro_autres(demo.load(synchro_voix, c.biblio, c.biblio)).then(infos_voix, c.biblio, [c.ecoute, c.desc_voix])
