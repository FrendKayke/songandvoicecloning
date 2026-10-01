"""Onglet « Bibliothèque de voix » : import ou micro, nettoyage, contrôle de qualité, renommage, suppression."""
import gradio as gr

from .. import nettoyage
from ..voix import (
    GARDER_NETTOYEE,
    GARDER_ORIGINAL,
    chemin_voix,
    delete_voice,
    infos_voix,
    list_voices,
    rename_voice,
    save_voice_choix,
)
from .commun import CONFIRMER_SUPPRESSION, espace_de_noms, synchro_voix


def reprendre_voix(name):
    """Recharge une voix de la bibliothèque dans « Ajouter une voix » pour la nettoyer."""
    p = chemin_voix(name)
    return str(p), f"{name} propre", "Voix rechargée dans « Ajouter une voix » : choisis un niveau et clique sur « Nettoyer »."


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    gr.Markdown(
        "### Ajouter une voix\n"
        "Importe un fichier (**wav, mp3 ou flac**) ou enregistre-toi au micro : **10 à 25 secondes**, "
        "dans une pièce calme, sans musique ni écho (idéalement en chantant, sinon en parlant). "
        "La qualité est vérifiée à l'import (durée, volume, saturation)."
    )
    with gr.Row():
        with gr.Column():
            audio_in = gr.Audio(sources=["upload", "microphone"], type="filepath",
                                label="Fichier audio ou enregistrement au micro")
            with gr.Accordion("🧽 Nettoyer la voix (micro bruyant, pièce qui résonne)", open=True):
                niveau = gr.Radio(list(nettoyage.NIVEAUX), value=list(nettoyage.NIVEAUX)[0],
                                  label="Niveau de nettoyage",
                                  info="« Léger » garde l'articulation intacte ; « Fort » retire aussi l'écho "
                                       "mais peut adoucir la diction. Écoute et compare avant d'enregistrer.")
                btn_clean = gr.Button("🧽 Nettoyer l'échantillon")
                audio_clean = gr.Audio(label="Version nettoyée", type="filepath", interactive=False)
                msg_clean = gr.Markdown()
        with gr.Column():
            nom = gr.Textbox(label="Nom de la voix", value="laurent")
            garder = gr.Radio([GARDER_ORIGINAL, GARDER_NETTOYEE], value=GARDER_ORIGINAL,
                              label="Version à enregistrer")
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
            btn_reprendre = gr.Button("🧽 Reprendre cette voix pour la nettoyer")
            btn_del = gr.Button("🗑️ Supprimer", variant="stop")
            msg_biblio = gr.Markdown()
    return espace_de_noms(locals())


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    def synchro_autres(evt):
        """Après une opération sur la bibliothèque : listes de voix des autres onglets à jour."""
        return evt.then(synchro_voix, o.chanson.voix, o.chanson.voix).then(synchro_voix, o.synthese.tts_voix, o.synthese.tts_voix).then(
            lambda v: gr.update(choices=list_voices(), value=[x for x in (v or []) if x in list_voices()]),
            o.rvc.rvc_biblio, o.rvc.rvc_biblio)

    synchro_autres(c.btn_save.click(save_voice_choix, [c.audio_in, c.audio_clean, c.garder, c.nom], [c.msg_voice, c.biblio]))
    c.btn_clean.click(nettoyage.nettoyer, [c.audio_in, c.niveau], [c.audio_clean, c.garder, c.msg_clean])
    # Nouvel échantillon : l'ancienne version nettoyée ne lui correspond plus
    c.audio_in.change(lambda: (None, GARDER_ORIGINAL, ""), None, [c.audio_clean, c.garder, c.msg_clean])
    c.btn_reprendre.click(reprendre_voix, c.biblio, [c.audio_in, c.nom, c.msg_biblio])
    synchro_autres(c.btn_ren.click(rename_voice, [c.biblio, c.nouveau_nom], [c.msg_biblio, c.biblio]))
    synchro_autres(c.btn_del.click(delete_voice, c.biblio, [c.msg_biblio, c.biblio], js=CONFIRMER_SUPPRESSION))
    c.biblio.change(infos_voix, c.biblio, [c.ecoute, c.desc_voix])
    # Voix ajoutées hors de l'application : listes à jour à chaque ouverture de la page
    synchro_autres(demo.load(synchro_voix, c.biblio, c.biblio)).then(infos_voix, c.biblio, [c.ecoute, c.desc_voix])
