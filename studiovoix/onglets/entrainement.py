"""Onglet « Entraîner ma voix (RVC) » : jeu d'enregistrements, entraînement, modèles prêts."""
import gradio as gr

from .. import rvc
from ..voix import list_voices
from .chanson import maj_conversion
from .commun import espace_de_noms


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    gr.Markdown(
        "Entraîne un **modèle de ta voix** (RVC) : bien plus fidèle que la conversion sans entraînement, "
        "surtout pour le chant. Il faut **10 à 30 minutes** d'enregistrements de toi seul, propres (sans "
        "musique, écho ni bruit), variés : chante des mélodies graves et aiguës, parle, lis un texte. "
        "Plusieurs fichiers courts vont très bien. Nettoie-les d'abord dans la Bibliothèque si ton micro est "
        "bruyant.\n\nSur ta RTX 4070, compte environ **1 à 2 heures** pour 300 époques avec 15 minutes "
        "d'enregistrements. Laisse la page ouverte ; si tu la fermes, l'entraînement continue et le modèle "
        "apparaîtra dans la liste. Relancer avec le même nom reprend un entraînement interrompu."
    )
    with gr.Row():
        with gr.Column():
            rvc_fichiers = gr.File(file_count="multiple", file_types=[".wav", ".mp3", ".flac"],
                                   label="Tes enregistrements (wav, mp3, flac)")
            rvc_biblio = gr.Dropdown(choices=list_voices(), value=[], multiselect=True,
                                     label="Ajouter des voix de ta bibliothèque (facultatif)")
        with gr.Column():
            rvc_nom = gr.Textbox(label="Nom du modèle", value="ma voix")
            rvc_duree = gr.Radio(list(rvc.DUREES), value=list(rvc.DUREES)[1], label="Durée d'entraînement")
            rvc_lot = gr.Slider(2, 16, value=8, step=2, label="Taille de lot",
                                info="8 convient à 12 Go de mémoire graphique ; baisse-la en cas d'erreur mémoire.")
            btn_rvc = gr.Button("🧠 Entraîner le modèle", variant="primary")
    rvc_controle = gr.Markdown(rvc.analyser_enregistrements(None, None))
    rvc_msg = gr.Markdown()
    gr.Markdown("### Mes modèles")
    rvc_tableau = gr.Markdown(rvc.tableau_modeles())
    with gr.Row():
        rvc_choix = gr.Dropdown(rvc.choix_modeles(), label="Modèle")
        btn_rvc_suppr = gr.Button("🗑️ Supprimer ce modèle", variant="stop")
    return espace_de_noms(locals())


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    for champ in (c.rvc_fichiers, c.rvc_biblio):
        champ.change(rvc.analyser_enregistrements, [c.rvc_fichiers, c.rvc_biblio], c.rvc_controle)

    def apres_modeles(evt):
        """Listes de modèles RVC à jour partout après un entraînement ou une suppression."""
        return evt.then(rvc.tableau_modeles, None, c.rvc_tableau).then(rvc.maj_modeles, c.rvc_choix, c.rvc_choix).then(
            maj_conversion, o.chanson.conversion, o.chanson.conversion).then(
            lambda v: gr.update(choices=[("Aucun", None)] + rvc.choix_modeles(), value=v), o.synthese.tts_rvc, o.synthese.tts_rvc)

    apres_modeles(c.btn_rvc.click(rvc.entrainer, [c.rvc_nom, c.rvc_fichiers, c.rvc_biblio, c.rvc_duree, c.rvc_lot], c.rvc_msg))
    apres_modeles(c.btn_rvc_suppr.click(rvc.supprimer_modele, c.rvc_choix, [c.rvc_msg, c.rvc_tableau],
                                      js="(m) => (m && confirm('Supprimer définitivement le modèle « ' + m + ' » ?')) ? m : null"))
    apres_modeles(demo.load(lambda: None, None, None))
