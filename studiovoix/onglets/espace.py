"""Onglet « Espace disque » (groupe Outils) : place prise par chaque moteur, retrait et réinstallation."""
import gradio as gr

from .. import espace
from .commun import espace_de_noms

CONFIRMER_RETRAIT = ("(c) => (c && confirm('Supprimer ces fichiers ? Tu pourras les réinstaller plus tard avec "
                     "« Réinstaller » puis METTRE_A_JOUR.bat.')) ? c : null")
CONFIRMER_VIDAGE = "(c) => (c && confirm('Vider ce dossier ?')) ? c : null"


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    gr.Markdown(
        "Les modèles occupent environ 100 Go. Retire ceux que tu n'utilises pas : ils ne seront plus réinstallés par "
        "les mises à jour, jusqu'à ce que tu cliques sur « Réinstaller » (puis METTRE_A_JOUR.bat). ACE-Step, Seed-VC, "
        "tes créations et tes modèles RVC entraînés ne sont jamais supprimés ici. La mesure parcourt tous les "
        "dossiers : elle peut prendre une minute."
    )
    btn_mesurer = gr.Button("📊 Mesurer l'espace occupé")
    tableau = gr.Markdown()
    with gr.Row():
        choix = gr.Dropdown(espace.CHOIX_RETRAIT, value=None, label="Moteur ou modèle")
        btn_retirer = gr.Button("🗑️ Retirer (supprimer ses fichiers)", variant="stop")
        btn_reinstaller = gr.Button("↩️ Réinstaller")
    with gr.Row():
        choix_vider = gr.Dropdown(espace.CHOIX_VIDER, value=None, label="Dossier à vider")
        btn_vider = gr.Button("🧹 Vider")
    msg = gr.Markdown()
    return espace_de_noms(locals())


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    c.btn_mesurer.click(espace.inventaire, None, c.tableau)
    c.btn_retirer.click(espace.retirer, c.choix, [c.msg, c.tableau], js=CONFIRMER_RETRAIT)
    c.btn_reinstaller.click(espace.reinstaller, c.choix, [c.msg, c.tableau])
    c.btn_vider.click(espace.vider, c.choix_vider, [c.msg, c.tableau], js=CONFIRMER_VIDAGE)
