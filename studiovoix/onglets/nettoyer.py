"""Onglet « Nettoyer une voix » : un échantillon (fichier, micro ou voix de la bibliothèque) → version nettoyée
(ClearerVoice / VoiceFixer), écoute et comparaison, enregistrement dans la bibliothèque."""
import gradio as gr

from .. import nettoyage
from ..voix import GARDER_NETTOYEE, chemin_voix, list_voices, save_voice_choix
from .commun import espace_de_noms, synchro_toutes_les_voix


def reprendre_voix(name):
    """Recharge une voix de la bibliothèque dans l'échantillon à nettoyer."""
    if not name or name not in list_voices():
        raise gr.Error("Choisis une voix de la bibliothèque.")
    return str(chemin_voix(name)), f"{name} propre", "Voix chargée : choisis un niveau et clique sur « Nettoyer »."


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    gr.Markdown(
        "**Nettoie un enregistrement** (micro bruyant, pièce qui résonne, souffle) avant d'en faire une voix : "
        "« Léger » garde l'articulation intacte, « Fort » retire aussi l'écho mais peut adoucir la diction, "
        "« Maximal » enchaîne les deux. Écoute la version nettoyée, compare avec l'original, puis enregistre-la "
        "dans tes voix sous un nom. Pour une voix déjà enregistrée, charge-la depuis la liste."
    )
    with gr.Row():
        with gr.Column():
            net_audio = gr.Audio(sources=["upload", "microphone"], type="filepath",
                                 label="Échantillon à nettoyer (fichier ou micro)")
            with gr.Row():
                net_voix = gr.Dropdown(choices=list_voices(), value=(list_voices() or [None])[0],
                                       label="Ou une voix déjà enregistrée", scale=3)
                btn_net_reprendre = gr.Button("⬆️ Charger cette voix", scale=1)
        with gr.Column():
            net_niveau = gr.Radio(list(nettoyage.NIVEAUX), value=list(nettoyage.NIVEAUX)[0],
                                  label="Niveau de nettoyage")
            btn_net = gr.Button("🧽 Nettoyer", variant="primary")
            net_clean = gr.Audio(label="Version nettoyée (écoute et compare)", type="filepath", interactive=False)
            net_msg = gr.Markdown()
    with gr.Row():
        net_nom = gr.Textbox(label="Nom de la voix nettoyée", value="laurent propre")
        btn_net_save = gr.Button("💾 Enregistrer la version nettoyée dans mes voix", variant="primary")
    net_save_msg = gr.Markdown()
    net_garder = gr.State(GARDER_NETTOYEE)
    return espace_de_noms(locals())


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    c.btn_net.click(nettoyage.nettoyer, [c.net_audio, c.net_niveau], [c.net_clean, c.net_garder, c.net_msg])
    # Nouvel échantillon : l'ancienne version nettoyée ne lui correspond plus
    c.net_audio.change(lambda: (None, ""), None, [c.net_clean, c.net_msg])
    c.btn_net_reprendre.click(reprendre_voix, c.net_voix, [c.net_audio, c.net_nom, c.net_msg])
    synchro_toutes_les_voix(
        c.btn_net_save.click(lambda a, n, nom: save_voice_choix(a, n, GARDER_NETTOYEE, nom),
                             [c.net_audio, c.net_clean, c.net_nom], [c.net_save_msg, o.bibliotheque.biblio]),
        o, sauf=o.bibliotheque.biblio)
