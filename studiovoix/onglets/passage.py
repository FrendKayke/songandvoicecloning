"""Onglet « Refaire un passage » (groupe Musique) : un passage d'une chanson ou d'une piste de jeu est réinventé
(tâche repaint d'ACE-Step), le reste est gardé, puis le traitement est refait. Nouvelle création."""
import gradio as gr

from .. import galerie
from ..outils import open_folder
from .commun import espace_de_noms

FILTRE = "Chansons et pistes de jeu"


def lister():
    """Chansons et pistes de jeu, les plus récentes d'abord."""
    chansons = galerie.lister("Chansons")
    pistes = galerie.lister("Bande-son de jeu")
    return sorted(chansons + pistes, key=lambda e: e[1], reverse=True)


def maj_liste(choisie=None):
    choix = lister()
    valeurs = [v for _, v in choix]
    return gr.update(choices=choix, value=choisie if choisie in valeurs else (valeurs[0] if valeurs else None))


def _details(chemin, version=1):
    """Détails de la création pour cet onglet : description, lecteur, versions, champs du passage."""
    md, audio, versions, desc, paroles, fin, *_ = galerie.details(chemin, version)
    return md, audio, versions, desc, paroles, fin


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    gr.Markdown(
        "Un refrain raté, une fin bizarre ? Indique le passage en secondes : seul ce passage est **réinventé** "
        "(tâche *repaint* d'ACE-Step), le reste est gardé, puis la suite du traitement est refaite (ta voix, "
        "retrait d'instruments, boucle…). Le résultat est une **nouvelle création**, l'originale reste dans la "
        "galerie. Tu peux modifier la description ou les paroles du passage."
    )
    with gr.Row():
        pas_liste = gr.Dropdown([], label="Chanson ou piste de jeu (la plus récente en premier)", scale=4)
        pas_maj = gr.Button("🔄 Actualiser", scale=0)
    with gr.Row():
        with gr.Column(scale=3):
            pas_details = gr.Markdown()
        with gr.Column(scale=2):
            pas_version = gr.Radio([1], value=1, label="Version", visible=False)
            pas_audio = gr.Audio(type="filepath", label="Écouter l'original")
    with gr.Row():
        pas_debut = gr.Number(value=0, label="Début (s)")
        pas_fin = gr.Number(value=10, label="Fin (s)")
        pas_force = gr.Radio(list(galerie.FORCES), value=list(galerie.FORCES)[1], label="Retouche")
    pas_desc = gr.Textbox(label="Description", lines=2)
    pas_paroles = gr.Textbox(label="Paroles (chansons)", lines=6)
    btn_pas = gr.Button("✏️ Refaire ce passage", variant="primary")
    pas_msg = gr.Markdown()
    pas_etat = gr.State()
    btn_pas_dossier = gr.Button("📂 Ouvrir le dossier de la nouvelle création")
    return espace_de_noms(locals())


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    sorties = [c.pas_details, c.pas_audio, c.pas_version, c.pas_desc, c.pas_paroles, c.pas_fin]
    for evt in (c.pas_maj.click, demo.load, c.onglet.select):
        evt(maj_liste, c.pas_liste, c.pas_liste)
    c.pas_liste.change(_details, c.pas_liste, sorties)
    c.pas_version.input(_details, [c.pas_liste, c.pas_version], sorties)
    c.btn_pas.click(galerie.refaire_passage,
                    [c.pas_liste, c.pas_version, c.pas_debut, c.pas_fin, c.pas_desc, c.pas_paroles, c.pas_force],
                    [c.pas_msg, c.pas_etat]).then(maj_liste, c.pas_etat, c.pas_liste)
    c.btn_pas_dossier.click(lambda d: open_folder(d) if d else None, c.pas_etat)
