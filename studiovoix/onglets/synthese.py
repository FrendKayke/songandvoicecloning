"""Onglet « Synthèse vocale » : lecture d'un texte avec une voix de la bibliothèque (+ RVC facultatif)."""
import json
from pathlib import Path

import gradio as gr

from .. import chatterbox, rvc
from .. import config as cfg
from ..voix import list_voices
from .commun import espace_de_noms


def synthese_puis_rvc(voix, texte, langue, exag, cfg_w, temp, graine, modele_rvc, demi_tons, progress=gr.Progress()):
    """Synthèse vocale, puis (facultatif) passage dans un modèle RVC pour coller davantage à ta voix."""
    fichier, msg = chatterbox.synthese(voix, texte, langue, exag, cfg_w, temp, graine, progress=progress)
    if not modele_rvc:
        return fichier, msg
    sortie = rvc.convertir(fichier, modele_rvc, demi_tons, Path(fichier).with_name("parole_rvc.wav"), progress=progress)
    creation = Path(fichier).parent / "creation.json"
    infos = json.loads(creation.read_text(encoding="utf-8"))
    infos["rvc"] = modele_rvc
    infos["versions"][0]["fichier"] = str(sortie)
    creation.write_text(json.dumps(infos, ensure_ascii=False, indent=1), encoding="utf-8")
    return str(sortie), f"{msg} Passée dans le modèle RVC « {modele_rvc} » : {sortie}"


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    gr.Markdown(
        "Fais lire un texte par une voix de ta bibliothèque (Chatterbox Multilingual, en local). "
        "Les textes longs sont découpés en phrases. Le premier lancement charge le modèle (environ 30 s). "
        "Chaque fichier produit porte un filigrane inaudible (Perth) qui le signale comme voix de synthèse."
    )
    with gr.Row():
        with gr.Column(scale=1):
            tts_voix = gr.Dropdown(choices=list_voices(), value=(list_voices() or [None])[0],
                                   label="Voix")
            tts_langue = gr.Dropdown(list(cfg.LANGUES), value="Français", label="Langue du texte")
        tts_texte = gr.Textbox(label="Texte à lire", lines=10, scale=2,
                               placeholder="Bonjour ! Ceci est un essai de ma voix de synthèse.")
    with gr.Accordion("Réglages (avancé)", open=False):
        tts_exag = gr.Slider(0.25, 2, value=0.5, step=0.05, label="Expressivité",
                             info="0,5 = neutre. Plus haut : plus expressif (et plus rapide) ; les extrêmes sont instables.")
        tts_cfg = gr.Slider(0, 1, value=0.5, step=0.05, label="Guidage / rythme",
                            info="Vers 0,3 si ta voix de référence parle vite. 0 si l'échantillon n'est pas "
                                 "dans la langue du texte (évite de garder l'accent).")
        tts_temp = gr.Slider(0.05, 5, value=0.8, step=0.05, label="Température (variété)")
        tts_graine = gr.Number(value=0, precision=0, label="Graine (0 = aléatoire, sinon résultat reproductible)")
    with gr.Row():
        tts_rvc = gr.Dropdown([("Aucun", None)] + rvc.choix_modeles(), value=None,
                              label="Passer ensuite dans un modèle RVC (facultatif)",
                              info="Rapproche encore la lecture de ta voix si tu as entraîné un modèle.")
        tts_rvc_ton = gr.Slider(-12, 12, value=0, step=1, label="Décalage de hauteur RVC (demi-tons)")
    tts_btn = gr.Button("🗣️ Lire le texte avec cette voix", variant="primary")
    tts_statut = gr.Markdown()
    tts_sortie = gr.Audio(label="Parole générée", type="filepath")
    return espace_de_noms(locals())


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    c.tts_btn.click(
        synthese_puis_rvc,
        [c.tts_voix, c.tts_texte, c.tts_langue, c.tts_exag, c.tts_cfg, c.tts_temp, c.tts_graine, c.tts_rvc, c.tts_rvc_ton],
        [c.tts_sortie, c.tts_statut],
    )
