"""Onglet « Créer une chanson » : modes, listes de style, description envoyée à ACE-Step, réglages voix, export."""
import gradio as gr

from .. import acestep, export, rvc
from .. import config as cfg
from ..pipeline import MODE_INSTRU, MODE_MA_VOIX, MODES, RETRAITS, creer_chanson
from ..voix import list_voices
from .commun import espace_de_noms, liste_style


def choix_conversion():
    return [("Seed-VC — sans entraînement (voix choisie ci-dessus)", "seedvc")] + [
        (f"RVC — ton modèle « {nom} » ({lib.rsplit('(', 1)[1]}", f"rvc:{nom}") for lib, nom in rvc.choix_modeles()]


def maj_conversion(actuelle):
    choix = choix_conversion()
    valeurs = [v for _, v in choix]
    return gr.update(choices=choix, value=actuelle if actuelle in valeurs else "seedvc")


def maj_mode(mode):
    """Affiche seulement les réglages utiles au mode choisi."""
    ma_voix = mode == MODE_MA_VOIX
    instru = mode == MODE_INSTRU
    label = {MODE_MA_VOIX: "Chanson finale (avec ta voix)", MODE_INSTRU: "Instrumental"}.get(mode, "Chanson (voix d'ACE-Step)")
    return (
        gr.update(visible=ma_voix),                     # voix de la bibliothèque
        gr.update(visible=not instru),                  # paroles
        gr.update(visible=not instru),                  # accordéon « Réglages voix »
        *[gr.update(visible=ma_voix)] * 4,              # décalage, étapes Seed-VC, volumes
        gr.update(label=label),                         # lecteur du résultat
        gr.update(visible=ma_voix),                     # étapes intermédiaires
        gr.update(visible=not instru),                  # voix chantées
    )


def aide_voix(voix_base, mode):
    """Aide affichée quand plusieurs voix chantent : comment dire qui chante quoi, et la limite du mode « ma voix »."""
    if voix_base not in acestep.PLUSIEURS_VOIX or mode == MODE_INSTRU:
        return gr.update(visible=False)
    texte = ("**Plusieurs voix** : dans les paroles, écris qui chante chaque partie après son titre, par exemple "
             "« Couplet 1 (homme) », « Couplet 2 (femme) », « Refrain (ensemble) », « Pont (chœur) ». Entre "
             "parenthèses dans une ligne, « (oh oh) », les mots sont chantés en chœur derrière la voix principale.")
    if mode == MODE_MA_VOIX:
        texte += ("\n\n⚠️ En mode « Chanson avec ma voix », la conversion transforme **toutes** les voix en la "
                  "tienne : tu obtiens des harmonies avec toi-même, pas un duo avec une autre personne. Pour un vrai "
                  "duo homme / femme, choisis le mode « Chanson avec la voix d'ACE-Step ».")
    return gr.update(value=texte, visible=True)


def apercu_description(genre, style, instruments, ambiance, extra, voix_base, mode):
    """Description qui sera envoyée à ACE-Step, recalculée à chaque changement des listes."""
    if mode == MODE_INSTRU:
        voix_base = "Automatique"
    return acestep.build_prompt(genre, style, instruments, ambiance, voix_base, extra)


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    mode = gr.Radio(
        MODES, value=MODE_MA_VOIX, label="Mode",
        info="« Voix d'ACE-Step » et « Instrumental » : musique seule, sans séparation ni conversion de voix.",
    )
    with gr.Row():
        with gr.Column():
            voix = gr.Dropdown(choices=list_voices(), value=(list_voices() or [None])[0], label="Voix à utiliser")
            genre = liste_style("genre", "Genre")
            style = liste_style("style", "Style / époque / production")
            instruments = liste_style("instruments", "Instruments")
            ambiance = liste_style("ambiance", "Ambiance")
            extra = liste_style("extra", "Autres consignes (facultatif)")
        with gr.Column():
            voix_base = gr.Dropdown(
                list(acestep.VOIX_CHANTEES), value="Automatique", label="Voix chantées",
                info="Une voix (homme ou femme), un duo, une voix principale avec des chœurs, ou un chœur. En mode "
                     "« ma voix », choisis le genre le plus proche de ta voix : moins de décalage à corriger.",
            )
            aide_plusieurs_voix = gr.Markdown(visible=False)
            paroles = gr.Textbox(
                label="Paroles (titres de parties : Couplet 1, Refrain, Pont… ; qui chante : « Refrain (ensemble) »)",
                lines=16,
                placeholder="Couplet 1 (homme)\nTes paroles…\n\nCouplet 2 (femme)\n…\n\nRefrain (ensemble)\n"
                            "Le refrain… (oh oh)",
            )
    description = gr.Textbox(
        label="Description envoyée à ACE-Step (modifiable)", lines=2,
        info="Construite à partir des listes ci-dessus, en anglais : c'est la langue que le modèle comprend le "
             "mieux. Tu peux la retoucher ; elle est recalculée si tu changes une liste. Pour exclure un "
             "instrument, ne l'écris pas ici (« sans basse » ajouterait de la basse).",
    )
    retirer = gr.Dropdown(
        [(v.capitalize(), k) for k, v in RETRAITS.items()], value=[], multiselect=True,
        label="Retirer de la musique",
        info="Garanti : la chanson est séparée en pistes (Demucs) et l'instrument choisi est supprimé du "
             "mix (environ 1 min de plus). En mode réflexion, ACE-Step est aussi prié de l'éviter.",
    )
    with gr.Row():
        langue = gr.Dropdown(list(cfg.LANGUES), value="Français", label="Langue des paroles")
        duree = gr.Slider(30, 240, value=120, step=10, label="Durée (s)")
        bpm = gr.Number(value=0, precision=0, label="BPM (0 = auto)")
        versions = gr.Radio([1, 2], value=1, label="Versions",
                            info="2 versions d'un coup pour garder la meilleure (plus long).")
        graine = gr.Number(value=0, precision=0, label="Graine (0 = aléatoire)",
                           info="Reprends une graine affichée après une création pour obtenir un résultat proche.")
        thinking = gr.Checkbox(value=True, label="Mode réflexion (LM) — meilleure structure",
                               info="Si le style demandé n'est pas respecté, décoche-le : le générateur "
                                    "suivra alors la description seule.")
    with gr.Accordion("Réglages voix (avancé)", open=False) as reglages:
        conversion = gr.Dropdown(choix_conversion(), value="seedvc", label="Conversion de ta voix",
                                 info="Un modèle RVC entraîné sur 10 à 30 min de ta voix est plus fidèle "
                                      "(onglet « Entraîner un modèle de ma voix »).")
        semitones = gr.Slider(-12, 12, value=0, step=1, label="Décalage de hauteur (demi-tons)",
                              info="Voix de base féminine → voix masculine : essaie -12. L'inverse : +12.")
        steps = gr.Slider(25, 50, value=40, step=5, label="Étapes de diffusion Seed-VC (30–50 conseillé pour le chant)")
        gain_voix = gr.Slider(0.5, 1.5, value=1.0, step=0.05, label="Volume voix")
        gain_instru = gr.Slider(0.5, 1.5, value=1.0, step=0.05, label="Volume instrumental")

    btn = gr.Button("🎵 Créer la chanson", variant="primary")
    statut = gr.Markdown()
    final = gr.Audio(label="Chanson finale (avec ta voix)", type="filepath")
    final_2 = gr.Audio(label="Version 2", type="filepath", visible=False)
    with gr.Row():
        chanson_cible = gr.Dropdown(list(export.CIBLES), value=list(export.CIBLES)[0], label="Volume de l'export")
        btn_export_chanson = gr.Button("💾 Exporter la chanson en MP3")
        chanson_mp3 = gr.File(label="MP3 à télécharger")
    chanson_export_msg = gr.Markdown()
    with gr.Accordion("Étapes intermédiaires", open=False) as intermediaires:
        brute = gr.Audio(label="Chanson brute ACE-Step", type="filepath")
        voix_conv = gr.Audio(label="Voix convertie", type="filepath")
        instru_out = gr.Audio(label="Instrumental", type="filepath")
    return espace_de_noms(locals())


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    c.btn.click(
        creer_chanson,
        [c.voix, c.genre, c.style, c.instruments, c.ambiance, c.extra, c.voix_base, c.paroles, c.langue, c.duree, c.bpm,
         c.thinking, c.semitones, c.steps, c.gain_voix, c.gain_instru, c.mode, c.description, c.retirer, c.versions, c.graine,
         c.conversion],
        [c.final, c.brute, c.voix_conv, c.instru_out, c.statut, c.final_2],
    )
    c.versions.change(lambda v: gr.update(visible=int(v) > 1), c.versions, c.final_2)
    c.btn_export_chanson.click(export.exporter_fichier, [c.final, c.chanson_cible], [c.chanson_mp3, c.chanson_export_msg])

    champs_style = [c.genre, c.style, c.instruments, c.ambiance, c.extra, c.voix_base, c.mode]
    for champ in champs_style:
        champ.change(apercu_description, champs_style, c.description)
    c.mode.change(
        maj_mode, c.mode,
        [c.voix, c.paroles, c.reglages, c.semitones, c.steps, c.gain_voix, c.gain_instru, c.final, c.intermediaires,
         c.voix_base],
    )
    for champ in (c.voix_base, c.mode):
        champ.change(aide_voix, [c.voix_base, c.mode], c.aide_plusieurs_voix)
