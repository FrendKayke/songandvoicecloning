"""Interface Gradio (onglets « Bibliothèque de voix », « Créer une chanson », « Synthèse vocale », « Modèles »)."""
import gradio as gr

from . import acestep, chatterbox, demucs, seedvc
from . import config as cfg
from .modeles import models_status_md
from .outils import open_folder
from .pipeline import MODE_INSTRU, MODE_MA_VOIX, MODES, RETRAITS, creer_chanson
from .styles import LISTES
from .voix import delete_voice, infos_voix, list_voices, rename_voice, save_voice

# Fenêtre de confirmation du navigateur avant suppression (annuler → None → rien n'est supprimé)
CONFIRMER_SUPPRESSION = "(v) => (v && confirm('Supprimer définitivement la voix « ' + v + ' » ?')) ? v : null"


def synchro_voix(courante):
    """Met à jour une liste de voix d'un autre onglet en gardant la sélection si elle existe encore."""
    voices = list_voices()
    return gr.update(choices=voices, value=courante if courante in voices else (voices[0] if voices else None))


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
    )


def apercu_description(genre, style, instruments, ambiance, extra, voix_base, mode):
    """Description qui sera envoyée à ACE-Step, recalculée à chaque changement des listes."""
    if mode == MODE_INSTRU:
        voix_base = "Automatique"
    return acestep.build_prompt(genre, style, instruments, ambiance, voix_base, extra)


def liste_style(cle, label, info=None):
    """Liste déroulante à choix multiples (libellés français, termes anglais) qui accepte aussi la saisie libre."""
    return gr.Dropdown(
        LISTES[cle], value=[], multiselect=True, allow_custom_value=True, label=label,
        info=info or "Choisis dans la liste, ou tape ton propre terme (en anglais de préférence) puis Entrée.",
    )


def build_ui():
    with gr.Blocks(title="Studio Voix") as demo:
        gr.Markdown("# 🎤 Studio Voix\nClone ta voix, écris tes paroles, choisis le style : la chanson est générée en local.")

        with gr.Tab("1. Bibliothèque de voix"):
            gr.Markdown(
                "### Ajouter une voix\n"
                "Importe un fichier (**wav, mp3 ou flac**) ou enregistre-toi au micro : **10 à 25 secondes**, "
                "dans une pièce calme, sans musique ni écho (idéalement en chantant, sinon en parlant). "
                "La qualité est vérifiée à l'import (durée, volume, saturation)."
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

        with gr.Tab("2. Créer une chanson"):
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
                    paroles = gr.Textbox(
                        label="Paroles (avec [Verse], [Chorus], [Bridge]…)", lines=16,
                        placeholder="[Verse 1]\nTes paroles…\n\n[Chorus]\nLe refrain…",
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
                thinking = gr.Checkbox(value=True, label="Mode réflexion (LM) — meilleure structure",
                                       info="Si le style demandé n'est pas respecté, décoche-le : le générateur "
                                            "suivra alors la description seule.")
            with gr.Accordion("Réglages voix (avancé)", open=False) as reglages:
                voix_base = gr.Dropdown(
                    ["Automatique", "Voix masculine", "Voix féminine"], value="Automatique",
                    label="Voix chantée de base générée par ACE-Step",
                    info="Choisis le genre le plus proche de ta voix : moins de décalage à corriger.",
                )
                semitones = gr.Slider(-12, 12, value=0, step=1, label="Décalage de hauteur (demi-tons)",
                                      info="Voix de base féminine → voix masculine : essaie -12. L'inverse : +12.")
                steps = gr.Slider(25, 50, value=40, step=5, label="Étapes de diffusion Seed-VC (30–50 conseillé pour le chant)")
                gain_voix = gr.Slider(0.5, 1.5, value=1.0, step=0.05, label="Volume voix")
                gain_instru = gr.Slider(0.5, 1.5, value=1.0, step=0.05, label="Volume instrumental")

            btn = gr.Button("🎵 Créer la chanson", variant="primary")
            statut = gr.Markdown()
            final = gr.Audio(label="Chanson finale (avec ta voix)", type="filepath")
            with gr.Accordion("Étapes intermédiaires", open=False) as intermediaires:
                brute = gr.Audio(label="Chanson brute ACE-Step", type="filepath")
                voix_conv = gr.Audio(label="Voix convertie", type="filepath")
                instru_out = gr.Audio(label="Instrumental", type="filepath")

        with gr.Tab("3. Synthèse vocale"):
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
            tts_btn = gr.Button("🗣️ Lire le texte avec cette voix", variant="primary")
            tts_statut = gr.Markdown()
            tts_sortie = gr.Audio(label="Parole générée", type="filepath")

        with gr.Tab("4. Modèles"):
            gr.Markdown(
                "Les modèles sont volumineux (plusieurs Go au total) et ne sont téléchargés qu'une seule fois. "
                "Vérifie ici leur présence et leur emplacement, ou lance le téléchargement."
            )
            status = gr.Markdown(models_status_md())
            btn_refresh = gr.Button("🔄 Actualiser l'état")
            log = gr.Textbox(label="Journal de téléchargement", lines=14, max_lines=14, autoscroll=True, interactive=False)
            with gr.Row():
                b_ace = gr.Button("⬇️ Télécharger ACE-Step", variant="primary")
                b_sv = gr.Button("⬇️ Télécharger Seed-VC", variant="primary")
                b_dm = gr.Button("⬇️ Télécharger Demucs", variant="primary")
                b_cb = gr.Button("⬇️ Télécharger Chatterbox", variant="primary")
            with gr.Row():
                o_ace = gr.Button("📂 Ouvrir dossier ACE-Step")
                o_sv = gr.Button("📂 Ouvrir dossier Seed-VC")
                o_dm = gr.Button("📂 Ouvrir dossier Demucs")
                o_cb = gr.Button("📂 Ouvrir dossier Chatterbox")
                o_data = gr.Button("📂 Ouvrir mes chansons")

        def synchro_autres(evt):
            """Après une opération sur la bibliothèque : listes de voix des autres onglets à jour."""
            return evt.then(synchro_voix, voix, voix).then(synchro_voix, tts_voix, tts_voix)

        synchro_autres(btn_save.click(save_voice, [audio_in, nom], [msg_voice, biblio]))
        synchro_autres(btn_ren.click(rename_voice, [biblio, nouveau_nom], [msg_biblio, biblio]))
        synchro_autres(btn_del.click(delete_voice, biblio, [msg_biblio, biblio], js=CONFIRMER_SUPPRESSION))
        biblio.change(infos_voix, biblio, [ecoute, desc_voix])
        # Voix ajoutées hors de l'application : listes à jour à chaque ouverture de la page
        synchro_autres(demo.load(synchro_voix, biblio, biblio)).then(infos_voix, biblio, [ecoute, desc_voix])
        tts_btn.click(
            chatterbox.synthese,
            [tts_voix, tts_texte, tts_langue, tts_exag, tts_cfg, tts_temp, tts_graine],
            [tts_sortie, tts_statut],
        )
        btn_refresh.click(models_status_md, None, status)
        for b, fn in ((b_ace, acestep.download), (b_sv, seedvc.download), (b_dm, demucs.download),
                      (b_cb, chatterbox.download)):
            b.click(fn, None, log).then(models_status_md, None, status)
        o_ace.click(lambda: open_folder(acestep.ckpt_dir()))
        o_sv.click(lambda: open_folder(seedvc.ckpt_dir()))
        o_dm.click(lambda: open_folder(demucs.ckpt_dir()))
        o_cb.click(lambda: open_folder(chatterbox.ckpt_dir()))
        o_data.click(lambda: open_folder(cfg.SONGS_DIR, create=True))
        demo.load(models_status_md, None, status)
        btn.click(
            creer_chanson,
            [voix, genre, style, instruments, ambiance, extra, voix_base, paroles, langue, duree, bpm,
             thinking, semitones, steps, gain_voix, gain_instru, mode, description, retirer],
            [final, brute, voix_conv, instru_out, statut],
        )
        champs_style = [genre, style, instruments, ambiance, extra, voix_base, mode]
        for champ in champs_style:
            champ.change(apercu_description, champs_style, description)
        mode.change(
            maj_mode, mode,
            [voix, paroles, reglages, semitones, steps, gain_voix, gain_instru, final, intermediaires],
        )
    return demo
