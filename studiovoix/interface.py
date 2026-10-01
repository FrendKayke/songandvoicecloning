"""Interface Gradio (onglets « Bibliothèque de voix », « Créer une chanson », « Synthèse vocale », « Modèles »)."""
import json
from pathlib import Path

import gradio as gr

from . import (acestep, bruitages, chatterbox, demucs, diffusion, export, galerie, jeu, modele3d, nettoyage, rvc,
               seedvc, serveur_acestep)
from . import config as cfg
from .modeles import models_status_md
from .outils import open_folder
from .pipeline import MODE_INSTRU, MODE_MA_VOIX, MODES, RETRAITS, creer_chanson
from .styles import LISTES
from .voix import (GARDER_NETTOYEE, GARDER_ORIGINAL, chemin_voix, delete_voice, infos_voix, list_voices,
                   rename_voice, save_voice_choix)

# Fenêtre de confirmation du navigateur avant suppression (annuler → None → rien n'est supprimé)
CONFIRMER_SUPPRESSION_CREATION = "(c) => (c && confirm('Supprimer définitivement cette création et tous ses fichiers ?')) ? c : null"
CONFIRMER_SUPPRESSION = "(v) => (v && confirm('Supprimer définitivement la voix « ' + v + ' » ?')) ? v : null"


def reprendre_voix(name):
    """Recharge une voix de la bibliothèque dans « Ajouter une voix » pour la nettoyer."""
    p = chemin_voix(name)
    return str(p), f"{name} propre", "Voix rechargée dans « Ajouter une voix » : choisis un niveau et clique sur « Nettoyer »."


def choix_conversion():
    return [("Seed-VC — sans entraînement (voix choisie ci-dessus)", "seedvc")] + [
        (f"RVC — ton modèle « {nom} » ({lib.rsplit('(', 1)[1]}", f"rvc:{nom}") for lib, nom in rvc.choix_modeles()]


def maj_conversion(actuelle):
    choix = choix_conversion()
    valeurs = [v for _, v in choix]
    return gr.update(choices=choix, value=actuelle if actuelle in valeurs else "seedvc")


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
                versions = gr.Radio([1, 2], value=1, label="Versions",
                                    info="2 versions d'un coup pour garder la meilleure (plus long).")
                graine = gr.Number(value=0, precision=0, label="Graine (0 = aléatoire)",
                                   info="Reprends une graine affichée après une création pour obtenir un résultat proche.")
                thinking = gr.Checkbox(value=True, label="Mode réflexion (LM) — meilleure structure",
                                       info="Si le style demandé n'est pas respecté, décoche-le : le générateur "
                                            "suivra alors la description seule.")
            with gr.Accordion("Réglages voix (avancé)", open=False) as reglages:
                voix_base = gr.Dropdown(
                    ["Automatique", "Voix masculine", "Voix féminine"], value="Automatique",
                    label="Voix chantée de base générée par ACE-Step",
                    info="Choisis le genre le plus proche de ta voix : moins de décalage à corriger.",
                )
                conversion = gr.Dropdown(choix_conversion(), value="seedvc", label="Conversion de ta voix",
                                         info="Un modèle RVC entraîné sur 10 à 30 min de ta voix est plus fidèle "
                                              "(onglet « Entraîner ma voix »).")
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

        with gr.Tab("3. Bande-son de jeu"):
            gr.Markdown(
                "Musiques **sans voix** pour ton jeu, par situation : écran titre, combat, victoire… "
                "Les musiques de fond sont faites pour tourner en boucle, les jingles sont courts. "
                "On décrit un style (JRPG, 16-bit…), jamais une œuvre existante : la musique produite est originale."
            )
            with gr.Row():
                jeu_projet = gr.Textbox(label="Nom du projet", value="mon jeu de cartes")
                jeu_epoque = gr.Dropdown(jeu.EPOQUES, value=jeu.EPOQUES[0][1], allow_custom_value=True,
                                         label="Époque / style général",
                                         info="Choisis dans la liste ou tape ton style (en anglais de préférence).")
                jeu_univers = gr.Dropdown(jeu.UNIVERS, value=[], multiselect=True, allow_custom_value=True,
                                          label="Univers", info="Facultatif ; plusieurs choix possibles.")
            jeu_situations = gr.Dropdown(
                jeu.choix_situations(), value=["titre", "combat", "victoire"], multiselect=True,
                allow_custom_value=True, label="Situations à générer",
                info="Tu peux taper ta propre situation (en anglais, par ex. « fire faction theme, aggressive "
                     "taiko drums ») : elle sera générée comme musique en boucle.",
            )
            jeu_extra = liste_style("instruments", "Instruments à ajouter à toutes les pistes (facultatif)")
            with gr.Accordion("🎼 Cohérence : partir d'un thème du projet", open=False):
                gr.Markdown(
                    "Génère d'abord un **thème principal** (par exemple l'écran titre), puis choisis-le ici : "
                    "« Même son » donne à toutes les pistes le même timbre et le même mixage ; « Variation du thème » "
                    "réarrange sa mélodie selon chaque situation (version combat, version calme…), comme les "
                    "leitmotivs des JRPG. En variation, les musiques de fond prennent la durée du thème et les "
                    "jingles utilisent seulement le même son."
                )
                with gr.Row():
                    jeu_ref = gr.Dropdown([], label="Thème de référence (pistes du projet)")
                    jeu_usage = gr.Radio(jeu.REFERENCES, value=jeu.REF_AUCUNE, label="Utilisation")
                    jeu_fidelite = gr.Slider(0.1, 1.0, value=0.5, step=0.05, label="Fidélité au thème (variation)",
                                             info="Haut : très proche du thème. Bas : plus libre.")
            with gr.Row():
                jeu_duree = gr.Slider(30, 180, value=90, step=10, label="Durée des musiques en boucle (s)")
                jeu_thinking = gr.Checkbox(value=False, label="Mode réflexion (LM)",
                                           info="Désactivé par défaut : la description est suivie plus fidèlement.")
                jeu_graine = gr.Number(value=0, precision=0, label="Graine (0 = aléatoire)")
            jeu_apercu = gr.Markdown()
            btn_jeu = gr.Button("🎮 Générer la bande-son", variant="primary")
            jeu_statut = gr.Markdown()
            with gr.Row():
                jeu_liste = gr.Dropdown([], label="Écouter une piste générée")
                jeu_audio = gr.Audio(label="Piste (en boucle dans le jeu)", type="filepath")
                jeu_jonction = gr.Audio(label="Jonction : 5 s de fin puis 5 s de début", type="filepath")
            with gr.Accordion("📦 Export pour le jeu (OGG / MP3, volume harmonisé, manifest.json)", open=False):
                gr.Markdown(
                    "Prend la piste la plus récente de chaque situation du projet, les met toutes au même volume "
                    "et écrit `data/jeux/<projet>/export/` : un fichier par situation et un `manifest.json` "
                    "(identifiant, fichiers, boucle, durée, BPM). OGG pour tous les navigateurs récents, MP3 en "
                    "secours ; les boucles restent exactes dans les deux formats."
                )
                with gr.Row():
                    jeu_cible = gr.Dropdown(list(export.CIBLES), value=list(export.CIBLES)[1], label="Volume cible")
                    jeu_formats = gr.CheckboxGroup([("OGG", "ogg"), ("MP3", "mp3")], value=["ogg", "mp3"],
                                                   label="Formats")
                btn_export_jeu = gr.Button("📦 Exporter le pack")
                jeu_export_msg = gr.Markdown()
                jeu_zip = gr.File(label="Archive à télécharger")

        with gr.Tab("4. Synthèse vocale"):
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

        with gr.Tab("5. Galerie") as gal_tab:
            with gr.Row():
                gal_filtre = gr.Radio(list(galerie.FILTRES), value="Tout", label="Afficher")
                gal_maj = gr.Button("🔄 Actualiser", scale=0)
            gal_liste = gr.Dropdown([], label="Création (la plus récente en premier)")
            gal_etat = gr.State()
            with gr.Row():
                with gr.Column(scale=3):
                    gal_details = gr.Markdown()
                with gr.Column(scale=2):
                    gal_version = gr.Radio([1], value=1, label="Version", visible=False)
                    gal_audio = gr.Audio(type="filepath", label="Écouter")
                    gal_modele = gr.Model3D(label="Modèle 3D", visible=False, clear_color=(0.92, 0.92, 0.92, 1.0))
                    with gr.Row():
                        gal_recreer = gr.Button("🔁 Recréer (même graine)")
                        gal_dossier = gr.Button("📂 Ouvrir le dossier")
                        gal_suppr = gr.Button("🗑️ Supprimer", variant="stop")
            gal_msg = gr.Markdown()
            with gr.Accordion("✏️ Refaire un passage (chansons et pistes de jeu)", open=False):
                gr.Markdown(
                    "Un refrain raté, une fin bizarre ? Indique le passage en secondes : seul ce passage est "
                    "réinventé (tâche *repaint* d'ACE-Step), le reste est gardé, puis la suite du traitement est "
                    "refaite (ta voix, retrait d'instruments, boucle…). Le résultat est une **nouvelle création**, "
                    "l'originale reste dans la galerie. Tu peux modifier la description ou les paroles du passage."
                )
                with gr.Row():
                    gal_debut = gr.Number(value=0, label="Début (s)")
                    gal_fin = gr.Number(value=10, label="Fin (s)")
                    gal_force = gr.Radio(list(galerie.FORCES), value=list(galerie.FORCES)[1], label="Retouche")
                gal_desc = gr.Textbox(label="Description", lines=2)
                gal_paroles = gr.Textbox(label="Paroles (chansons)", lines=6)
                gal_refaire = gr.Button("✏️ Refaire ce passage", variant="primary")

        with gr.Tab("6. Entraîner ma voix (RVC)"):
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

        with gr.Tab("7. Bruitages"):
            gr.Markdown(
                "Effets sonores pour ton jeu, à partir d'une **description** (en français ou en anglais) ou d'une "
                "**image**. « Préparer le prompt » traduit et précise ta description en anglais (Qwen3-VL) ; tu peux "
                "le retoucher avant de générer. Stable Audio Open produit 1 à 3 variantes en 44,1 kHz. "
                "ACE-Step est arrêté automatiquement pendant la génération (mémoire graphique), puis relancé à la "
                "chanson suivante."
            )
            with gr.Row():
                with gr.Column():
                    sfx_texte = gr.Textbox(label="Description du bruitage", lines=3,
                                           placeholder="une porte de château en bois qui grince puis claque")
                    sfx_exemples = gr.Dropdown([(lib, txt) for lib, txt in bruitages.EXEMPLES], label="Exemples (jeu de cartes)",
                                               allow_custom_value=True,
                                               info="Choisis un exemple : il remplit le prompt anglais directement.")
                with gr.Column():
                    sfx_image = gr.Image(type="filepath", label="Ou une image (facultatif) : les sons de la scène")
            btn_sfx_prep = gr.Button("🧠 Préparer le prompt (traduction et précision)")
            sfx_prompt = gr.Textbox(label="Prompt envoyé à Stable Audio (anglais, modifiable)", lines=2)
            with gr.Row():
                sfx_nom = gr.Textbox(label="Nom", value="bruitage")
                sfx_duree = gr.Slider(1, bruitages.DUREE_MAX, value=3, step=0.5, label="Durée (s)")
                sfx_variantes = gr.Radio([1, 2, 3], value=2, label="Variantes")
                sfx_graine = gr.Number(value=0, precision=0, label="Graine (0 = aléatoire)")
                sfx_etapes = gr.Slider(20, 200, value=100, step=10, label="Étapes (qualité / temps)")
            btn_sfx = gr.Button("🔊 Générer le bruitage", variant="primary")
            sfx_statut = gr.Markdown()
            sfx_dossier = gr.State()
            with gr.Row():
                sfx_liste = gr.Dropdown([], label="Écouter une variante")
                sfx_audio = gr.Audio(label="Bruitage", type="filepath")
            with gr.Row():
                sfx_cible = gr.Dropdown(list(export.CIBLES), value=list(export.CIBLES)[1], label="Volume de l'export")
                sfx_formats = gr.CheckboxGroup([("OGG", "ogg"), ("MP3", "mp3"), ("WAV", "wav")], value=["ogg", "mp3"],
                                               label="Formats")
                btn_sfx_export = gr.Button("📦 Exporter la variante écoutée")
            sfx_export_msg = gr.Markdown()

        with gr.Tab("8. Modèles 3D"):
            gr.Markdown(
                "Un **modèle 3D** à partir d'une **image** d'objet (personnage, arme, objet de carte…) : Hunyuan3D-2 "
                "retire le fond, sculpte la forme puis peint la texture. Une image nette, un seul objet, fond simple : "
                "c'est ce qui marche le mieux. Résultat en GLB (visionneuse ci-dessous, utilisable tel quel dans un "
                "jeu web, Blender, Unity ou Godot) et en OBJ si demandé. "
                "ACE-Step est arrêté automatiquement pendant la génération : la texture demande beaucoup de mémoire "
                "graphique."
            )
            with gr.Accordion("✍️ Pas d'image ? Décris l'objet : une image est générée d'abord (Qwen3-VL + Stable Diffusion XL)",
                              open=False):
                m3_texte = gr.Textbox(label="Description de l'objet (français ou anglais)", lines=2,
                                      placeholder="une potion de soin, fiole en verre rouge avec un bouchon de liège")
                btn_m3_prep = gr.Button("🧠 Préparer le prompt de l'image (traduction et précision)")
                m3_prompt = gr.Textbox(label="Prompt envoyé à Stable Diffusion XL (anglais, modifiable)", lines=2)
                with gr.Row():
                    m3_img_graine = gr.Number(value=0, precision=0, label="Graine de l'image (0 = aléatoire)")
                    btn_m3_image = gr.Button("🖼️ Générer l'image de l'objet", variant="primary")
                m3_img_msg = gr.Markdown()
            m3_img_graine_ok = gr.State()
            with gr.Row():
                with gr.Column():
                    m3_image = gr.Image(type="filepath", label="Image de l'objet (PNG ou JPG, ou l'image générée ci-dessus)")
                with gr.Column():
                    m3_nom = gr.Textbox(label="Nom", value="modele")
                    m3_qualite = gr.Radio(list(modele3d.QUALITES), value=modele3d.QUALITE_DEFAUT, label="Qualité")
                    m3_texture = gr.Checkbox(value=True, label="Peindre la texture (plusieurs minutes de plus)")
                    m3_graine = gr.Number(value=0, precision=0, label="Graine (0 = aléatoire)")
                    m3_formats = gr.CheckboxGroup(modele3d.FORMATS, value=["glb"], label="Formats")
            btn_m3 = gr.Button("🧊 Créer le modèle 3D", variant="primary")
            m3_statut = gr.Markdown()
            m3_dossier = gr.State()
            with gr.Row():
                m3_vue = gr.Model3D(label="Modèle 3D (glisser pour tourner, molette pour zoomer)",
                                    clear_color=(0.92, 0.92, 0.92, 1.0), scale=3)
                with gr.Column(scale=1):
                    m3_detouree = gr.Image(label="Image détourée", interactive=False)
                    m3_fichiers = gr.File(label="Fichiers produits", file_count="multiple", interactive=False)
                    btn_m3_dossier = gr.Button("📂 Ouvrir le dossier")

        with gr.Tab("9. Modèles"):
            gr.Markdown(
                "Les modèles sont volumineux (plusieurs Go au total) et ne sont téléchargés qu'une seule fois. "
                "Vérifie ici leur présence et leur emplacement, ou lance le téléchargement."
            )
            status = gr.Markdown(models_status_md())
            btn_refresh = gr.Button("🔄 Actualiser l'état")
            with gr.Accordion("🎛️ Carte graphique et serveur ACE-Step", open=False):
                gr.Markdown(
                    "Le serveur ACE-Step (génération musicale) occupe ~8 Go de mémoire graphique tant qu'il tourne. "
                    "Avec la libération automatique, il est arrêté avant les bruitages, la 3D, les illustrations, la "
                    "synthèse vocale, le nettoyage et l'entraînement RVC, puis relancé à la chanson suivante "
                    "(la première génération après une relance prend une minute de plus). "
                    f"Journal du serveur : `{serveur_acestep.journal()}`."
                )
                with gr.Row():
                    gpu_auto = gr.Checkbox(value=serveur_acestep.LIBERATION_AUTO,
                                           label="Libérer automatiquement la carte graphique")
                    gpu_etat = gr.Markdown(serveur_acestep.etat())
                with gr.Row():
                    btn_gpu_stop = gr.Button("⏹️ Arrêter ACE-Step maintenant")
                    btn_gpu_maj = gr.Button("🔄 État du serveur")
                gpu_msg = gr.Markdown()
            log = gr.Textbox(label="Journal de téléchargement", lines=14, max_lines=14, autoscroll=True, interactive=False)
            with gr.Row():
                b_ace = gr.Button("⬇️ Télécharger ACE-Step", variant="primary")
                b_sv = gr.Button("⬇️ Télécharger Seed-VC", variant="primary")
                b_dm = gr.Button("⬇️ Télécharger Demucs", variant="primary")
                b_cb = gr.Button("⬇️ Télécharger Chatterbox", variant="primary")
                b_nt = gr.Button("⬇️ Télécharger le nettoyage", variant="primary")
                b_rvc = gr.Button("⬇️ Télécharger RVC (modèles de base)", variant="primary")
            with gr.Row():
                b_dif = gr.Button("⬇️ Télécharger Qwen3-VL, Hunyuan3D et SDXL", variant="primary")
                b_sfx = gr.Button("⬇️ Télécharger Stable Audio Open (jeton requis)", variant="primary")
            with gr.Accordion("🔑 Jeton Hugging Face (nécessaire pour Stable Audio Open)", open=not diffusion.jeton_present()):
                gr.Markdown(
                    f"1. Connecte-toi sur Hugging Face et accepte la licence sur {diffusion.LICENCE_BRUITAGES} ; "
                    f"2. crée un jeton (type « Read ») sur {diffusion.JETONS} ; 3. colle-le ici. "
                    "Il est enregistré dans `StudioVoix\\hf-home\\token`, jamais ailleurs."
                )
                with gr.Row():
                    hf_jeton = gr.Textbox(label="Jeton", type="password", placeholder="hf_…")
                    btn_jeton = gr.Button("Enregistrer le jeton")
                hf_msg = gr.Markdown("✅ Un jeton est déjà enregistré." if diffusion.jeton_present() else "")
            with gr.Row():
                o_ace = gr.Button("📂 Ouvrir dossier ACE-Step")
                o_sv = gr.Button("📂 Ouvrir dossier Seed-VC")
                o_dm = gr.Button("📂 Ouvrir dossier Demucs")
                o_cb = gr.Button("📂 Ouvrir dossier Chatterbox")
                o_nt = gr.Button("📂 Ouvrir dossier nettoyage")
                o_rvc = gr.Button("📂 Ouvrir dossier RVC")
                o_dif = gr.Button("📂 Ouvrir dossier des modèles de diffusion")
                o_data = gr.Button("📂 Ouvrir mes chansons")

        def synchro_autres(evt):
            """Après une opération sur la bibliothèque : listes de voix des autres onglets à jour."""
            return evt.then(synchro_voix, voix, voix).then(synchro_voix, tts_voix, tts_voix).then(
                lambda v: gr.update(choices=list_voices(), value=[x for x in (v or []) if x in list_voices()]),
                rvc_biblio, rvc_biblio)

        synchro_autres(btn_save.click(save_voice_choix, [audio_in, audio_clean, garder, nom], [msg_voice, biblio]))
        btn_clean.click(nettoyage.nettoyer, [audio_in, niveau], [audio_clean, garder, msg_clean])
        # Nouvel échantillon : l'ancienne version nettoyée ne lui correspond plus
        audio_in.change(lambda: (None, GARDER_ORIGINAL, ""), None, [audio_clean, garder, msg_clean])
        btn_reprendre.click(reprendre_voix, biblio, [audio_in, nom, msg_biblio])
        synchro_autres(btn_ren.click(rename_voice, [biblio, nouveau_nom], [msg_biblio, biblio]))
        synchro_autres(btn_del.click(delete_voice, biblio, [msg_biblio, biblio], js=CONFIRMER_SUPPRESSION))
        biblio.change(infos_voix, biblio, [ecoute, desc_voix])
        # Voix ajoutées hors de l'application : listes à jour à chaque ouverture de la page
        synchro_autres(demo.load(synchro_voix, biblio, biblio)).then(infos_voix, biblio, [ecoute, desc_voix])
        tts_btn.click(
            synthese_puis_rvc,
            [tts_voix, tts_texte, tts_langue, tts_exag, tts_cfg, tts_temp, tts_graine, tts_rvc, tts_rvc_ton],
            [tts_sortie, tts_statut],
        )
        btn_refresh.click(models_status_md, None, status)
        gpu_auto.change(serveur_acestep.regler_liberation, gpu_auto, gpu_msg)
        btn_gpu_stop.click(serveur_acestep.arreter_depuis_interface, None, gpu_etat)
        btn_gpu_maj.click(serveur_acestep.etat, None, gpu_etat)
        for b, fn in ((b_ace, acestep.download), (b_sv, seedvc.download), (b_dm, demucs.download),
                      (b_cb, chatterbox.download), (b_nt, nettoyage.download), (b_rvc, rvc.download),
                      (b_dif, lambda: diffusion.download(["qwen", "forme3d", "texture3d", "image"])),
                      (b_sfx, lambda: diffusion.download(["bruitages"]))):
            b.click(fn, None, log).then(models_status_md, None, status)
        o_ace.click(lambda: open_folder(acestep.ckpt_dir()))
        o_sv.click(lambda: open_folder(seedvc.ckpt_dir()))
        o_dm.click(lambda: open_folder(demucs.ckpt_dir()))
        o_cb.click(lambda: open_folder(chatterbox.ckpt_dir()))
        o_nt.click(lambda: open_folder(nettoyage.ckpt_dir()))
        o_rvc.click(lambda: open_folder(cfg.RVC_DIR))
        o_dif.click(lambda: open_folder(diffusion.hf_home() / "hub", create=True))
        btn_jeton.click(diffusion.enregistrer_jeton, hf_jeton, hf_msg)
        # Bruitages
        sfx_exemples.change(lambda v: v or "", sfx_exemples, sfx_prompt)
        btn_sfx_prep.click(bruitages.preparer, [sfx_texte, sfx_image], sfx_prompt)
        btn_sfx.click(bruitages.generer,
                      [sfx_prompt, sfx_nom, sfx_duree, sfx_variantes, sfx_graine, sfx_etapes, sfx_image, sfx_texte],
                      [sfx_statut, sfx_liste, sfx_audio, sfx_dossier])
        sfx_liste.change(lambda p: p, sfx_liste, sfx_audio)
        btn_sfx_export.click(export.exporter_fichier_formats, [sfx_audio, sfx_cible, sfx_formats],
                             [sfx_export_msg])
        # Modèles 3D
        btn_m3.click(modele3d.generer,
                     [m3_image, m3_nom, m3_qualite, m3_texture, m3_graine, m3_formats, m3_prompt, m3_texte,
                      m3_img_graine_ok],
                     [m3_statut, m3_vue, m3_detouree, m3_fichiers, m3_dossier])
        btn_m3_prep.click(modele3d.preparer_prompt, m3_texte, m3_prompt)
        btn_m3_image.click(modele3d.generer_image, [m3_prompt, m3_img_graine], [m3_image, m3_img_graine_ok, m3_img_msg])
        m3_image.upload(lambda: None, None, m3_img_graine_ok)  # image importée : la graine de l'image générée ne vaut plus
        btn_m3_dossier.click(lambda d: open_folder(d) if d else None, m3_dossier)
        for champ in (rvc_fichiers, rvc_biblio):
            champ.change(rvc.analyser_enregistrements, [rvc_fichiers, rvc_biblio], rvc_controle)

        def apres_modeles(evt):
            """Listes de modèles RVC à jour partout après un entraînement ou une suppression."""
            return evt.then(rvc.tableau_modeles, None, rvc_tableau).then(rvc.maj_modeles, rvc_choix, rvc_choix).then(
                maj_conversion, conversion, conversion).then(
                lambda v: gr.update(choices=[("Aucun", None)] + rvc.choix_modeles(), value=v), tts_rvc, tts_rvc)

        apres_modeles(btn_rvc.click(rvc.entrainer, [rvc_nom, rvc_fichiers, rvc_biblio, rvc_duree, rvc_lot], rvc_msg))
        apres_modeles(btn_rvc_suppr.click(rvc.supprimer_modele, rvc_choix, [rvc_msg, rvc_tableau],
                                          js="(m) => (m && confirm('Supprimer définitivement le modèle « ' + m + ' » ?')) ? m : null"))
        apres_modeles(demo.load(lambda: None, None, None))
        o_data.click(lambda: open_folder(cfg.SONGS_DIR, create=True))
        demo.load(models_status_md, None, status)
        btn.click(
            creer_chanson,
            [voix, genre, style, instruments, ambiance, extra, voix_base, paroles, langue, duree, bpm,
             thinking, semitones, steps, gain_voix, gain_instru, mode, description, retirer, versions, graine,
             conversion],
            [final, brute, voix_conv, instru_out, statut, final_2],
        )
        versions.change(lambda v: gr.update(visible=int(v) > 1), versions, final_2)
        btn_export_chanson.click(export.exporter_fichier, [final, chanson_cible], [chanson_mp3, chanson_export_msg])
        sorties_details = [gal_details, gal_audio, gal_version, gal_desc, gal_paroles, gal_fin, gal_modele]
        for evt in (gal_filtre.change, gal_maj.click, demo.load):
            evt(galerie.maj_liste, [gal_filtre, gal_liste], gal_liste)
        gal_liste.change(galerie.details, [gal_liste], sorties_details)
        # La visionneuse 3D (Babylon.js) ne s'initialise pas si sa valeur arrive pendant que l'onglet est caché
        # (chargement de la page) et ignore une valeur identique : on la vide puis on réaffiche la création.
        gal_tab.select(lambda: None, None, gal_modele).then(galerie.details, [gal_liste, gal_version], sorties_details)
        gal_version.input(galerie.details, [gal_liste, gal_version], sorties_details)
        gal_recreer.click(galerie.recreer, [gal_liste, gal_version], [gal_msg, gal_etat]).then(
            galerie.maj_liste, [gal_filtre, gal_etat], gal_liste)
        gal_refaire.click(galerie.refaire_passage,
                          [gal_liste, gal_version, gal_debut, gal_fin, gal_desc, gal_paroles, gal_force],
                          [gal_msg, gal_etat]).then(galerie.maj_liste, [gal_filtre, gal_etat], gal_liste)
        gal_suppr.click(galerie.supprimer, gal_liste, gal_msg, js=CONFIRMER_SUPPRESSION_CREATION).then(
            galerie.maj_liste, [gal_filtre], gal_liste)
        gal_dossier.click(lambda c: open_folder(c) if c else None, gal_liste)
        btn_export_jeu.click(export.exporter_pack, [jeu_projet, jeu_cible, jeu_formats], [jeu_zip, jeu_export_msg])
        champs_jeu = [jeu_epoque, jeu_univers, jeu_situations, jeu_extra, jeu_duree]
        for champ in champs_jeu:
            champ.change(jeu.apercu, champs_jeu, jeu_apercu)
        demo.load(jeu.apercu, champs_jeu, jeu_apercu)
        btn_jeu.click(jeu.generer_bande_son,
                      [jeu_projet, jeu_epoque, jeu_univers, jeu_situations, jeu_extra, jeu_duree, jeu_thinking,
                       jeu_graine, jeu_ref, jeu_usage, jeu_fidelite],
                      [jeu_statut, jeu_liste, jeu_audio, jeu_jonction]).then(
            jeu.maj_references, [jeu_projet, jeu_ref], jeu_ref)
        jeu_projet.change(jeu.maj_references, [jeu_projet, jeu_ref], jeu_ref)
        demo.load(jeu.maj_references, [jeu_projet, jeu_ref], jeu_ref)
        jeu_liste.change(jeu.ecouter, jeu_liste, [jeu_audio, jeu_jonction])
        champs_style = [genre, style, instruments, ambiance, extra, voix_base, mode]
        for champ in champs_style:
            champ.change(apercu_description, champs_style, description)
        mode.change(
            maj_mode, mode,
            [voix, paroles, reglages, semitones, steps, gain_voix, gain_instru, final, intermediaires],
        )
    return demo
