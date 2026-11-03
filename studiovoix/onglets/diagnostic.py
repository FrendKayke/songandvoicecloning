"""Onglet « Diagnostic et carte graphique » : serveur ACE-Step, moteurs gardés en mémoire, mesure de la carte,
diagnostic rapide et essai complet des moteurs."""
import gradio as gr

from .. import diagnostic, residents, serveur_acestep
from .commun import espace_de_noms


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    gr.Markdown("**Premier réflexe en cas de problème** : « Diagnostic rapide », puis envoie le fichier `rapport.txt`.")
    with gr.Accordion("🎛️ Carte graphique : performances et serveur ACE-Step", open=True):
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
        with gr.Row():
            ace_lm = gr.Dropdown(list(serveur_acestep.MODELES_LM), value=serveur_acestep.libelle_modele_lm(),
                                 label="Modèle de langage d'ACE-Step (structure et cohérence des chansons)")
        gpu_msg = gr.Markdown()
        gr.Markdown(
            "**Modèles gardés en mémoire** : la diffusion (images, 3D, vidéos, photos, bruitages), la synthèse "
            "vocale et le nettoyage restent ouverts entre deux générations : la suivante démarre sans relire ses "
            "modèles sur le disque, la carte calcule au lieu d'attendre. Un seul moteur ouvert à la fois ; il "
            f"est fermé avant une chanson (ACE-Step) et après {residents.GARDER_MIN:g} minutes sans génération."
        )
        with gr.Row():
            res_actif = gr.Checkbox(value=residents.ACTIF, label="Garder les modèles en mémoire entre deux générations",
                                    interactive=residents.GARDER_MIN > 0)
            res_etat = gr.Markdown(residents.etat())
        with gr.Row():
            btn_res_fermer = gr.Button("⏏️ Fermer le moteur ouvert (rendre la mémoire)")
            btn_res_maj = gr.Button("🔄 État des moteurs")
        with gr.Row():
            btn_gpu_mesure = gr.Button("📈 Mesurer la carte graphique pendant 1 minute")
            gr.Markdown("Lance une génération dans un autre onglet, puis clique ici : l'utilisation réelle de la "
                        "carte s'affiche en direct (mesures de nvidia-smi).")
        gpu_mesure = gr.Markdown()
    with gr.Accordion("🩺 Diagnostic et essai complet des moteurs", open=True):
        gr.Markdown(
            "**Diagnostic rapide** (moins d'une minute) : carte graphique, espace disque, chaque moteur "
            "(PyTorch, CUDA), modèles manquants. **Essai complet** (10 à 20 minutes) : une génération courte "
            "par moteur, avec sa durée et la mémoire graphique utilisée. Les fichiers d'essai vont dans "
            "`data\\diagnostic`, pas dans la galerie. Envoie le fichier `rapport.txt` pour un dépannage."
        )
        with gr.Row():
            btn_diag = gr.Button("🩺 Diagnostic rapide")
            btn_essai = gr.Button("🧪 Essai complet des moteurs")
        diag_rapport = gr.Markdown()
        diag_fichier = gr.File(label="Rapport à envoyer", interactive=False)
    return espace_de_noms(locals())


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    c.gpu_auto.change(serveur_acestep.regler_liberation, c.gpu_auto, c.gpu_msg)
    c.btn_diag.click(diagnostic.rapide, None, [c.diag_rapport, c.diag_fichier])
    c.btn_essai.click(diagnostic.complet, None, [c.diag_rapport, c.diag_fichier])
    c.btn_gpu_stop.click(serveur_acestep.arreter_depuis_interface, None, c.gpu_etat)
    c.btn_gpu_maj.click(serveur_acestep.etat, None, c.gpu_etat)
    c.btn_gpu_mesure.click(diagnostic.surveiller_gpu, None, c.gpu_mesure)
    c.ace_lm.change(serveur_acestep.regler_modele_lm, c.ace_lm, c.gpu_msg).then(serveur_acestep.etat, None, c.gpu_etat)
    c.res_actif.change(residents.regler, c.res_actif, c.res_etat)
    c.btn_res_fermer.click(residents.fermer_depuis_interface, None, c.res_etat)
    c.btn_res_maj.click(residents.etat, None, c.res_etat)
