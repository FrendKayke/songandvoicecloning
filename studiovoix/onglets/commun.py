"""Éléments partagés par les onglets de l'interface."""
from types import SimpleNamespace

import gradio as gr

from ..styles import LISTES
from ..voix import list_voices

# Fenêtre de confirmation du navigateur avant suppression (annuler → None → rien n'est supprimé)
CONFIRMER_SUPPRESSION_CREATION = "(c) => (c && confirm('Supprimer définitivement cette création et tous ses fichiers ?')) ? c : null"
CONFIRMER_SUPPRESSION = "(v) => (v && confirm('Supprimer définitivement la voix « ' + v + ' » ?')) ? v : null"


def espace_de_noms(variables):
    """Composants d'un onglet (les variables locales de sa fonction construire), accessibles par c.nom."""
    return SimpleNamespace(**{k: v for k, v in variables.items() if not k.startswith("_")})


def synchro_voix(courante):
    """Met à jour une liste de voix d'un autre onglet en gardant la sélection si elle existe encore."""
    voices = list_voices()
    return gr.update(choices=voices, value=courante if courante in voices else (voices[0] if voices else None))


def liste_style(cle, label, info=None):
    """Liste déroulante à choix multiples (libellés français, termes anglais) qui accepte aussi la saisie libre."""
    return gr.Dropdown(
        LISTES[cle], value=[], multiselect=True, allow_custom_value=True, label=label,
        info=info or "Choisis dans la liste, ou tape ton propre terme (en anglais de préférence) puis Entrée.",
    )
