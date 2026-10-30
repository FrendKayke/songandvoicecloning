#!/bin/bash
# Faux uv.exe / python.exe pour simuler l'installateur sous Linux : journalise l'appel dans $JOURNAL ;
# « uv venv <dossier> » crée un faux python dans <dossier>/Scripts ; ECHEC=<texte> fait échouer l'appel qui le contient ;
# « diffusion.py manquants … » répond « MANQUANTS: $FAUX_MANQUANTS » (modèles absents du cache simulé).
echo "$(basename "$0") $* [cwd=$(pwd)]" >> "$JOURNAL"
if [[ "$(basename "$0")" == "uv.exe" && "$1" == "venv" ]]; then
  d="${@: -1}"; mkdir -p "$d/Scripts"; cp "$0" "$d/Scripts/python.exe"
fi
if [[ -n "$ECHEC" && "$*" == *"$ECHEC"* ]]; then echo "échec simulé" >&2; exit 7; fi
if [[ "$*" == *"diffusion.py manquants"* ]]; then echo "MANQUANTS: $FAUX_MANQUANTS"; fi
echo "sortie normale"; echo "progression sur stderr" >&2
exit 0
