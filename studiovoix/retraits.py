"""Moteurs et modèles retirés pour gagner de la place (Outils → Espace disque).

La liste est dans <StudioVoix>/moteurs-retires.txt, une clé par ligne : l'installateur la lit et saute ces étapes
(sinon il réinstallerait aussitôt ce qui vient d'être supprimé). Clés : chatterbox, nettoyage, rvc, diffusion et
diffusion:<modèle> (qwen, bruitages, zimage, personnages, forme3d, texture3d).
"""
from . import config as cfg

CONSEIL_REINSTALLER = ("il a été retiré pour gagner de la place. Pour le réinstaller : Outils → Modèles → "
                       "« Espace disque » → « Réinstaller », puis lance METTRE_A_JOUR.bat (ou INSTALLER.bat).")


def fichier():
    return cfg.ENG_DIR / "moteurs-retires.txt"


def liste():
    try:
        return [l_.strip() for l_ in fichier().read_text(encoding="utf-8").splitlines() if l_.strip()]
    except OSError:
        return []


def retire(cle):
    """Vrai si la clé (ou, pour diffusion:<modèle>, tout le moteur de diffusion) est retirée."""
    cles = liste()
    return cle in cles or (cle.startswith("diffusion:") and "diffusion" in cles)


def ajouter(cle):
    if cle not in liste():
        cfg.ENG_DIR.mkdir(parents=True, exist_ok=True)
        fichier().write_text("\n".join(liste() + [cle]) + "\n", encoding="utf-8")


def enlever(cle):
    reste = [c for c in liste() if c != cle and not (cle == "diffusion" and c.startswith("diffusion:"))]
    if reste:
        fichier().write_text("\n".join(reste) + "\n", encoding="utf-8")
    elif fichier().exists():
        fichier().unlink()


def conseil(cle, defaut="Relance INSTALLER.bat : seules les étapes manquantes seront faites."):
    """Conseil à afficher quand un moteur manque : réinstaller s'il a été retiré, sinon relancer l'installateur."""
    return CONSEIL_REINSTALLER[0].upper() + CONSEIL_REINSTALLER[1:] if retire(cle) else defaut
