"""Projets de jeu : un même nom réunit bande-son, bruitages, illustrations, cartes composées et modèles 3D, et le pack
du jeu (export.exporter_pack) les exporte ensemble."""
import json

from . import config as cfg


def tous():
    """Noms de projet connus, tous onglets confondus."""
    noms = set()
    for racine in (cfg.GAMES_DIR, cfg.CARDS_DIR):
        if racine.exists():
            noms |= {d.name for d in racine.iterdir() if d.is_dir()}
    for racine in (cfg.SFX_DIR, cfg.MODELS3D_DIR):
        for f in racine.glob("*/creation.json") if racine.exists() else []:
            try:
                noms.add(json.loads(f.read_text(encoding="utf-8")).get("projet") or "")
            except ValueError:
                pass
    return sorted(n for n in noms if n)


def nom(projet, defaut=None):
    """Nom de projet nettoyé (lettres, chiffres, espaces, - et _), ou defaut s'il est vide."""
    n = "".join(c for c in (projet or "").strip() if c.isalnum() or c in "-_ ").strip()
    return n or defaut
