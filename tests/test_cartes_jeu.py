"""Jeu de cartes : composition d'une carte, choix de la variante, pack complet (illustrations, cartes, 3D)."""
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import gradio as gr
import numpy as np
import pytest
from PIL import Image

from studiovoix import cartes, compo_cartes, config as cfg, export, projets
from studiovoix.outils import ecrire_creation

CIBLE = list(export.CIBLES)[1]


def _image(chemin, couleur=(200, 30, 30), taille=(1152, 864)):
    chemin.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", taille, couleur).save(chemin)
    return chemin


def test_composer_une_carte(env):
    ill = _image(env / "dragon.png")
    carte, infos = compo_cartes.composer(ill, "Dragon ancien", "7", "Créature — Dragon", "Vol.", "Il dort.", "6", "5",
                                         "Feu", "Légendaire", "© 2026")
    assert carte.size == (750, 1050) and carte.mode == "RGB"
    assert infos == {"taille_texte": 32, "lignes_effet": 1, "lignes_ambiance": 1}
    assert carte.getpixel((375, 360)) == (200, 30, 30)  # l'illustration remplit sa fenêtre
    assert carte.getpixel((5, 5)) == (255, 255, 255)  # coins arrondis
    rouge, vert, bleu = carte.getpixel((30, 600))  # cadre aux couleurs de la faction Feu
    assert rouge > vert and rouge > bleu
    # texte très long : la police rapetisse jusqu'à ce que tout tienne ; format haute définition proportionnel
    grande, infos_hd = compo_cartes.composer(ill, "X", "", "", "mot " * 300, "", "", "", "Eau", "Rare", "",
                                             "Haute définition (1500×2100)")
    assert grande.size == (1500, 2100) and infos_hd["taille_texte"] < 64


def test_composer_et_enregistrer(env):
    ill = _image(env / "potion.png", (20, 160, 60))
    msg, png, fichiers = compo_cartes.composer_et_enregistrer(
        "Mon Jeu", str(ill), "Potion de soin", "2", "Objet", "Rend 3 PV.", "", "", "", "Nature", "Commune", "",
        compo_cartes.FORMAT_DEFAUT)
    png = Path(png)
    assert png == cfg.CARDS_DIR / "Mon Jeu" / "composees" / "potion-de-soin.png" and "composée" in msg
    assert [Path(f).suffix for f in fichiers] == [".png", ".webp"]
    assert Image.open(png).info.get("dpi", (0, 0))[0] == pytest.approx(300, abs=1)
    champs = json.loads(png.with_suffix(".json").read_text(encoding="utf-8"))
    assert champs["cout"] == "2" and champs["faction"] == "Nature" and champs["illustration"] == str(ill)
    with pytest.raises(gr.Error, match="illustration"):
        compo_cartes.composer_et_enregistrer("p", None, "x", "", "", "", "", "", "", "Feu", "Rare", "", "")
    with pytest.raises(gr.Error, match="nom"):
        compo_cartes.composer_et_enregistrer("p", str(ill), " ", "", "", "", "", "", "", "Feu", "Rare", "", "")


def _illustration(projet, horodatage, nom, n=2):
    d = cfg.CARDS_DIR / projet / horodatage
    versions = []
    for i in range(1, n + 1):
        f = _image(d / f"variante_{i}.png", (40 * i, 0, 0))
        Image.open(f).save(f.with_suffix(".webp"))
        versions.append({"graine": 10 + i, "dossier": ".", "fichier": str(f)})
    ecrire_creation(d, {"type": "carte", "projet": projet, "nom": nom, "largeur": 1152, "hauteur": 864,
                        "description": f"{nom} art", "choisie": 1, "versions": versions})
    return d


def _modele_3d(horodatage, nom, projet, web=True):
    d = cfg.MODELS3D_DIR / horodatage
    d.mkdir(parents=True)
    for f in ("forme.glb", "modele.glb") + (("modele_web.glb",) if web else ()):
        (d / f).write_bytes(b"glTF" + f.encode())
    ecrire_creation(d, {"type": "3d", "nom": nom, "projet": projet, "faces_obtenues": 9000,
                        "versions": [{"graine": 1, "dossier": ".", "fichier": str(d / "modele.glb")}]})
    return d


def test_choisir_une_variante(env):
    d = _illustration("Mon Jeu", "20260101_100000", "Dragon", n=3)
    msg, fichier = cartes.choisir(str(d), SimpleNamespace(index=2))
    assert "Variante 3 gardée" in msg and fichier.endswith("variante_3.png")
    assert json.loads((d / "creation.json").read_text(encoding="utf-8"))["choisie"] == 3
    with pytest.raises(gr.Error):
        cartes.choisir(str(d), SimpleNamespace(index=7))


def test_pack_complet_du_projet(env):
    _illustration("Mon Jeu", "20260101_100000", "Dragon")
    recent = _illustration("Mon Jeu", "20260102_100000", "Dragon", n=3)  # plus récent : c'est lui qui part
    cartes.choisir(str(recent), SimpleNamespace(index=1))
    _illustration("Autre jeu", "20260103_100000", "Elfe")
    compo_cartes.composer_et_enregistrer("Mon Jeu", str(recent / "variante_2.png"), "Dragon ancien", "7", "Créature",
                                         "Vol.", "", "6", "5", "Feu", "Rare", "", compo_cartes.FORMAT_DEFAUT)
    _modele_3d("20260101_120000", "Épée", "Mon Jeu")
    _modele_3d("20260101_130000", "Bouclier", "Mon Jeu", web=False)
    _modele_3d("20260101_140000", "Arc", "Autre jeu")
    assert projets.tous() == ["Autre jeu", "Mon Jeu"]

    archive, msg = export.exporter_pack("Mon Jeu", CIBLE, ["ogg"], progress=lambda *a, **k: None)
    m = json.loads((cfg.GAMES_DIR / "Mon Jeu" / "export" / "manifest.json").read_text(encoding="utf-8"))
    assert "1 illustration(s), 1 carte(s) composée(s), 2 modèle(s) 3D" in msg
    (ill,) = m["illustrations"]
    assert ill["fichier"] == "illustrations/dragon.webp" and ill["graine"] == 12  # variante 2 du plus récent
    exporte = np.asarray(Image.open(cfg.GAMES_DIR / "Mon Jeu" / "export" / ill["fichier"]).convert("RGB"))
    r, v, b = (int(x) for x in exporte[0, 0])  # WebP : avec perte, à quelques unités près
    assert abs(r - 80) <= 4 and v <= 4 and b <= 4
    (carte,) = m["cartes"]
    assert carte == {"id": "dragon-ancien", "nom": "Dragon ancien", "fichier": "cartes/dragon-ancien.webp", "cout": "7",
                     "type": "Créature", "effet": "Vol.", "ambiance": "", "attaque": "6", "defense": "5",
                     "faction": "Feu", "rarete": "Rare"}
    bouclier, epee = m["modeles3d"]
    assert epee["fichier"] == "modeles3d/epee.glb" and epee["web"] is True  # version web choisie quand elle existe
    assert bouclier["fichier"] == "modeles3d/bouclier.glb" and bouclier["web"] is False
    assert (cfg.GAMES_DIR / "Mon Jeu" / "export" / "modeles3d" / "epee.glb").read_bytes() == b"glTFmodele_web.glb"
    noms = set(zipfile.ZipFile(archive).namelist())
    assert {"manifest.json", "illustrations/dragon.webp", "cartes/dragon-ancien.webp", "modeles3d/epee.glb"} <= noms
    assert not any("elfe" in n or "arc" in n for n in noms) and Path(archive).name == "Mon Jeu_pack.zip"
