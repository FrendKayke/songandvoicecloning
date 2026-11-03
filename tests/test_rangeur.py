"""Rangeur : créations copiées en ordre (un dossier par type, noms lisibles, liste.csv, .zip)."""
import csv
import zipfile
from pathlib import Path

import pytest

from studiovoix import config as cfg
from studiovoix import rangeur
from studiovoix.outils import ecrire_creation


def _creation(racine, nom_dossier, infos, fichiers):
    d = Path(racine) / nom_dossier
    d.mkdir(parents=True)
    versions = []
    for f in fichiers:
        (d / f).write_bytes(f.encode())
        versions.append({"graine": 1, "dossier": ".", "fichier": str(d / f)})
    ecrire_creation(d, {**infos, "versions": versions})
    return d


@pytest.fixture
def creations(env):
    _creation(cfg.IMAGES_DIR, "20261101_100000", {"type": "image", "nom": "Château", "description_fr": "un château",
                                                   "choisie": 2}, ["variante_1.png", "variante_2.png"])
    _creation(cfg.IMAGES_DIR, "20261102_100000", {"type": "image", "nom": "Château", "description": "x"},
              ["variante_1.png"])
    _creation(cfg.SFX_DIR, "20261103_090000", {"type": "bruitage", "nom": "porte", "projet": "Donjon"},
              ["variante_1.wav", "variante_2.wav"])
    d = _creation(cfg.MODELS3D_DIR, "20261103_120000", {"type": "3d", "nom": "Potion", "projet": "donjon"},
                  ["modele.glb"])
    (d / "modele.obj").write_text("v 0 0 0")
    _creation(cfg.GAMES_DIR / "Donjon" / "combat", "20261020_080000",
              {"type": "jeu", "libelle": "Combat", "projet": None}, ["piste.wav"])
    _creation(cfg.SONGS_DIR, "20261015_200000", {"type": "chanson", "description": "pop ballad, piano, emotional "
                                                  "female vocals, slow tempo"}, ["chanson_finale.wav"])


def test_selection_par_projet_type_et_dates(creations):
    noms = lambda r: sorted(i.get("nom") or i.get("libelle") or i["type"] for _, i, _ in r)  # noqa: E731
    # projet sans tenir compte de la casse ; la bande-son est retrouvée par son dossier jeux/<projet>/
    assert noms(rangeur.selection("Donjon")) == ["Combat", "Potion", "porte"]
    assert noms(rangeur.selection(rangeur.TOUS, ["Images"])) == ["Château", "Château"]
    assert len(rangeur.selection(rangeur.TOUS, None, "01/11/2026", "2026-11-02")) == 2
    assert len(rangeur.selection(rangeur.TOUS)) == 6
    assert "Images 2" in rangeur.apercu(rangeur.TOUS, ["Images"])
    with pytest.raises(Exception, match="mal lue"):
        rangeur.selection(rangeur.TOUS, None, "hier")
    with pytest.raises(Exception, match="avant la date de début"):
        rangeur.selection(rangeur.TOUS, None, "03/11/2026", "01/11/26")


def test_ranger_tout(creations):
    msg, fichiers, dossier = rangeur.ranger(nom="sauvegarde", progress=lambda *a, **k: None)
    d = Path(dossier)
    assert d.parent == cfg.RANGEMENTS_DIR
    copies = sorted(p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file() and p.suffix != ".zip")
    assert copies == sorted([
        "Images/2026-11-01_Château_1.png", "Images/2026-11-01_Château_2.png", "Images/2026-11-02_Château.png",
        "Bruitages/2026-11-03_porte_1.wav", "Bruitages/2026-11-03_porte_2.wav",
        "Modèles 3D/2026-11-03_Potion.glb", "Modèles 3D/2026-11-03_Potion_modele.obj",
        "Bande-son/2026-10-20_Combat.wav",
        "Chansons/2026-10-15_pop ballad, piano, emotional female vocals.wav",
        "liste.csv"])
    assert (d / "Images/2026-11-01_Château_2.png").read_bytes() == b"variante_2.png"
    with open(d / "liste.csv", encoding="utf-8-sig") as f:
        lignes = list(csv.reader(f, delimiter=";"))
    assert lignes[0][0] == "Date" and len(lignes) == 10
    potion = [l_ for l_ in lignes if l_[2] == "Potion"]
    assert potion[0][:2] == ["03/11/2026", "Modèles 3D"] and potion[0][4] == "donjon"
    z = Path(fichiers[0])
    assert z.suffix == ".zip"
    with zipfile.ZipFile(z) as zf:
        assert "Images/2026-11-01_Château_2.png" in zf.namelist() and "liste.csv" in zf.namelist()
    assert "6 création(s), 9 fichier(s)" in msg


def test_seulement_la_variante_choisie(creations):
    _, _, dossier = rangeur.ranger(rangeur.TOUS, ["Images", "Bruitages"], toutes=False, archive=False,
                                   progress=lambda *a, **k: None)
    d = Path(dossier)
    images = sorted(p.name for p in (d / "Images").iterdir())
    assert images == ["2026-11-01_Château.png", "2026-11-02_Château.png"]
    assert (d / "Images/2026-11-01_Château.png").read_bytes() == b"variante_2.png"  # la variante choisie
    assert [p.name for p in (d / "Bruitages").iterdir()] == ["2026-11-03_porte.wav"]
    assert not list(d.glob("*.zip"))


def test_rien_a_ranger(env):
    with pytest.raises(Exception, match="Aucune création"):
        rangeur.ranger(progress=lambda *a, **k: None)
