"""Musiques en série : une musique par ligne d'un tableau ou d'une liste, dans le même style (faux serveur ACE-Step)."""
import csv
import json
import zipfile
from pathlib import Path

import gradio as gr
import pytest
import soundfile as sf

from conftest import no_progress, write_tone
from studiovoix import acestep, diffusion, serie_musique
from studiovoix import config as cfg


def test_durees_et_noms():
    assert [serie_musique.duree_de(t, 60) for t in ("90", "90 s", "1:30", "1 min 30", "2min", "", "deux", "5",
                                                     "9999", "45,5")] == [90, 90, 90, 90, 120, 60, 60, 10, 600, 46]
    assert serie_musique.nom_audio("Boss final.mp3") == "Boss final" and serie_musique.nom_audio("menu") == "menu"
    assert serie_musique.nom_audio("a/b:c") == "a_b_c" and serie_musique.nom_audio("") == ""
    commun = serie_musique.style_commun(["Orchestral"], [], [], [], [], "16-bit synths")
    assert commun.endswith("16-bit synths")
    d = serie_musique.description_finale("boss battle, epic", commun, True)
    assert d.startswith("boss battle, epic") and d.endswith("instrumental")
    assert serie_musique.paroles_de({"paroles": "Refrain\nla la"}, serie_musique.MODE_CHANT).startswith("[Chorus]")
    assert serie_musique.paroles_de({"paroles": "la la"}, serie_musique.MODE_INSTRU) == "[Instrumental]"
    assert serie_musique.paroles_de({"paroles": ""}, serie_musique.MODE_CHANT) == "[Instrumental]"


def test_musiques_en_serie_depuis_un_tableau(env, fake_acestep, monkeypatch):
    """Colonnes devinées, une tâche ACE-Step par ligne (description traduite + style commun, graines qui se
    suivent, musique de référence envoyée), noms exacts, formats au même volume, ligne en échec puis reprise."""
    srv = fake_acestep()
    c = env / "quiz.csv"
    c.write_text("Nom;Description;Durée;Paroles\nboss.mp3;combat contre le boss final;1:30;\n"
                 "Menu;menu calme ECHEC;;\nmenu;victoire;20;On a gagné\n", encoding="utf-8")
    feuille, col_d, col_n, col_p, col_t, apercu_, msg = serie_musique.analyser(str(c))
    assert (col_d["value"], col_n["value"], col_p["value"], col_t["value"]) == ("B", "A", "D", "C")
    lot = serie_musique.entrees(str(c), "CSV", True, "B", "A", "D", "C", duree_defaut=45)
    assert [(e["base"], e["duree"]) for e in lot] == [("boss", 90), ("Menu", 45), ("menu_2", 20)]
    assert lot[2]["doublon"]  # même nom sans tenir compte de la casse (Windows)
    tableau, info = serie_musique.apercu(str(c), "CSV", True, "B", "A", "D", "C", "", ["Orchestral"], [], [], [],
                                         [], "", "Automatique", serie_musique.MODE_CHANT, 45)
    assert [ligne[3] for ligne in tableau["data"]] == ["instrumentale", "instrumentale", "chantée"]
    assert "3 musique(s)" in info and "1 nom(s) de fichier en double" in info

    traductions = []

    def faux_decrire(mode, texte, nombre=None, **_):
        traductions.append((mode, texte, nombre))
        return "1. final boss battle\n2. calm menu ECHEC\n3. victory fanfare"

    monkeypatch.setattr(diffusion, "present", lambda nom: True)
    monkeypatch.setattr(diffusion, "decrire", faux_decrire)
    vrai_generer = acestep.generer

    def generer(params, dests, *a, **k):  # « ECHEC » dans la description : erreur du serveur simulée
        if "ECHEC" in params["prompt"]:
            raise gr.Error("serveur ACE-Step indisponible")
        return vrai_generer(params, dests, *a, **k)

    monkeypatch.setattr(serie_musique.acestep, "generer", generer)
    reference = write_tone(env / "theme.wav", seconds=5.0)
    msg, choix, premier, archive, dossier = serie_musique.generer(
        str(c), "CSV", True, "B", "A", "D", "C", "", ["Orchestral"], [], [], [], [], "16-bit synths", "Automatique",
        serie_musique.MODE_CHANT, "Français", 45, str(reference), True, True, 1, 7, ["wav", "mp3"],
        next(iter(serie_musique.export.CIBLES)), "Quiz", 0, progress=no_progress)
    assert traductions[0][0] == "musiques" and traductions[0][2] == 3 and "1. combat contre le boss final" in \
        traductions[0][1]
    assert len(srv.payloads) == 2 and "2/3" in msg and "⚠️ musique 2 (Menu)" in msg and "Reprendre" in msg
    p1, p3 = srv.payloads
    assert p1["prompt"].startswith("final boss battle, ") and p1["prompt"].endswith("16-bit synths, instrumental")
    assert p1["lyrics"] == "[Instrumental]" and float(p1["audio_duration"]) == 90 and p1["seed"] == "7"
    assert p3["prompt"].startswith("victory fanfare") and "instrumental" not in p3["prompt"]
    assert p3["lyrics"] == "On a gagné" and p3["vocal_language"] == "fr" and p3["seed"] == "9"
    assert srv.fichiers[0]["reference_audio"][0].startswith("reference")
    d = Path(dossier)
    assert d.name.endswith("_Quiz") and (d / "boss.wav").exists() and (d / "boss.mp3").exists()
    assert (d / "brut" / "boss.wav").exists() and not (d / "Menu.wav").exists()
    assert sf.info(str(d / "boss.wav")).samplerate == cfg.SR
    assert [lib for lib, _ in choix["choices"]] == ["1 · boss", "3 · menu_2"] and premier.endswith("boss.wav")
    with zipfile.ZipFile(archive) as z:
        assert {"boss.wav", "boss.mp3", "menu_2.mp3", "lot.csv"} <= set(z.namelist())
    lignes = list(csv.reader((d / "lot.csv").read_text(encoding="utf-8-sig").splitlines(), delimiter=";"))
    assert lignes[2][2] == "Menu" and lignes[2][7] == "non" and lignes[1][7] == "oui"
    # la cause de l'échec disparaît : seule la musique manquante repart
    lot_json = d / "lot.json"
    lot_json.write_text(lot_json.read_text(encoding="utf-8").replace(" ECHEC", ""), encoding="utf-8")
    n = len(srv.payloads)
    msg, choix, *_ = serie_musique.reprendre(str(d), progress=no_progress)
    assert len(srv.payloads) == n + 1 and srv.payloads[-1]["prompt"].startswith("calm menu") and "3/3" in msg
    assert (d / "Menu.mp3").exists()
    infos = json.loads(lot_json.read_text(encoding="utf-8"))
    assert infos["reference"] == "reference.wav" and infos["musiques"][0]["description"] == "combat contre le boss final"


def test_liste_collee_sans_traduction_et_deux_versions(env, fake_acestep, monkeypatch):
    srv = fake_acestep()
    monkeypatch.setattr(diffusion, "present", lambda nom: False)  # Qwen absent : descriptions telles quelles
    msg, choix, premier, archive, dossier = serie_musique.generer(
        None, None, True, None, None, None, None, "medieval tavern\n\nspace station\n", [], [], [], [], [], "", "Automatique",
        serie_musique.MODE_INSTRU, "Français", 30, None, True, False, 2, 0, ["ogg"], "", "", 0, progress=no_progress)
    # sans musique de référence, envoi en JSON (valeurs typées)
    assert "telles quelles" in msg and len(srv.payloads) == 2 and srv.payloads[0]["batch_size"] == 2
    assert srv.payloads[0]["prompt"].startswith("medieval") and srv.payloads[1]["thinking"] is False
    d = Path(dossier)
    assert (d / "001_medieval_tavern.ogg").exists() and (d / "001_medieval_tavern_v2.ogg").exists()
    assert len(choix["choices"]) == 4
    with pytest.raises(gr.Error, match="Aucune description"):
        serie_musique.entrees(liste=" \n ")
    with pytest.raises(gr.Error, match="Aucun lot"):
        serie_musique.reprendre(None, progress=no_progress)
