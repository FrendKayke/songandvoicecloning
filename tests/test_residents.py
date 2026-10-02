"""Moteurs résidents : le moteur reste ouvert entre deux tâches et garde ses modèles (residents.py, moteurs/resident.py,
_garder de moteurs/diffusion.py)."""
import importlib.util
import json
import shutil
import sys
import textwrap
import time
from types import SimpleNamespace

import gradio as gr
import pytest

from studiovoix import config as cfg, outils, residents, serveur_acestep

_spec = importlib.util.spec_from_file_location("diffusion_moteur_res", cfg.MOTEURS_DIR / "diffusion.py")
moteur = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(moteur)  # importable sans torch

FAUX = textwrap.dedent('''
    import json, os, sys
    CHARGES = []  # « modèle » gardé d'une tâche à l'autre

    def principal(argv):
        t = json.load(open(argv[1], encoding="utf-8"))
        deja = bool(CHARGES)
        if not deja:
            CHARGES.append(os.getpid())
        if t.get("erreur"):
            print("ERREUR : mémoire de la carte graphique insuffisante.", flush=True)
            sys.exit(3)
        if t.get("plante"):
            raise RuntimeError("boum")
        if t.get("meurt"):
            os._exit(9)
        print("PROGRESSION 1/1 génération", flush=True)
        print("RESULTAT " + json.dumps({"pid": os.getpid(), "deja": deja, "texte": t.get("texte"),
                                        "cwd": os.getcwd(), "var": os.environ.get("FAUSSE_VAR")}), flush=True)

    if __name__ == "__main__":
        import resident
        delai = resident.demande(sys.argv[1:])
        if delai is None:
            principal(sys.argv[1:])
        else:
            resident.servir(principal, delai)
''')


@pytest.fixture
def faux(env, monkeypatch, tmp_path):
    moteurs = tmp_path / "moteurs"
    moteurs.mkdir()
    shutil.copy(cfg.MOTEURS_DIR / "resident.py", moteurs / "resident.py")
    (moteurs / "faux.py").write_text(FAUX, encoding="utf-8")
    monkeypatch.setattr(residents, "ACTIF", True)
    monkeypatch.setattr(residents, "GARDER_MIN", 15.0)
    d = tmp_path / "tâches é"  # accents : la tâche passe en JSON ASCII
    d.mkdir()
    return SimpleNamespace(script=moteurs / "faux.py", dossier=d, n=0)


def _tache(f, **t):
    f.n += 1
    chemin = f.dossier / f"tache_{f.n}.json"
    chemin.write_text(json.dumps(t, ensure_ascii=False), encoding="utf-8")
    return chemin


def _lancer(f, resident="Faux", env=None, suivi=None, **t):
    lignes = outils.lancer_moteur([sys.executable, str(f.script), "action", str(_tache(f, **t))], f.dossier,
                                  env or {"FAUSSE_VAR": "1"}, resident, suivi, resident=resident)
    return json.loads(next(l_ for l_ in reversed(lignes) if l_.startswith("RESULTAT "))[9:])


def test_modeles_gardes_entre_deux_taches(faux):
    progression = []
    r1 = _lancer(faux, texte="élan ☀", suivi=lambda i, n: progression.append((i, n)))
    r2 = _lancer(faux)
    assert r1["pid"] == r2["pid"] and not r1["deja"] and r2["deja"]  # même processus, modèle gardé
    assert r1["texte"] == "élan ☀" and r1["var"] == "1" and progression == [(1, 1)]
    assert r1["cwd"] == str(faux.dossier)
    assert "Faux" in residents.etat() and "2 génération(s)" in residents.etat()
    # environnement différent (par exemple modèles téléchargés entre-temps) : moteur relancé
    r3 = _lancer(faux, env={"FAUSSE_VAR": "2"})
    assert r3["pid"] != r1["pid"] and r3["var"] == "2"


def test_apres_une_erreur_le_moteur_repart_neuf(faux):
    pid = _lancer(faux)["pid"]
    with pytest.raises(gr.Error, match="Faux : mémoire de la carte graphique"):
        _lancer(faux, erreur=True)
    r = _lancer(faux)
    assert r["pid"] != pid and not r["deja"]
    with pytest.raises(gr.Error, match="RuntimeError: boum"):  # erreur imprévue : la trace est montrée
        _lancer(faux, plante=True)
    with pytest.raises(gr.Error, match="Faux a échoué"):  # moteur mort pendant la tâche
        _lancer(faux, meurt=True)
    assert not _lancer(faux)["deja"]


def test_un_seul_moteur_resident_a_la_fois(faux):
    _lancer(faux, "Faux")
    premier = residents._moteurs["Faux"]
    _lancer(faux, "Autre")
    assert list(residents._moteurs) == ["Autre"] and not premier.vivant()
    # un moteur lancé à part (Demucs, Seed-VC, RVC…) ferme d'abord les moteurs résidents
    autre = residents._moteurs["Autre"]
    _lancer(faux, resident=None)
    assert not autre.vivant() and not residents._moteurs


def test_ace_step_ferme_les_moteurs_residents(faux, monkeypatch):
    _lancer(faux)
    m = residents._moteurs["Faux"]
    monkeypatch.setattr(serveur_acestep, "repond", lambda timeout=3: True)  # serveur déjà prêt
    serveur_acestep.assurer()
    assert not m.vivant()


def test_fermeture_apres_inactivite(faux, monkeypatch):
    monkeypatch.setattr(residents, "GARDER_MIN", 1 / 60)  # 1 s
    pid = _lancer(faux)["pid"]
    m = residents._moteurs["Faux"]
    m.proc.wait(timeout=20)  # le moteur s'arrête de lui-même
    r = _lancer(faux)
    assert r["pid"] != pid and not r["deja"]


def test_reglage(faux):
    _lancer(faux)
    m = residents._moteurs["Faux"]
    assert "désactivé" in residents.regler(False) and not m.vivant()
    assert not _lancer(faux)["deja"]  # processus neuf à chaque tâche
    assert not residents._moteurs
    residents.regler(True)
    _lancer(faux)
    assert "Aucun moteur" in residents.fermer_depuis_interface()


@pytest.mark.parametrize("script", ["diffusion.py", "chatterbox_tts.py", "nettoyage_voix.py"])
def test_vrais_moteurs_en_mode_resident(env, monkeypatch, script):
    """Les vrais scripts (sans torch ici) démarrent en mode résident et répondent à une tâche mal formée."""
    monkeypatch.setattr(residents, "ACTIF", True)
    lignes = list(residents.lancer(script, sys.executable, cfg.MOTEURS_DIR / script, ["a", "b", "c"], env, {}))
    assert any(l_.startswith("ERREUR : usage") for l_ in lignes)
    assert script not in residents._moteurs  # code 2 : moteur fermé


def test_garder_les_modeles_dans_le_moteur(monkeypatch):
    monkeypatch.setenv("STUDIOVOIX_MEMOIRE_MODELES", "10")
    monkeypatch.setattr(moteur, "_CHARGES", {})
    journal = []

    def garder(nom, go):
        return moteur._garder((nom,), lambda: journal.append(f"charge {nom}") or nom, go,
                              vers_cpu=lambda o: journal.append(f"cpu {o}"), vers_gpu=lambda o: journal.append(f"gpu {o}"))

    assert garder("A", 4) == "A" and journal == ["charge A"]
    journal.clear()
    garder("B", 4)
    assert journal == ["cpu A", "charge B"]  # A quitte la carte, reste en mémoire vive
    journal.clear()
    assert garder("A", 4) == "A" and journal == ["cpu B", "gpu A"]  # pas rechargé
    journal.clear()
    garder("C", 5)  # 4 + 4 + 5 > 10 : le moins récent (B) est libéré
    assert list(moteur._CHARGES) == [("A",), ("C",)] and journal[-1] == "charge C"
    moteur._faire_place(5, sauf=("C",))  # A part d'abord ; C (5 + 5 = 10) tient encore
    assert list(moteur._CHARGES) == [("C",)]
    moteur._oublier(("C",))
    assert not moteur._CHARGES


def test_hors_ligne_reellement_applique(monkeypatch):
    """huggingface_hub lit HF_HUB_OFFLINE à son import : la variable d'environnement posée ensuite était sans
    effet. Le moteur règle directement la valeur lue par ses requêtes, dans les deux sens."""
    import huggingface_hub
    from huggingface_hub import constants

    depots = SimpleNamespace(repos=[SimpleNamespace(repo_id="a/b", revisions=[1])])
    monkeypatch.setattr(huggingface_hub, "scan_cache_dir", lambda: depots)
    monkeypatch.setattr(moteur, "_HORS_LIGNE_IMPOSE", False)
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False)
    moteur._hors_ligne_si_present("a/b")
    assert constants.HF_HUB_OFFLINE is True
    moteur._hors_ligne_si_present("a/b", "c/d")  # un dépôt absent : en ligne pour pouvoir le lire
    assert constants.HF_HUB_OFFLINE is False


def test_attente_de_la_tache_suivante_sans_bloquer(faux):
    """Deux tâches qui arrivent ensemble passent l'une après l'autre (verrou)."""
    import threading

    resultats = []
    fils = [threading.Thread(target=lambda: resultats.append(_lancer(faux))) for _ in range(3)]
    t0 = time.time()
    for f in fils:
        f.start()
    for f in fils:
        f.join(timeout=60)
    assert len(resultats) == 3 and len({r["pid"] for r in resultats}) == 1 and time.time() - t0 < 60
