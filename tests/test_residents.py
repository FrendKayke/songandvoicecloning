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


def test_depot_incomplet_hors_ligne_puis_en_ligne(monkeypatch, tmp_path):
    """Dépôt présent mais incomplet (téléchargement interrompu, constaté avec Qwen3-VL-4B) : hors ligne, le fichier
    manquant est introuvable ; l'action est refaite une fois en ligne."""
    import huggingface_hub
    from huggingface_hub import constants

    monkeypatch.setattr(huggingface_hub, "scan_cache_dir",
                        lambda: SimpleNamespace(repos=[SimpleNamespace(repo_id="a/b", revisions=[1])]))
    monkeypatch.setattr(moteur, "_HORS_LIGNE_IMPOSE", False)
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False)
    essais = []

    def action(chemin):
        moteur._hors_ligne_si_present("a/b")
        essais.append(constants.HF_HUB_OFFLINE)
        if constants.HF_HUB_OFFLINE:
            raise OSError("Can't load image processor for 'a/b'")

    monkeypatch.setitem(moteur.ACTIONS, "decrire", action)
    moteur.principal(["decrire", "t.json"])
    assert essais == [True, False] and moteur._EN_LIGNE_FORCE is False
    # autre erreur, ou réseau déjà permis : pas de second essai
    monkeypatch.setitem(moteur.ACTIONS, "decrire", lambda c: (_ for _ in ()).throw(OSError("image illisible")))
    monkeypatch.setattr(huggingface_hub, "scan_cache_dir", lambda: SimpleNamespace(repos=[]))
    with pytest.raises(OSError, match="illisible"):
        moteur.principal(["decrire", "t.json"])


def test_demucs_et_seed_vc_gardes_d_une_chanson_a_l_autre(fake_engines, monkeypatch):
    """Demucs et Seed-VC dans un même moteur résident (moteurs/separation.py) : modèles chargés une fois ; avant
    ACE-Step ils sont rangés en mémoire vive (« --ranger »), pas fermés."""
    from studiovoix import demucs, seedvc
    from conftest import write_tone

    monkeypatch.setattr(residents, "ACTIF", True)
    monkeypatch.setattr(residents, "GARDER_MIN", 15.0)
    morceau = fake_engines / "chanson.wav"
    write_tone(morceau, seconds=2)
    write_tone(cfg.VOICES_DIR / "moi.wav", seconds=3)
    journal = []
    monkeypatch.setattr(residents._Moteur, "executer", _espion(residents._Moteur.executer, journal))
    for i in (1, 2):
        d = fake_engines / f"chanson{i}"
        d.mkdir()
        voix, instru = demucs.separate_vocals(morceau, d)
        assert voix.name == "vocals.wav" and instru.name == "no_vocals.wav" and "htdemucs_ft" in str(voix)
        assert seedvc.convert_voice(voix, cfg.VOICES_DIR / "moi.wav", 0, 10, d).exists()
        if i == 1:
            m = residents._moteurs["Séparation"]
            monkeypatch.setattr(serveur_acestep, "repond", lambda timeout=3: True)
            serveur_acestep.assurer()  # chanson suivante : ACE-Step a besoin de la carte
            assert m.vivant() and residents._moteurs["Séparation"] is m  # rangé, pas fermé
    texte = "\n".join(journal)
    assert texte.count("chargement de htdemucs_ft") == 1 and texte.count("chargement des modèles") == 1
    assert "Seed-VC : modèles déjà en mémoire" in texte and "Demucs : htdemucs_ft déjà en mémoire" in texte
    assert "Modèles de séparation et de conversion en mémoire vive" in texte
    with pytest.raises(gr.Error, match="Demucs a échoué"):  # erreur : le moteur repart neuf
        demucs.separate_vocals(fake_engines / "absent.wav", fake_engines / "chanson1")
    assert "Séparation" not in residents._moteurs


def _espion(executer, journal):
    def espion(self, argv):
        for ligne in executer(self, argv):
            journal.append(ligne)
            yield ligne
    return espion
