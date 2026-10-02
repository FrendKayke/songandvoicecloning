"""Diagnostic : rapport rapide (environnements simulés) et essai complet (moteurs remplacés), sans carte graphique."""
import sys
import textwrap
from pathlib import Path

import gradio as gr

from conftest import write_tone
from studiovoix import config as cfg
from studiovoix import diagnostic


def _faux_python(dossier, cuda=True, casse=False):
    """Script qui imite « python -c SONDE » d'un environnement (Windows : python.exe = ce Python + script)."""
    script = dossier / "faux_python.py"
    script.write_text(textwrap.dedent(f"""
        import sys
        if {casse}: sys.stderr.write("ModuleNotFoundError: No module named 'torch'\\n"); sys.exit(1)
        print("SONDE 3.12.4 2.6.0+cu126 {cuda} NVIDIA GeForce RTX 4070")
    """), encoding="utf-8")
    return script


def test_sonde(env, monkeypatch):
    for nom in ("bon", "sans_cuda", "casse"):
        (env / nom).mkdir()
    bon = _faux_python(env / "bon")
    sans_cuda = _faux_python(env / "sans_cuda", cuda=False)
    casse = _faux_python(env / "casse", casse=True)
    lancer = diagnostic.subprocess.run

    def run(cmd, **k):  # le « Python » de l'environnement est notre script
        return lancer([sys.executable, cmd[0]], **k) if cmd[0].endswith(".py") else lancer(cmd, **k)

    monkeypatch.setattr(diagnostic.subprocess, "run", run)
    assert diagnostic.sonde(str(bon)) == (True, "Python 3.12.4, PyTorch 2.6.0+cu126, carte : NVIDIA GeForce RTX 4070")
    ok, detail = diagnostic.sonde(str(sans_cuda))
    assert not ok and "CUDA indisponible" in detail
    ok, detail = diagnostic.sonde(str(casse))
    assert not ok and "No module named 'torch'" in detail
    ok, detail = diagnostic.sonde(str(env / "absent.exe"))
    assert not ok and "INSTALLER.bat" in detail


def test_diagnostic_rapide(env, monkeypatch):
    monkeypatch.setattr(diagnostic, "carte_graphique", lambda: "RTX 4070, pilote 580.1, mémoire 1.0 / 12.0 Go")
    monkeypatch.setattr(diagnostic, "sonde", lambda py: (True, f"ok {Path(py).name}"))
    *_, (texte, fichier) = list(diagnostic.rapide())
    assert "✅ **Carte graphique** : RTX 4070" in texte and "✅ **Diffusion** : ok" in texte
    assert "❌ **ACE-Step** : manquants" in texte  # modèles absents dans le dossier de test
    assert "Serveur ACE-Step" in texte and Path(fichier).read_text(encoding="utf-8").startswith("# Diagnostic rapide")
    assert Path(fichier).parent.parent == cfg.DATA_DIR / "diagnostic"


def test_essai_complet(env, monkeypatch):
    from studiovoix import acestep, chatterbox, demucs, diffusion, nettoyage, seedvc

    write_tone(cfg.VOICES_DIR / "moi.wav", seconds=5)
    galerie_avant = {n: getattr(cfg, n) for n in ("SONGS_DIR", "TTS_DIR", "SFX_DIR", "CARDS_DIR")}
    vus = {}

    def ace(params, dests, *a, **k):
        vus["tts_dir_pendant"] = cfg.TTS_DIR
        write_tone(dests[0], seconds=1)
        return [(str(dests[0]), 7)]

    monkeypatch.setattr(diagnostic, "carte_graphique", lambda: None)
    monkeypatch.setattr(acestep, "generer", ace)
    monkeypatch.setattr(demucs, "separate_vocals", lambda song, d: (Path(song), Path(song)))
    monkeypatch.setattr(seedvc, "convert_voice", lambda *a: a[-1] / "vc.wav")
    monkeypatch.setattr(chatterbox, "synthese", lambda voix, *a, **k: (str(cfg.TTS_DIR / f"{voix}.wav"), "ok"))
    monkeypatch.setattr(nettoyage, "nettoyer", lambda *a, **k: (str(cfg.CLEAN_DIR / "n.wav"), None, "ok"))
    monkeypatch.setattr(diffusion, "decrire", lambda *a, **k: "door creak")
    monkeypatch.setattr(diffusion, "image", lambda p, sorties, *a, **k: {"fichiers": [str(sorties[0])], "graines": [42]})
    monkeypatch.setattr(diffusion, "present", lambda nom: nom != "bruitages")
    monkeypatch.setattr(diffusion, "personnage",
                        lambda p, refs, sorties, *a, **k: vus.setdefault("references", refs) and {"fichiers": [str(sorties[0])]})

    def forme(*a, **k):
        raise RuntimeError("CUDA error: no kernel image is available")
    monkeypatch.setattr(diffusion, "forme3d", forme)
    etats = list(diagnostic.complet())
    texte, fichier = etats[-1]
    assert "✅ **ACE-Step (génération musicale)** : 10 s de musique, graine 7" in texte
    assert "✅ **Chatterbox (synthèse vocale)** : lecture écrite (moi.wav)" in texte
    assert "✅ **Qwen3-VL (description)** : « door creak »" in texte
    assert "✅ **FLUX.2 klein 4B (personnage d'après une référence)** : image 768×768" in texte
    assert [Path(r).name for r in vus["references"]] == ["zimage.png"]  # l'image de Z-Image sert de référence
    assert "❌ **Stable Audio Open (bruitage)** : Stable Audio Open non téléchargé" in texte  # échec n'arrête rien
    assert "❌ **Hunyuan3D-2 (forme + texture)** : RuntimeError : CUDA error: no kernel image" in texte
    assert Path(fichier).name == "rapport.txt" and "Essai complet" in Path(fichier).read_text(encoding="utf-8")
    # pendant l'essai la galerie était redirigée vers le dossier du diagnostic, puis restaurée
    assert Path(fichier).parent in vus["tts_dir_pendant"].parents
    assert {n: getattr(cfg, n) for n in galerie_avant} == galerie_avant
    assert len(etats) >= 10  # le rapport s'affiche au fil des étapes


def test_essai_sans_voix(env, monkeypatch):
    monkeypatch.setattr(diagnostic, "carte_graphique", lambda: None)
    etapes = {nom: f for nom, f, _ in diagnostic._etapes(env, None)}
    try:
        etapes["Chatterbox (synthèse vocale)"]()
        raise AssertionError("une erreur était attendue")
    except gr.Error as e:
        assert "aucune voix" in str(getattr(e, "message", e))


def test_mesure_de_la_carte_graphique(monkeypatch):
    mesures = iter([(10.0, 2.0, 60.0), (95.0, 9.5, 190.0), (100.0, 10.2, None), (98.0, 10.0, 185.0)] * 50)
    monkeypatch.setattr(diagnostic.shutil, "which", lambda nom: "nvidia-smi")
    monkeypatch.setattr(diagnostic, "_mesure_gpu", lambda: next(mesures))
    suivi = diagnostic.SuiviGPU(periode=0.01)
    with suivi:
        import time
        time.sleep(0.2)
    assert len(suivi.mesures) >= 4
    suivi.mesures = suivi.mesures[:4]
    assert suivi.resume() == ("GPU 76 % en moyenne (pointe 100 %, occupé à plus de 50 % 75% du temps), mémoire "
                              "jusqu'à 10.2 Go, 190 W au plus")
    assert diagnostic.SuiviGPU().resume() == ""
    # en direct : une ligne par mesure, puis le bilan et le conseil (graphe « Cuda » du Gestionnaire des tâches)
    horloge = iter(range(0, 1000, 2))
    monkeypatch.setattr(diagnostic.time, "monotonic", lambda: next(horloge))
    etats = list(diagnostic.surveiller_gpu(duree=6, attendre=lambda s: None))
    assert "**Maintenant**" in etats[0] and "%" in etats[0] and "Bilan sur 6 s" in etats[-1]
    assert "Cuda" in etats[-1]
    monkeypatch.setattr(diagnostic.shutil, "which", lambda nom: None)
    assert "nvidia-smi introuvable" in next(diagnostic.surveiller_gpu())


def test_moteur_du_modele_de_langage_ace_step(env):
    from studiovoix import serveur_acestep

    assert diagnostic.moteur_lm_ace()[0] is None
    venv = cfg.ACESTEP_DIR / ".venv"
    venv.mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = D:\\StudioVoix\\python\\cpython-3.12\nversion_info = 3.12.10\n")
    ok, detail = diagnostic.moteur_lm_ace()
    assert ok is False and "3.12.10" in detail and "METTRE_A_JOUR" in detail
    (venv / "pyvenv.cfg").write_text("version_info = 3.11.13\n")
    assert diagnostic.moteur_lm_ace() == (True, "Python 3.11.13 : moteur rapide (nano-vLLM) disponible")
    serveur_acestep.journal().write_text("[API Server] Warning: vLLM backend is unavailable on Windows because Triton "
                                         "is not installed")
    ok, detail = diagnostic.moteur_lm_ace()
    assert ok is False and "Triton absent" in detail
