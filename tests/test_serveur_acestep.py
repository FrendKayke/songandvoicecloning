"""Serveur ACE-Step géré par l'application : démarrage, attente, arrêt pour libérer la carte graphique, reprise d'un
serveur lancé autrement. Le « serveur » est un vrai processus séparé (petit serveur HTTP qui répond à /health)."""
import os
import socket
import subprocess
import sys
import textwrap
import time

import gradio as gr
import pytest

from conftest import no_progress
from studiovoix import config as cfg
from studiovoix import serveur_acestep as srv

FAUX_SERVEUR = textwrap.dedent("""
    import sys, http.server
    print("chargement des modèles…", flush=True)
    if sys.argv[2] == "plante":
        print("ERREUR : CUDA indisponible", flush=True); sys.exit(1)
    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200 if self.path == "/health" else 404); self.end_headers(); self.wfile.write(b"{}")
        def log_message(self, *a): pass
    http.server.HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
""")


def _port_libre():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def serveur(env, monkeypatch):
    port = _port_libre()
    script = env / "faux_serveur.py"
    script.write_text(FAUX_SERVEUR, encoding="utf-8")
    monkeypatch.setattr(cfg, "ACESTEP_URL", f"http://127.0.0.1:{port}")
    cfg.ACESTEP_DIR.mkdir(parents=True)
    mode = {"m": "ok"}
    monkeypatch.setattr(srv, "commande", lambda: [sys.executable, str(script), str(port), mode["m"]])
    monkeypatch.setattr(srv, "LIBERATION_AUTO", True)
    monkeypatch.setattr(srv, "_processus", None)
    yield mode, script, port
    srv.arreter(attente=5)


def test_demarrage_puis_liberation_puis_relance(serveur):
    assert not srv.repond()
    etapes = []
    srv.assurer(lambda p, desc="": etapes.append(desc), timeout=30)
    assert srv.repond() and "chargement des modèles" in srv.journal().read_text(encoding="utf-8")
    assert "🟢" in srv.etat()
    assert not srv.demarrer()  # déjà en marche : pas de second serveur
    assert srv.liberer_gpu(lambda p, desc="": etapes.append(desc)) is True
    assert not srv.repond() and "⚪" in srv.etat()
    assert any("Libération de la carte graphique" in e for e in etapes)
    assert srv.liberer_gpu() is False  # rien à arrêter
    srv.assurer(no_progress, timeout=30)  # la génération suivante le relance
    assert srv.repond()


def test_liberation_desactivee(serveur):
    srv.assurer(no_progress, timeout=30)
    assert "désactivée" in srv.regler_liberation(False)
    assert srv.liberer_gpu() is False and srv.repond()
    assert "automatiquement" in srv.regler_liberation(True)


def test_serveur_qui_plante_au_demarrage(serveur):
    mode, *_ = serveur
    mode["m"] = "plante"
    with pytest.raises(gr.Error, match="CUDA indisponible"):
        srv.assurer(no_progress, timeout=30)


def test_ace_step_absent_ou_distant(env, monkeypatch):
    monkeypatch.setattr(srv, "_processus", None)
    with pytest.raises(gr.Error, match="INSTALLER.bat"):
        srv.assurer(no_progress, timeout=1)
    monkeypatch.setattr(cfg, "ACESTEP_URL", "http://192.0.2.1:8001")
    assert not srv.local() and srv.demarrer() is False and srv.liberer_gpu() is False and "distant" in srv.etat()


def test_moteurs_gourmands_liberent_la_carte(env, monkeypatch):
    from studiovoix import diffusion

    appels = []
    monkeypatch.setattr(srv, "liberer_gpu", lambda progress=None: appels.append("liberer"))
    monkeypatch.setattr(diffusion, "lancer_moteur", lambda *a, **k: appels.append("moteur") or ['RESULTAT {"x": 1}'])
    assert diffusion.lancer("decrire", {}, env / "d", "Qwen") == {"x": 1}
    assert appels == ["liberer", "moteur"]


@pytest.mark.skipif(os.name != "nt", reason="reprise d'un serveur externe : Windows (PowerShell, taskkill)")
def test_reprise_d_un_serveur_lance_autrement(serveur):
    _, script, port = serveur
    externe = subprocess.Popen([sys.executable, str(script), str(port), "ok"])
    try:
        t0 = time.time()
        while not srv.repond(1) and time.time() - t0 < 30:
            time.sleep(0.3)
        assert srv.repond() and srv._processus is None
        assert srv.arreter(attente=20) is True and not srv.repond()
        assert externe.wait(timeout=20) is not None
    finally:
        if externe.poll() is None:
            externe.kill()


def test_modele_de_langage_1_7b_sur_une_carte_de_12_go(monkeypatch):
    """ACE-Step range les cartes de 12 Go (11,99 Go annoncés par une RTX 4070) avec celles de 8 Go (modèle de langage
    0.6B seul) : MAX_CUDA_VRAM les fait passer dans la catégorie 12–16 Go (1.7B)."""
    monkeypatch.delenv("MAX_CUDA_VRAM", raising=False)
    monkeypatch.delenv("ACESTEP_LM_MODEL_PATH", raising=False)
    monkeypatch.setattr(srv, "MODELE_LM", "1.7B")
    for memoire, attendu in ((11.99, "12.5"), (12.0, "12.5"), (16.0, None), (8.0, None), (None, None)):
        monkeypatch.setattr(srv, "memoire_gpu_go", lambda m=memoire: m)
        env = srv.env_serveur()
        assert env.get("MAX_CUDA_VRAM") == attendu, memoire
        assert env["ACESTEP_LM_MODEL_PATH"] == "acestep-5Hz-lm-1.7B"
    monkeypatch.setattr(srv, "memoire_gpu_go", lambda: 11.99)
    assert "0.6B" in srv.regler_modele_lm("0.6B : plus léger")
    env = srv.env_serveur()
    assert "MAX_CUDA_VRAM" not in env and env["ACESTEP_LM_MODEL_PATH"] == "acestep-5Hz-lm-0.6B"
    assert srv.libelle_modele_lm().startswith("0.6B")
    monkeypatch.setenv("MAX_CUDA_VRAM", "10")  # réglage de l'utilisateur : jamais remplacé
    srv.regler_modele_lm(next(iter(srv.MODELES_LM)))
    assert srv.env_serveur()["MAX_CUDA_VRAM"] == "10"
