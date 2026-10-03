"""Ollama : ses modèles chargés sur la carte graphique sont déchargés avant nos générations (faux serveur HTTP)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from studiovoix import diagnostic, ollama, outils, serveur_acestep


@pytest.fixture
def faux_ollama(monkeypatch):
    etat = {"modeles": [{"name": "llama3.1:8b", "size_vram": 5_600_000_000},
                        {"name": "nomic-embed-text", "size_vram": 0}], "decharges": []}

    class Gestion(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, obj):
            corps = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(corps)))
            self.end_headers()
            self.wfile.write(corps)

        def do_GET(self):
            assert self.path == "/api/ps"
            self._json({"models": etat["modeles"]})

        def do_POST(self):
            assert self.path == "/api/generate"
            demande = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert demande["keep_alive"] == 0 and "prompt" not in demande
            etat["decharges"].append(demande["model"])
            etat["modeles"] = [m for m in etat["modeles"] if m["name"] != demande["model"]]
            self._json({"model": demande["model"], "done": True, "done_reason": "unload"})

    serveur = ThreadingHTTPServer(("127.0.0.1", 0), Gestion)
    threading.Thread(target=serveur.serve_forever, daemon=True).start()
    monkeypatch.setenv("OLLAMA_HOST", f"0.0.0.0:{serveur.server_address[1]}")  # adresse d'écoute : on vise 127.0.0.1
    yield etat
    serveur.shutdown()


def test_decharger_les_modeles_sur_la_carte(faux_ollama):
    assert ollama.modeles_charges()[0] == ("llama3.1:8b", 5.6)
    messages = []
    assert ollama.liberer(lambda p, desc="": messages.append(desc)) == ["llama3.1:8b"]
    assert faux_ollama["decharges"] == ["llama3.1:8b"]  # le modèle sur le processeur seulement est laissé
    assert "Ollama décharge llama3.1:8b" in messages[0]
    assert ollama.liberer() == []  # plus rien sur la carte


def test_avant_chaque_moteur_et_avant_ace_step(faux_ollama, env, monkeypatch, tmp_path):
    import sys

    script = tmp_path / "rien.py"
    script.write_text("print('PROGRESSION 1/1 ok')\n")
    outils.lancer_moteur([sys.executable, str(script)], tmp_path, None, "Essai")
    assert faux_ollama["decharges"] == ["llama3.1:8b"]
    faux_ollama["modeles"].append({"name": "qwen3:14b", "size_vram": 9e9})
    serveur_acestep.liberer_gpu()
    assert faux_ollama["decharges"][-1] == "qwen3:14b"
    # le journal du moteur garde toute sa sortie (dépannage à distance)
    journal = outils.journal_moteur("Essai").read_text(encoding="utf-8")
    assert "=== Essai" in journal and "PROGRESSION 1/1 ok" in journal and "fin (code 0)" in journal


def test_diagnostic_et_reglage(faux_ollama, monkeypatch):
    ok, detail = ollama.etat()
    assert "llama3.1:8b (5.6 Go sur la carte)" in detail and "déchargé(s) automatiquement" in detail
    monkeypatch.setattr(ollama, "ACTIF", False)
    assert ollama.liberer() == [] and faux_ollama["decharges"] == []
    assert ollama.etat()[0] is False
    assert diagnostic.programmes_sur_la_carte() in (None, "") or isinstance(diagnostic.programmes_sur_la_carte(), str)


def test_ollama_absent():
    assert ollama.modeles_charges() == [] and ollama.liberer() == [] and ollama.etat()[0] is None
