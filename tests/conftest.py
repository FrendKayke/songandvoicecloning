"""Outils de test sans GPU : faux serveur ACE-Step (HTTP), faux Demucs et faux Seed-VC (sous-processus)."""
import json
import sys
import textwrap
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from studiovoix import acestep, config as cfg  # noqa: E402

SR = 44100


def write_tone(path, seconds=3.0, freq=220.0, amp=0.3, sr=SR, channels=1):
    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
    y = (amp * np.sin(2 * np.pi * freq * t)).astype("float32")
    if channels == 2:
        y = np.stack([y, y], axis=1)
    sf.write(str(path), y, sr)
    return Path(path)


def energie(path, freq):
    """Amplitude relative de la fréquence freq dans un fichier (pour savoir si une piste y est présente)."""
    y, sr = sf.read(str(path), always_2d=True)
    y = y.mean(axis=1)
    spectre = np.abs(np.fft.rfft(y)) / len(y)
    return float(spectre[int(round(freq * len(y) / sr))])


def no_progress(*args, **kwargs):
    pass


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Dossiers de données et de moteurs isolés dans un dossier temporaire."""
    data = tmp_path / "data"
    for d in (data / "voices", data / "songs"):
        d.mkdir(parents=True)
    monkeypatch.setattr(cfg, "DATA_DIR", data)
    monkeypatch.setattr(cfg, "VOICES_DIR", data / "voices")
    monkeypatch.setattr(cfg, "SONGS_DIR", data / "songs")
    monkeypatch.setattr(cfg, "ENG_DIR", tmp_path / "StudioVoix")
    monkeypatch.setattr(cfg, "ACESTEP_DIR", tmp_path / "StudioVoix" / "ace-step")
    monkeypatch.setattr(cfg, "SEEDVC_DIR", tmp_path / "StudioVoix" / "seed-vc")
    monkeypatch.setattr(cfg, "SEEDVC_PYTHON", str(tmp_path / "absent" / "python.exe"))
    monkeypatch.setattr(acestep.time, "sleep", lambda s: None)
    return tmp_path


# --- Faux serveur ACE-Step ---------------------------------------------------
class FakeAceStep:
    """Imite /health, /release_task, /query_result et /v1/audio."""

    def __init__(self, wav_bytes, fail_with_thinking=False, pending_polls=1, broken_polls=0):
        self.wav_bytes = wav_bytes
        self.fail_with_thinking = fail_with_thinking
        self.pending_polls = pending_polls
        self.broken_polls = broken_polls
        self.payloads = []
        self.polls = 0
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _json(self, obj, code=200):
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/health":
                    return self._json({"data": {"status": "ok"}, "code": 200})
                if self.path.startswith("/v1/audio"):
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(fake.wav_bytes)))
                    self.end_headers()
                    self.wfile.write(fake.wav_bytes)
                    return
                self._json({"error": "inconnu"}, 404)

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n) or b"{}")
                if self.path == "/release_task":
                    fake.payloads.append(body)
                    return self._json({"data": {"task_id": f"t{len(fake.payloads)}"}, "code": 200})
                if self.path == "/query_result":
                    fake.polls += 1
                    if fake.polls <= fake.broken_polls:
                        return self._json({"error": "occupé"}, 500)
                    tid = body["task_id_list"][0]
                    last = fake.payloads[-1]
                    if fake.fail_with_thinking and last.get("thinking"):
                        return self._json({"data": [{"task_id": tid, "status": 2, "result": "échec LM"}]})
                    if fake.polls <= fake.broken_polls + fake.pending_polls:
                        return self._json({"data": [{"task_id": tid, "status": 0, "result": None}]})
                    result = json.dumps([{"file": "/v1/audio?path=%2Ftmp%2Fsong.wav"}])
                    return self._json({"data": [{"task_id": tid, "status": 1, "result": result}]})
                self._json({"error": "inconnu"}, 404)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


@pytest.fixture
def fake_acestep(env, monkeypatch):
    song = write_tone(env / "song_src.wav", seconds=4.0, channels=2)
    servers = []

    def make(**kw):
        s = FakeAceStep(song.read_bytes(), **kw)
        monkeypatch.setattr(cfg, "ACESTEP_URL", s.url)
        servers.append(s)
        return s

    yield make
    for s in servers:
        s.close()


# --- Faux moteurs en sous-processus -------------------------------------------
# Fréquence de la tonalité écrite dans chaque piste par le faux Demucs en mode 4 pistes
FREQ_PISTES = {"drums": 100.0, "bass": 200.0, "other": 300.0, "vocals": 400.0}

FAKE_DEMUCS = textwrap.dedent('''
    import sys, shutil
    from pathlib import Path
    import numpy as np, soundfile as sf
    args = sys.argv[1:]
    assert args[args.index("-n") + 1] == "htdemucs", args
    out = Path(args[args.index("-o") + 1]); song = Path(args[-1])
    d = out / "htdemucs" / song.stem
    d.mkdir(parents=True)
    if "--two-stems=vocals" in args:
        shutil.copy(song, d / "vocals.wav"); shutil.copy(song, d / "no_vocals.wav")
    else:  # 4 pistes, chacune une tonalité reconnaissable
        info = sf.info(str(song)); t = np.arange(info.frames) / info.samplerate
        for piste, f in %r.items():
            y = (0.2 * np.sin(2 * np.pi * f * t)).astype("float32")
            sf.write(str(d / f"{piste}.wav"), np.stack([y, y], 1), info.samplerate)
''' % FREQ_PISTES)

FAKE_SEEDVC = textwrap.dedent('''
    import sys, json, shutil, argparse
    from pathlib import Path
    p = argparse.ArgumentParser()
    for a in ("--source", "--target", "--output", "--diffusion-steps", "--f0-condition",
              "--auto-f0-adjust", "--semi-tone-shift", "--fp16"):
        p.add_argument(a)
    a = p.parse_args()
    Path("appel.json").write_text(json.dumps(sys.argv[1:]))
    src = Path(a.source)
    shutil.copy(src, Path(a.output) / f"vc_{src.stem}_{Path(a.target).stem}_1.0_{a.diffusion_steps}_0.7.wav")
''')


@pytest.fixture
def fake_engines(env, monkeypatch):
    """Faux Demucs (module python -m demucs) et faux Seed-VC (inference.py), lancés avec ce Python."""
    pkgs = env / "fakepkgs" / "demucs"
    pkgs.mkdir(parents=True)
    (pkgs / "__init__.py").write_text("")
    (pkgs / "__main__.py").write_text(FAKE_DEMUCS)
    monkeypatch.setenv("PYTHONPATH", str(env / "fakepkgs"))
    cfg.SEEDVC_DIR.mkdir(parents=True)
    (cfg.SEEDVC_DIR / "inference.py").write_text(FAKE_SEEDVC)
    monkeypatch.setattr(cfg, "SEEDVC_PYTHON", sys.executable)
    return env
