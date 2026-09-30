import urllib.request

from studiovoix import config as cfg
from studiovoix.interface import build_ui
from studiovoix.modeles import models_status_md


def test_etat_des_modeles(env):
    md = models_status_md()
    assert md.count("❌ absent") == 3
    assert str(cfg.VOICES_DIR) in md


def test_demarrage_de_l_interface(env):
    demo = build_ui()
    demo.queue().launch(prevent_thread_lock=True, server_port=None, quiet=True)
    try:
        with urllib.request.urlopen(demo.local_url, timeout=30) as r:
            assert r.status == 200
    finally:
        demo.close()
