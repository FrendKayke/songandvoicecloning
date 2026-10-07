import urllib.request

from studiovoix import config as cfg
from studiovoix.interface import build_ui
from studiovoix.modeles import models_status_md


def test_etat_des_modeles(env):
    md = models_status_md()
    assert md.count("❌ absent") == 7
    assert str(cfg.VOICES_DIR) in md


def test_demarrage_de_l_interface(env):
    demo = build_ui()
    demo.queue().launch(prevent_thread_lock=True, server_port=None, quiet=True)
    try:
        with urllib.request.urlopen(demo.local_url, timeout=30) as r:
            assert r.status == 200
    finally:
        demo.close()


def test_affichage_selon_le_mode():
    from studiovoix.interface import maj_mode
    from studiovoix.pipeline import MODE_INSTRU, MODE_MA_VOIX, MODE_VOIX_ACE

    vis = lambda mode: [u.get("visible") for u in maj_mode(mode)]  # noqa: E731
    assert vis(MODE_MA_VOIX) == [True, True, True, True, True, True, True, None, True, True]
    assert vis(MODE_VOIX_ACE) == [False, True, True, False, False, False, False, None, False, True]
    assert vis(MODE_INSTRU) == [False, False, False, False, False, False, False, None, False, False]
    assert maj_mode(MODE_INSTRU)[7]["label"] == "Instrumental"


def test_apercu_de_la_description():
    from studiovoix.interface import apercu_description
    from studiovoix.pipeline import MODE_INSTRU, MODE_VOIX_ACE

    args = (["8-bit chiptune, retro video game music"], [], ["square wave synth, chiptune arpeggios"], ["happy, joyful"], [])
    assert apercu_description(*args, "Voix féminine", MODE_VOIX_ACE).endswith("happy, joyful, female vocals")
    assert "vocals" not in apercu_description(*args, "Voix féminine", MODE_INSTRU)


def test_catalogue_des_styles():
    from studiovoix.styles import LISTES

    for cle, choix in LISTES.items():
        libelles = [a for a, _ in choix]
        valeurs = [b for _, b in choix]
        assert len(set(libelles)) == len(libelles) and len(set(valeurs)) == len(valeurs), cle
        assert all(v.isascii() for v in valeurs if v != "French variété pop"), cle  # termes anglais
        assert not any(" sans " in f" {v} " for v in valeurs), cle
