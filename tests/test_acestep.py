import gradio as gr
import pytest

from conftest import no_progress, write_tone
from studiovoix import acestep


def test_build_prompt():
    # « triste » tapé à la main est le libellé de la liste des ambiances : traduit
    assert acestep.build_prompt("pop", " 80s ", "", "triste", "Voix féminine", "") == "pop, 80s, sad, female vocals"
    assert acestep.build_prompt("rock", "", "", "", "Voix masculine", "x") == "rock, male vocals, x"
    assert acestep.build_prompt("rock", "", "", "", "Automatique", None) == "rock"


def test_build_prompt_avec_listes():
    genre = ["8-bit chiptune, retro video game music", " vaporwave "]  # choix de liste + saisie libre
    assert acestep.build_prompt(genre, [], ["square wave synth, chiptune arpeggios"], None, "Automatique", []) == (
        "8-bit chiptune, retro video game music, vaporwave, square wave synth, chiptune arpeggios")


def test_generation_nominale(fake_acestep, env):
    srv = fake_acestep(pending_polls=2)
    dest = acestep.acestep_generate("pop", "[Verse]\nla", "fr", 60, 120, True, env / "s.wav", no_progress)
    assert dest.read_bytes() == srv.wav_bytes
    p = srv.payloads[0]
    assert p["prompt"] == "pop" and p["vocal_language"] == "fr" and p["bpm"] == 120
    assert p["audio_duration"] == 60.0 and p["thinking"] is True and p["audio_format"] == "wav"
    # La description doit arriver telle quelle au générateur : pas de réécriture par le modèle de langage
    assert p["use_cot_caption"] is False and p["use_cot_language"] is False


def test_bpm_auto_non_envoye(fake_acestep, env):
    srv = fake_acestep()
    acestep.acestep_generate("pop", "x", "fr", 60, 0, False, env / "s.wav", no_progress)
    assert "bpm" not in srv.payloads[0]


def test_serveur_occupe_tolere(fake_acestep, env):
    """Des réponses en erreur pendant le chargement ne doivent pas faire échouer la génération."""
    srv = fake_acestep(broken_polls=3)
    acestep.acestep_generate("pop", "x", "fr", 60, 0, False, env / "s.wav", no_progress)
    assert srv.polls >= 4


def test_nouvel_essai_sans_reflexion(fake_acestep, env):
    srv = fake_acestep(fail_with_thinking=True)
    acestep.acestep_generate("pop", "x", "fr", 60, 0, True, env / "s.wav", no_progress)
    assert [p["thinking"] for p in srv.payloads] == [True, False]


def test_serveur_injoignable(env, monkeypatch):
    monkeypatch.setattr(acestep.cfg, "ACESTEP_URL", "http://127.0.0.1:9")
    with pytest.raises(gr.Error):
        acestep.wait_acestep(no_progress, timeout=0.5)


# --- Versions, graines, fichiers téléversés ---------------------------------------------------
from conftest import energie  # noqa: E402


def test_plusieurs_versions_et_graines(fake_acestep, env):
    srv = fake_acestep()
    res = acestep.generer({"prompt": "chiptune", "lyrics": "[Instrumental]"},
                          [env / "a.wav", env / "b.wav"], no_progress, graine=1234)
    (a, ga), (b, gb) = res
    p = srv.payloads[0]
    assert p["batch_size"] == 2 and p["use_random_seed"] is False and p["seed"] == f"1234,{gb}"
    assert ga == 1234 and gb > 0
    assert energie(a, 220) > 0.05 and energie(b, 440) > 0.05  # chaque fichier va à sa version


def test_graine_aleatoire_mais_connue(fake_acestep, env):
    srv = fake_acestep()
    ((_, g),) = acestep.generer({"prompt": "x"}, [env / "a.wav"], no_progress, graine=0)
    assert srv.payloads[0]["seed"] == str(g) and g > 0


def test_fichiers_televerses_en_multipart(fake_acestep, env):
    srv = fake_acestep()
    ref = write_tone(env / "thème.wav", seconds=2)
    acestep.generer({"prompt": "x", "task_type": "cover", "audio_cover_strength": 0.4, "thinking": False},
                    [env / "a.wav"], no_progress, fichiers={"reference_audio": ref, "src_audio": ref})
    p, f = srv.payloads[0], srv.fichiers[0]
    assert p["task_type"] == "cover" and p["audio_cover_strength"] == "0.4"
    assert p["thinking"] == "false" and p["use_cot_caption"] == "false"  # texte relu par _to_bool côté serveur
    assert f["reference_audio"] == ("thème.wav", ref.read_bytes()) and f["src_audio"][1] == ref.read_bytes()


def test_nouvel_essai_garde_la_graine(fake_acestep, env):
    srv = fake_acestep(fail_with_thinking=True)
    acestep.generer({"prompt": "x", "thinking": True}, [env / "a.wav"], no_progress, graine=77)
    assert [p["seed"] for p in srv.payloads] == ["77", "77"]


def test_termes_francais_traduits_pour_ace_step():
    """« Medieval, medievaux, medievale » tapés librement donnaient une production pop (basse, batterie)."""
    from studiovoix import styles

    p = acestep.build_prompt(["Medieval"], ["medievale"], ["Luth", "vielle a roue", "bagpipes"], ["Médiéval"],
                             "Automatique", ["raspy vocals"])
    assert "medieval folk music, early music" in p and "lute" in p and "hurdy-gurdy" in p and "bagpipes" in p
    assert "medievale" not in p and "Medieval," not in p
    assert p.count("medieval folk music, early music, acoustic period instruments") == 1  # pas de doublon dans un champ
    # un libellé du catalogue tapé à la main (accents, majuscules) → ses termes anglais ; le reste passe tel quel
    assert styles.musique(["CHANSON DE TAVERNE", "synthwave"]).startswith("medieval tavern drinking song")
    assert styles.musique("dark ambient drone") == "dark ambient drone"
    assert styles.texte(["médiéval"]) == "médiéval"  # texte() reste neutre (illustrations)


def test_titres_de_section_en_balises():
    paroles = ("On croyait que ce s'rait banal,\n\nRefrain\nBuvons tous ensemble,\nLe roi n'en saura rien !\n\n"
               "Couplet 2\nY avait un vieux à la flûte,\n\nRefrain\n\nCouplet 3 :\nDes Templiers en goguette,\n"
               "[Pont]\nLa la la\n[Verse 4]\nDéjà balisé")
    b = acestep.baliser_paroles(paroles)
    assert b.splitlines() == [
        "On croyait que ce s'rait banal,", "", "[Chorus]", "Buvons tous ensemble,", "Le roi n'en saura rien !", "",
        "[Verse 2]", "Y avait un vieux à la flûte,", "", "[Chorus]", "Buvons tous ensemble,",
        "Le roi n'en saura rien !", "", "[Verse 3]", "Des Templiers en goguette,", "[Bridge]", "La la la",
        "[Verse 4]", "Déjà balisé"]
    # une vraie phrase qui commence par un mot de section n'est pas touchée
    assert acestep.baliser_paroles("Refrain de ma vie, tu chantes encore") == "Refrain de ma vie, tu chantes encore"


def test_plusieurs_voix():
    """Duo, chœur : termes de la documentation d'ACE-Step dans la description ; qui chante chaque partie, écrit
    après son titre, devient une précision de la balise ([Verse 1 - male vocal])."""
    p = acestep.build_prompt("pop", "", "", "", "Duo homme et femme", "")
    assert p == "pop, male and female duet, male vocals, female vocals, vocal harmonies"
    assert "choir" in acestep.build_prompt("", "", "", "", "Chœur (tout le monde chante ensemble)", "")
    assert set(acestep.PLUSIEURS_VOIX) == set(acestep.VOIX_CHANTEES) - {"Automatique", "Voix masculine", "Voix féminine"}
    b = acestep.baliser_paroles("Couplet 1 (homme)\nJe pars\n\nCouplet 2 - femme\nJe reste\n\nRefrain : tous ensemble"
                                "\nOn chante (oh oh)\n\nPont (chœur)\nLa la\n\nRefrain (ensemble)\n\n[Chorus - anthemic]\nx")
    assert b.splitlines() == [
        "[Verse 1 - male vocal]", "Je pars", "", "[Verse 2 - female vocal]", "Je reste", "",
        "Refrain : tous ensemble", "On chante (oh oh)", "", "[Bridge - choir]", "La la", "",
        "[Chorus - duet, harmonies]", "", "[Chorus - anthemic]", "x"]
    # « Refrain : tous ensemble » (deux mots) n'est pas un chanteur connu : laissé tel quel, comme une parole
    assert acestep.baliser_paroles("Refrain (ensemble)\nOn chante\n\nRefrain (ensemble)").splitlines() == [
        "[Chorus - duet, harmonies]", "On chante", "", "[Chorus - duet, harmonies]", "On chante"]


def test_longueurs_lues_par_ace_step():
    """ACE-Step coupe sans prévenir la description (256 jetons avec ses consignes) et les paroles (2048 jetons)."""
    assert acestep.longueurs_trop_grandes({"prompt": "pop, piano", "lyrics": "[Verse]\nla"}) == []
    a = acestep.longueurs_trop_grandes({"prompt": "epic orchestral, " * 60 + "FIN", "lyrics": "la la\n" * 1000})
    assert len(a) == 2 and "Description de" in a[0] and "FIN" in a[0] and "ne sera pas chantée" in a[1]
