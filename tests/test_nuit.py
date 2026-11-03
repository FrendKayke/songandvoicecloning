"""File de tâches pour la nuit : lecture du fichier, vérification avant la nuit, exécution avec échecs consignés."""
import json

import pytest

from studiovoix import nuit


def _ecrire(tmp_path, texte, codage="utf-8"):
    f = tmp_path / "taches.txt"
    f.write_bytes(texte.encode(codage))
    return f


def test_lecture_du_fichier():
    reglages, taches = nuit.lire(
        "# commentaire\neteindre: oui\n\n== Chanson : Ballade\nstyle: pop, piano\nvoix: ace\nparoles:\nCouplet 1\n"
        "Il est: tard\n\n== IMAGE : Château\ndescription: un château\n== modèle 3D\nqualité: low poly\n")
    assert reglages == {"eteindre": "oui"}
    assert [t["type"] for t in taches] == ["chanson", "image", "3d"]
    assert taches[0]["nom"] == "Ballade" and taches[2]["nom"] == ""
    # la clé sans valeur prend tout jusqu'à la tâche suivante, « : » compris
    assert taches[0]["champs"]["paroles"] == "Couplet 1\nIl est: tard"
    assert taches[2]["champs"] == {"qualite": "low poly"}


@pytest.mark.parametrize("texte,attendu", [
    ("== peinture : x\n", "type de tâche inconnu"),
    ("== image : x\ncouleur: rouge\n", "n'est pas une clé de image"),
    ("== image : x\nune ligne sans clé\n", "n'est pas une ligne"),
    ("vitesse: 3\n== image : x\n", "réglage inconnu"),
])
def test_fautes_de_lecture(texte, attendu):
    with pytest.raises(nuit.ErreurFichier, match=attendu):
        nuit.lire(texte)


def test_choix_des_listes():
    from studiovoix import modele3d

    assert nuit.choix("low poly facette", list(modele3d.QUALITES), "x", "Qualité").startswith("Low poly facetté")
    assert nuit.choix("", ["a"], "défaut", "x") == "défaut"
    with pytest.raises(nuit.ErreurFichier, match="inconnu"):
        nuit.choix("zzz", ["a", "b"], "a", "Format")
    assert nuit.langue("anglais") == "Anglais" and nuit.langue("en") == "Anglais" and nuit.langue("") == "Français"


def test_verification_avant_la_nuit(env, tmp_path):
    """Toutes les fautes sont signalées d'un coup, avec la ligne de la tâche, avant toute génération."""
    f = _ecrire(tmp_path, "== chanson : A\nstyle: pop\nvoix: Inconnue\nparoles:\nla la\n"
                          "== chanson : B\nstyle: rock\nvoix: ace\n"
                          "== 3d : C\nimage: absente.png\n"
                          "== image : D\nformat: pentagone\ndescription: x\n"
                          "== lecture : E\nvoix: Inconnue\n")
    with pytest.raises(nuit.ErreurFichier) as e:
        nuit.verifier(f)
    msg = str(e.value)
    assert "« A » (ligne 1)" in msg and "absente de la bibliothèque" in msg
    assert "« B » (ligne 6)" in msg and "paroles" in msg
    assert "Fichier introuvable" in msg and "Format inconnu" in msg and "il manque le texte" in msg


def test_fichier_du_bloc_notes_en_cp1252(env, tmp_path):
    f = _ecrire(tmp_path, "== image : Été\ndescription: un château médiéval\n", "cp1252")
    _, pretes = nuit.verifier(f)
    assert pretes[0][0]["nom"] == "Été"


def test_execution_une_tache_en_echec_n_arrete_pas_la_nuit(env, tmp_path, monkeypatch):
    from studiovoix import images, modele3d, pipeline

    appels = []

    def faux_prompt(description, prompt, *a, progress=None, **k):
        return prompt or f"EN: {description}"

    def faux_generer(prompt, photo, usage, styles, fmt, n, graine, nom="", description_fr=None, progress=None):
        appels.append(("image", prompt, styles, fmt, n, graine, nom))
        progress(0.5, desc="Génération…")
        if "ECHEC" in prompt:
            raise RuntimeError("plus de mémoire graphique")
        d = tmp_path / f"img_{nom}"
        d.mkdir()
        return f"✅ {n} images", [], str(d)

    def fausse_chanson(voix, *a, mode=None, description=None, versions=1, graine=0, progress=None):
        appels.append(("chanson", voix, a[7], mode, description, a[8], versions))
        d = tmp_path / "chanson"
        d.mkdir()
        return str(d / "finale.wav"), None, None, None, "Chanson prête.", None

    monkeypatch.setattr(images, "prompt_pret", faux_prompt)
    monkeypatch.setattr(images, "generer", faux_generer)
    monkeypatch.setattr(pipeline, "creer_chanson", fausse_chanson)
    monkeypatch.setattr(modele3d, "prompt_pret", lambda *a, **k: pytest.fail("pas de 3D ici"))
    f = _ecrire(tmp_path, "== image : Un\nprompt: a castle ECHEC\n"
                          "== image : Deux\ndescription: un château\nstyle: aquarelle, crayon\nvariantes: 9\n"
                          "graine: 42\nformat: carré\n"
                          "== musique : Trois\nstyle: médiéval, taverne\nduree: 1:30\n")
    lignes = []
    rapport = nuit.executer(f, sortie=lignes.append)
    assert [t["etat"] for t in rapport["taches"]] == ["echec", "reussie", "reussie"]
    assert rapport["reussies"] == 2 and rapport["echecs"] == 1
    assert "plus de mémoire graphique" in rapport["taches"][0]["message"]
    _, prompt, styles, fmt, n, graine, nom = appels[1]
    assert prompt == "EN: un château" and styles == ["aquarelle", "crayon"] and n == 4 and graine == 42
    assert fmt.startswith("Carré") and nom == "Deux"
    _, voix, langue, mode, description, duree, versions = appels[2]
    assert voix is None and mode == pipeline.MODE_INSTRU and duree == 90 and langue == "Français"
    assert "medieval" in description.lower()  # libellé français remplacé par les termes anglais
    dossier = rapport["dossier"]
    from pathlib import Path

    assert json.loads((Path(dossier) / "rapport.json").read_text(encoding="utf-8"))["echecs"] == 1
    assert "RuntimeError" in (Path(dossier) / "erreurs.txt").read_text(encoding="utf-8")
    assert (Path(dossier) / "taches.txt").exists()  # copie du fichier lancé
    journal = (Path(dossier) / "journal.txt").read_text(encoding="utf-8")
    assert "[2/3] image « Deux »" in journal and "50 % Génération…" in journal and "2 réussie(s)" in journal
    # reprise à la tâche 3
    appels.clear()
    rapport = nuit.executer(f, a_partir_de=3, sortie=lambda _: None)
    assert [t["numero"] for t in rapport["taches"]] == [3]


def test_arret_apres_plusieurs_echecs(env, tmp_path, monkeypatch):
    from studiovoix import images

    monkeypatch.setattr(images, "prompt_pret", lambda d, p, *a, **k: p)
    monkeypatch.setattr(images, "generer", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("non")))
    f = _ecrire(tmp_path, "arreter apres: 2\n" + "".join(f"== image : {i}\nprompt: x\n" for i in range(5)))
    rapport = nuit.executer(f, sortie=lambda _: None)
    assert len(rapport["taches"]) == 2 and rapport["echecs"] == 2


def test_ligne_de_commande(env, tmp_path, capsys):
    exemple = tmp_path / "exemple.txt"
    assert nuit.principal(["--exemple", str(exemple)]) == 0
    # l'exemple fourni passe la vérification (sa chanson n'a pas besoin d'une voix de la bibliothèque)
    assert nuit.principal(["--verifier", str(exemple)]) == 0
    assert "5 tâche(s)" in capsys.readouterr().out
    faux = _ecrire(tmp_path, "== image : x\n")
    assert nuit.principal(["--verifier", str(faux)]) == 3
    assert "il manque" in capsys.readouterr().out
