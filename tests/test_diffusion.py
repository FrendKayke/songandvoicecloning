"""Moteur de diffusion (faux moteur en sous-processus) : bruitages, descriptions, jeton, état des modèles."""
import importlib.util
import json
import sys
import textwrap
from pathlib import Path

import gradio as gr
import pytest
import soundfile as sf

from conftest import no_progress
from studiovoix import bruitages, config as cfg, diffusion, modele3d
from studiovoix.modeles import models_status_md

_spec = importlib.util.spec_from_file_location("diffusion_moteur", cfg.MOTEURS_DIR / "diffusion.py")
moteur = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(moteur)  # importable sans torch

FAUX_MOTEUR = textwrap.dedent("""
    import json, os, sys
    from pathlib import Path
    import numpy as np, soundfile as sf
    action = sys.argv[1]
    journal = Path(os.environ["FAUX_JOURNAL"])
    if action == "telecharger":
        journal.open("a").write(json.dumps({"action": action, "args": sys.argv[2:]}) + chr(10))
        print("Modèles prêts.", flush=True); sys.exit(0)
    t = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    journal.open("a").write(json.dumps({"action": action, "tache": t, "cwd": os.getcwd(),
                                        "u2net": os.environ.get("U2NET_HOME")}) + chr(10))
    if action == "decrire":
        print("PROGRESSION 1/2 x", flush=True); print("PROGRESSION 2/2 y", flush=True)
        if t.get("image"): texte = "metal clink on a table"
        else: texte = "wooden door creaking then slamming" if t["mode"] == "bruitage" else "a red potion bottle"
        print("RESULTAT " + json.dumps({"texte": texte}), flush=True)
    elif action == "bruitage":
        if "ERREUR" in t["prompt"]:
            print("ERREUR : mémoire de la carte graphique insuffisante.", flush=True); sys.exit(3)
        print("PROGRESSION 1/2 x", flush=True); print("PROGRESSION 2/2 y", flush=True)
        d = Path(t["dossier"]); d.mkdir(parents=True, exist_ok=True); fichiers = []
        for i in range(1, t["variantes"] + 1):
            f = d / f"variante_{i}.wav"; sf.write(str(f), np.zeros((int(44100 * t["duree"]), 2), "float32"), 44100)
            fichiers.append(str(f))
        graines = [t["graine"] or 111] + [222, 333][: t["variantes"] - 1]
        print("RESULTAT " + json.dumps({"fichiers": fichiers, "graines": graines, "frequence": 44100}), flush=True)
    elif action == "image":
        print("PROGRESSION 1/2 x", flush=True); print("PROGRESSION 2/2 y", flush=True)
        from PIL import Image
        for s, g in zip(t["sorties"], t["graines"]):  # vrais PNG (gr.Image relit les fichiers), taille demandée / 16
            Path(s).parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (t["largeur"] // 16, t["hauteur"] // 16), (g % 256, 30, 30)).save(s)
        print("RESULTAT " + json.dumps({"fichiers": t["sorties"], "graines": t["graines"]}), flush=True)
    elif action == "forme3d":
        if not Path(t["image"]).exists() or "casse" in Path(t["image"]).read_text(errors="ignore"):
            print("ERREUR : image illisible", flush=True); sys.exit(2)
        n = 6 if t["texture"] else 4
        for i in range(1, n + 1): print(f"PROGRESSION {i}/{n} x", flush=True)
        d = Path(t["dossier"]); d.mkdir(parents=True, exist_ok=True)
        (d / "image_detouree.png").write_bytes(b"png"); (d / "forme.glb").write_bytes(b"glTF")
        res = {"forme": str(d / "forme.glb"), "faces": 1234, "graine": t["graine"] or 555, "texture": None, "obj": None}
        if t["texture"]:
            (d / "modele.glb").write_bytes(b"glTF"); res["texture"] = str(d / "modele.glb")
            if t.get("web"):
                (d / "modele_web.glb").write_bytes(b"g"); res["web"] = str(d / "modele_web.glb")
        if "obj" in t["formats"]:
            base = "modele" if t["texture"] else "forme"
            (d / f"{base}.obj").write_text("o x"); (d / "material.mtl").write_text("newmtl material_0")
            (d / "material_0.png").write_bytes(b"png"); res["obj"] = str(d / f"{base}.obj")
        print("RESULTAT " + json.dumps(res), flush=True)
    elif action == "alleger":
        print("PROGRESSION 1/1 x", flush=True)
        Path(t["sortie"]).write_bytes(b"g" * 100)
        textures = 1 if "modele" in Path(t["entree"]).name else 0
        print("RESULTAT " + json.dumps({"sortie": t["sortie"], "textures": textures, "avant": 4_000_000,
                                        "apres": 400_000}), flush=True)
    print("TERMINE -", flush=True)
""")


@pytest.fixture
def faux_diffusion(env, monkeypatch):
    moteurs = env / "moteurs"
    moteurs.mkdir(exist_ok=True)
    (moteurs / "diffusion.py").write_text(FAUX_MOTEUR, encoding="utf-8")
    monkeypatch.setattr(cfg, "MOTEURS_DIR", moteurs)
    monkeypatch.setattr(cfg, "DIFFUSION_PYTHON", sys.executable)
    monkeypatch.setattr(cfg, "DIFFUSION_DIR", env / "StudioVoix" / "diffusion")
    cfg.DIFFUSION_DIR.mkdir(parents=True)
    monkeypatch.setenv("HF_HOME", str(env / "StudioVoix" / "hf-home"))
    monkeypatch.delenv("U2NET_HOME", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("FAUX_JOURNAL", str(env / "journal.jsonl"))
    for nom in diffusion.MODELES:  # modèles « présents » dans le cache
        for f in diffusion.MODELES[nom][2]:
            depot, f = f if isinstance(f, tuple) else (diffusion.MODELES[nom][0], f)
            p = diffusion._dossier_depot(depot) / "snapshots" / "abc" / f
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"0")
    return env


def _journal():
    return [json.loads(l_) for l_ in (cfg.DATA_DIR.parent / "journal.jsonl").read_text().splitlines()]


def test_consignes_du_vrai_moteur():
    assert set(moteur.CONSIGNES) == {"son", "objet", "bruitage", "image", "carte"}
    assert set(moteur.MODELES) == set(diffusion.MODELES)  # mêmes noms côté application et côté moteur


def test_preparer_le_prompt(faux_diffusion):
    assert bruitages.preparer("une porte qui grince", None, progress=no_progress) == "wooden door creaking then slamming"
    im = faux_diffusion / "scene.png"
    im.write_bytes(b"png")
    assert bruitages.preparer("", str(im), progress=no_progress) == "metal clink on a table"
    j = _journal()
    assert j[0]["tache"] == {"mode": "bruitage", "texte": "une porte qui grince"}
    assert j[1]["tache"] == {"mode": "son", "image": str(im)} and j[1]["cwd"] == str(cfg.DIFFUSION_DIR)
    assert j[1]["u2net"] == str(cfg.DIFFUSION_DIR / "u2net")
    with pytest.raises(gr.Error, match="Décris"):
        bruitages.preparer("  ", None, progress=no_progress)


def test_generer_bruitage(faux_diffusion):
    etapes = []
    msg, liste, premiere, dossier = bruitages.generer("door creak", "Porte!", 2.5, 3, 42, 50, None, "une porte",
                                                      progress=lambda p, desc="": etapes.append(desc))
    d = next(cfg.SFX_DIR.iterdir())
    assert dossier == str(d) and premiere == str(d / "variante_1.wav") and len(liste["choices"]) == 3
    assert sf.info(premiere).duration == pytest.approx(2.5) and "Graines : 42, 222, 333" in msg
    infos = json.loads((d / "creation.json").read_text(encoding="utf-8"))
    assert infos["type"] == "bruitage" and infos["nom"] == "Porte" and infos["description_fr"] == "une porte"
    assert [v["graine"] for v in infos["versions"]] == [42, 222, 333]
    t = _journal()[0]["tache"]
    assert t["negatif"] == bruitages.NEGATIF and t["etapes"] == 50 and t["duree"] == 2.5
    assert any("génération du bruitage" in e for e in etapes)


def test_erreurs(faux_diffusion, monkeypatch):
    with pytest.raises(gr.Error, match="prompt"):
        bruitages.generer("", "x", 3, 1, 0, 100, progress=no_progress)
    with pytest.raises(gr.Error, match="Stable Audio : mémoire"):
        bruitages.generer("ERREUR", "x", 3, 1, 0, 100, progress=no_progress)
    (diffusion.ckpt_dir("bruitages") / "snapshots" / "abc" / "vae" / "diffusion_pytorch_model.safetensors").unlink()
    with pytest.raises(gr.Error, match="jeton"):
        bruitages.generer("x", "x", 3, 1, 0, 100, progress=no_progress)
    assert "Stable Audio Open 1.0 (bruitages)" in diffusion.missing_components()
    monkeypatch.setattr(cfg, "DIFFUSION_PYTHON", str(faux_diffusion / "absent.exe"))
    with pytest.raises(gr.Error, match="INSTALLER.bat"):
        bruitages.preparer("x", None, progress=no_progress)


def test_jeton_et_telechargement(faux_diffusion):
    assert not diffusion.jeton_present()
    with pytest.raises(gr.Error, match="hf_"):
        diffusion.enregistrer_jeton("abc")
    assert "enregistré" in diffusion.enregistrer_jeton("  hf_abcdefghijklmnopqrstuvwxyz ")
    assert (diffusion.hf_home() / "token").read_text() == "hf_abcdefghijklmnopqrstuvwxyz" and diffusion.jeton_present()
    (diffusion.hf_home() / "token").unlink()
    assert "❌" in list(diffusion.download(["bruitages"]))[0]  # sans jeton, pas de téléchargement lancé
    dernier = list(diffusion.download(["qwen", "zimage"]))[-1]
    assert "Terminé" in dernier and _journal()[-1]["args"] == ["qwen", "zimage", "detourage"]


def test_etat_des_modeles(faux_diffusion):
    assert "| Diffusion : Qwen3-VL, Stable Audio Open, Z-Image-Turbo, Hunyuan3D-2 | ✅ présent |" in models_status_md()


def test_galerie_bruitage(faux_diffusion):
    from studiovoix import galerie

    bruitages.generer("door creak", "Porte", 2, 2, 42, 50, None, "une porte", progress=no_progress)
    (lib, dossier), = galerie.lister("Bruitages")
    assert "🔊 Bruitage · Porte — une porte" in lib
    md, fichier, versions, *_ = galerie.details(dossier, 2)
    assert "Demande : une porte (2.0 s)" in md and fichier.endswith("variante_2.wav") and versions["visible"]
    msg, nouveau = galerie.recreer(dossier, 2, progress=no_progress)
    assert "graine 222" in msg and _journal()[-1]["tache"]["graine"] == 222 and _journal()[-1]["tache"]["variantes"] == 1
    with pytest.raises(gr.Error, match="chansons et les pistes"):
        galerie.refaire_passage(dossier, 1, 0, 1, "", "", "", progress=no_progress)


def test_modele_3d_depuis_une_image(faux_diffusion):
    im = faux_diffusion / "tasse.PNG"
    im.write_bytes(b"png")
    etapes = []
    msg, glb, detouree, fichiers, dossier = modele3d.generer(str(im), "Tasse!", "Fine (plus lente, plus de mémoire)",
                                                            True, 7, ["obj"], progress=lambda p, desc="": etapes.append(desc))
    d = next(cfg.MODELS3D_DIR.iterdir())
    assert dossier == str(d) and glb == str(d / "modele.glb") and detouree == str(d / "image_detouree.png")
    assert (d / "image.png").read_bytes() == b"png"  # image de départ copiée (extension normalisée)
    assert [Path(f).name for f in fichiers] == ["forme.glb", "material.mtl", "material_0.png", "modele.glb", "modele.obj"]
    assert "1234 faces, graine 7" in msg and "non peinte" not in msg
    infos = json.loads((d / "creation.json").read_text(encoding="utf-8"))
    assert infos["type"] == "3d" and infos["nom"] == "Tasse" and infos["image"] == "image.png"
    assert infos["octree"] == 384 and infos["faces"] == 100000 and infos["etapes"] == 5 and infos["texture"]
    assert infos["formats"] == ["glb", "obj"] and infos["versions"][0]["graine"] == 7
    t = _journal()[0]["tache"]
    assert t["formats"] == ["glb", "obj"] and t["octree"] == 384 and t["faces"] == 100000 and t["image"] == str(d / "image.png")
    assert any("peinture de la texture" in e for e in etapes)


def test_modele_3d_sans_texture_et_erreurs(faux_diffusion):
    im = faux_diffusion / "epee.jpg"
    im.write_bytes(b"jpg")
    msg, glb, _, fichiers, dossier = modele3d.generer(str(im), "", "Normale", False, 0, [], progress=no_progress)
    d = Path(dossier)
    assert glb == str(d / "forme.glb") and fichiers == [str(d / "forme.glb")] and "graine 555" in msg
    infos = json.loads((d / "creation.json").read_text(encoding="utf-8"))
    assert infos["nom"] == "modele" and infos["octree"] == 256 and not infos["texture"] and infos["formats"] == ["glb"]
    with pytest.raises(gr.Error, match="Importe une image"):
        modele3d.generer(None, "x", "Normale", True, 0, ["glb"], progress=no_progress)
    (faux_diffusion / "doc.txt").write_text("x")
    with pytest.raises(gr.Error, match="Format d'image"):
        modele3d.generer(str(faux_diffusion / "doc.txt"), "x", "Normale", True, 0, ["glb"], progress=no_progress)
    (diffusion.ckpt_dir("texture3d") / "snapshots" / "abc" / "hunyuan3d-paint-v2-0-turbo" / "unet"
     / "diffusion_pytorch_model.safetensors").unlink()
    with pytest.raises(gr.Error, match="Hunyuan3D-2 texture"):
        modele3d.generer(str(im), "x", "Normale", True, 0, ["glb"], progress=no_progress)
    *_, dossier = modele3d.generer(str(im), "x", "Normale", False, 0, ["glb"], progress=no_progress)  # forme seule : OK
    assert Path(dossier).is_dir()


def test_galerie_modele_3d(faux_diffusion):
    from studiovoix import galerie

    im = faux_diffusion / "tasse.png"
    im.write_bytes(b"png")
    modele3d.generer(str(im), "Tasse", "Normale", True, 7, ["glb"], progress=no_progress)
    (lib, dossier), = galerie.lister("Modèles 3D")
    assert "🧊 Modèle 3D · Tasse — depuis une image" in lib
    md, audio, versions, desc, paroles, fin, modele, _ = galerie.details(dossier)
    assert "qualité Normale, 1234 faces, texturé" in md and audio is None and not versions["visible"]
    assert modele["value"] == str(Path(dossier) / "modele.glb") and modele["visible"]
    msg, nouveau = galerie.recreer(dossier, 1, progress=no_progress)
    assert "graine 7" in msg and nouveau != dossier and _journal()[-1]["tache"]["graine"] == 7
    assert galerie.lire(nouveau)["nom"] == "Tasse" and (Path(nouveau) / "image.png").exists()
    with pytest.raises(gr.Error, match="chansons et les pistes"):
        galerie.refaire_passage(dossier, 1, 0, 1, "", "", "", progress=no_progress)
    assert "supprimée" in galerie.supprimer(dossier) and galerie.lister("Modèles 3D")[0][1] == nouveau


def test_texte_vers_3d(faux_diffusion):
    from studiovoix import galerie

    with pytest.raises(gr.Error, match="Décris l'objet"):
        modele3d.preparer_prompt(" ", progress=no_progress)
    assert modele3d.preparer_prompt("une potion rouge", progress=no_progress) == "a red potion bottle"
    with pytest.raises(gr.Error, match="prompt de l'image"):
        modele3d.generer_image("", 0, progress=no_progress)
    image, graine, msg = modele3d.generer_image("a red potion bottle", 12, progress=no_progress)
    assert graine == 12 and "graine 12" in msg and Path(image).read_bytes()[:4] == b"\x89PNG"
    assert Path(image).parent.parent == cfg.MODELS3D_DIR / "images" and Path(image).name == "objet.png"
    assert (Path(image).parent / "prompt.txt").read_text(encoding="utf-8") == "a red potion bottle"
    t = _journal()[-1]
    assert t["action"] == "image" and t["tache"]["etapes"] == 9 and t["tache"]["graines"] == [12]
    assert t["tache"]["prompt"].startswith("a red potion bottle, single object") and t["tache"]["largeur"] == 1024
    # le dossier des images n'est pas une création : la galerie l'ignore
    assert galerie.lister("Modèles 3D") == []
    *_, dossier = modele3d.generer(image, "Potion", "Normale", True, 0, ["glb"], "a red potion bottle",
                                   "une potion rouge", graine, progress=no_progress)
    infos = json.loads((Path(dossier) / "creation.json").read_text(encoding="utf-8"))
    assert infos["description"] == "a red potion bottle" and infos["description_fr"] == "une potion rouge"
    assert infos["image_graine"] == 12 and (Path(dossier) / "image.png").exists()
    (lib, chemin), = galerie.lister("Modèles 3D")
    assert chemin == dossier and "Potion — une potion rouge" in lib
    md, *_ = galerie.details(dossier)
    assert "demande : une potion rouge" in md


def test_illustrations_de_cartes(faux_diffusion):
    from studiovoix import cartes, galerie

    assert cartes.preparer("un chevalier", progress=no_progress) == "a red potion bottle"
    assert _journal()[-1]["tache"] == {"mode": "carte", "texte": "un chevalier"}
    with pytest.raises(gr.Error, match="Décris la scène"):
        cartes.preparer(" ", progress=no_progress)
    with pytest.raises(gr.Error, match="prompt de la scène"):
        cartes.generer("Mon Jeu", "x", "", [], "", None, 1, 0, progress=no_progress)
    styles = ["16-bit pixel art, limited palette, crisp pixels", "violet et or"]  # un choix de liste + une saisie libre
    msg, images, dossier, projets = cartes.generer("Mon Jeu!", "Chevalier", "a golden knight.", styles, "soft rim light",
                                                   "Carte entière (portrait 2:3)", 3, 42, True, "un chevalier",
                                                   progress=no_progress)
    d = Path(dossier)
    assert d.parent == cfg.CARDS_DIR / "Mon Jeu" and projets["value"] == "Mon Jeu" and "Mon Jeu" in projets["choices"]
    assert [Path(f).name for f, _ in images] == ["variante_1.png", "variante_2.png", "variante_3.png"]
    assert images[0][1] == "Variante 1 (graine 42)" and "832×1248" in msg
    assert all(Path(f).with_suffix(".webp").exists() for f, _ in images)
    t = _journal()[-1]["tache"]
    assert (t["largeur"], t["hauteur"], t["etapes"]) == (832, 1248, 9) and t["graines"][0] == 42 and len(t["graines"]) == 3
    assert t["prompt"] == ("a golden knight. Art style: 16-bit pixel art, limited palette, crisp pixels, violet et or, "
                           "soft rim light. No text, no letters, no frame.")
    # le style est mémorisé pour le projet, et rechargé quand on revient au projet
    assert cartes.style_projet("Mon Jeu") == (styles, "soft rim light", "Carte entière (portrait 2:3)")
    maj, consignes, fmt = cartes.charger_style("Mon Jeu")
    assert maj["value"] == styles and consignes == "soft rim light" and fmt == "Carte entière (portrait 2:3)"
    assert cartes.style_projet("nouveau")[0] == cartes.STYLES_DEFAUT
    # galerie : image affichée, recréation avec la graine d'une variante
    (lib, chemin), = galerie.lister("Illustrations")
    assert chemin == dossier and "🃏 Illustration · Mon Jeu — Chevalier — un chevalier" in lib
    md, audio, versions, *_, modele, image = galerie.details(chemin, 2)
    assert "Projet **Mon Jeu**, carte **Chevalier**, 832×1248" in md and audio is None
    assert image["value"].endswith("variante_2.png") and image["visible"] and not modele["visible"]
    graine_v2 = t["graines"][1]
    msg, nouveau = galerie.recreer(chemin, 2, progress=no_progress)
    assert nouveau != chemin and _journal()[-1]["tache"]["graines"] == [graine_v2]
    assert galerie.lire(nouveau)["styles"] == styles and len(galerie.lister("Illustrations")) == 2
    assert not any(p.name == "style.json" for p in Path(nouveau).iterdir())  # style au niveau du projet seulement


def test_modele_3d_web_et_allegement(faux_diffusion, monkeypatch):
    from studiovoix import serveur_acestep

    im = faux_diffusion / "epee.png"
    im.write_bytes(b"png")
    msg, glb, _, fichiers, dossier = modele3d.generer(str(im), "Épée", "Web léger (jeu dans le navigateur)", True, 3,
                                                      ["glb"], progress=no_progress)
    t = _journal()[-1]["tache"]
    assert t["web"] is True and t["faces"] == 10000 and t["octree"] == 192
    assert "version web légère : modele_web.glb" in msg and any(f.endswith("modele_web.glb") for f in fichiers)
    infos = json.loads((Path(dossier) / "creation.json").read_text(encoding="utf-8"))
    assert infos["web"] is True and infos["versions"][0]["web"].endswith("modele_web.glb")
    liberations = []
    monkeypatch.setattr(serveur_acestep, "liberer_gpu", lambda progress=None: liberations.append(1))
    msg, fichiers = modele3d.alleger(dossier, progress=no_progress)
    assert "modele_web.glb : 4.0 Mo → 400 Ko" in msg and liberations == []  # sur le processeur : ACE-Step reste
    assert _journal()[-1]["tache"]["entree"].endswith("modele.glb")
    # forme seule : rien à réduire
    *_, d2 = modele3d.generer(str(im), "x", "Normale", False, 0, ["glb"], progress=no_progress)
    msg, _ = modele3d.alleger(d2, progress=no_progress)
    assert "forme_web.glb" in msg and "pas de texture" in msg
    with pytest.raises(gr.Error, match="Crée d'abord"):
        modele3d.alleger(None, progress=no_progress)


def test_modeles_3d_par_lot(faux_diffusion):
    images = []
    for nom, contenu in (("tasse.png", b"png"), ("casse.png", b"casse"), ("epee.jpg", b"jpg")):
        (faux_diffusion / nom).write_bytes(contenu)
        images.append(str(faux_diffusion / nom))
    rapport, dernier, dossier = modele3d.generer_lot(images, "Carte", "Aperçu (rapide)", False, 0, ["glb"],
                                                     progress=no_progress)
    assert "**2/3 modèle(s) créé(s)**" in rapport and "❌ casse.png" in rapport and "image illisible" in rapport
    assert "« Carte_tasse »" in rapport and "« Carte_epee »" in rapport
    assert dernier.endswith("forme.glb") and Path(dossier).is_dir()
    assert len([d for d in cfg.MODELS3D_DIR.iterdir() if (d / "creation.json").exists()]) == 2
    with pytest.raises(gr.Error, match="Ajoute des images"):
        modele3d.generer_lot([], "x", "Normale", False, 0, ["glb"], progress=no_progress)
