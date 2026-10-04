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
    elif action in ("image", "personnage"):
        if action == "personnage": assert all(Path(r).exists() for r in t["references"])
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
    elif action == "video":
        from PIL import Image
        if "ECHEC" in t["prompt"]:
            print("ERREUR : mémoire de la carte graphique insuffisante.", flush=True); sys.exit(3)
        n = t["etapes"] + 3
        for i in range(1, n + 1): print(f"PROGRESSION {i}/{n} x", flush=True)
        Path(t["sortie"]).write_bytes(b"mp4")
        derniere = str(Path(t["sortie"]).with_name("derniere_image.png"))
        Image.new("RGB", (t["largeur"] // 16, t["hauteur"] // 16), (9, 9, 9)).save(derniere)
        print("RESULTAT " + json.dumps({"sortie": t["sortie"], "graine": t["graine"], "images": t["images"],
                                        "duree": round(t["images"] / 24, 2), "largeur": t["largeur"],
                                        "hauteur": t["hauteur"], "derniere_image": derniere}), flush=True)
    elif action == "assembler":
        Path(t["sortie"]).write_bytes(b"".join(Path(c).read_bytes() for c in t["clips"]))
        print("RESULTAT " + json.dumps({"sortie": t["sortie"], "images": 49 * len(t["clips"]),
                                        "duree": 2.0 * len(t["clips"])}), flush=True)
    elif action == "detourer":
        from PIL import Image
        for i in (1, 2, 3): print(f"PROGRESSION {i}/3 x", flush=True)
        im = Image.open(t["entree"]).convert("RGBA" if not t["fond"] else "RGB")
        Path(t["sortie"]).parent.mkdir(parents=True, exist_ok=True); im.save(t["sortie"])
        if t["masque"]: Image.new("L", im.size, 255).save(t["masque"])
        couverture = 0.0 if im.width < 10 else 0.5  # image minuscule : « sujet non reconnu »
        print("RESULTAT " + json.dumps({"sortie": t["sortie"], "masque": t["masque"], "largeur": im.width,
                                        "hauteur": im.height, "couverture": couverture}), flush=True)
    elif action == "ameliorer":
        from PIL import Image
        for i in (1, 2, 3): print(f"PROGRESSION {i}/3 x", flush=True)
        im = Image.open(t["entree"]); im = im.resize((im.width * t["echelle"], im.height * t["echelle"]))
        im.save(t["sortie"])
        print("RESULTAT " + json.dumps({"sortie": t["sortie"], "largeur": im.width, "hauteur": im.height,
                                        "visages": 1 if t["visages"] else 0, "modele": "x"}), flush=True)
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
            p = (diffusion.dossier_photos() / f if depot == diffusion.LOCAL_PHOTOS
                 else diffusion._dossier_depot(depot) / "snapshots" / "abc" / f)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"0")
    return env


def _journal():
    return [json.loads(l_) for l_ in (cfg.DATA_DIR.parent / "journal.jsonl").read_text().splitlines()]


def test_consignes_du_vrai_moteur():
    assert set(moteur.CONSIGNES) == {"son", "objet", "bruitage", "image", "carte", "video", "scene", "retouche", "histoire"}
    # le texte de l'utilisateur est ajouté à toutes les consignes de reformulation (oubli constaté pour « video »)
    assert moteur.MODES_TEXTE == {"objet", "bruitage", "carte", "video", "scene", "retouche", "histoire"}
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
    list(diffusion.download(["personnages"]))  # Z-Image présent : seul klein est téléchargé
    assert _journal()[-1]["args"] == ["personnages", "detourage"]
    import shutil
    shutil.rmtree(diffusion.ckpt_dir("zimage"))  # klein reprend l'encodeur de texte de Z-Image : téléchargé avec
    list(diffusion.download(["personnages"]))
    assert _journal()[-1]["args"] == ["personnages", "zimage", "detourage"]


def test_etat_des_modeles(faux_diffusion):
    assert "| Diffusion : Qwen3-VL, Stable Audio Open, Z-Image-Turbo, FLUX.2 klein, Wan 2.2, BiRefNet, Real-ESRGAN, GFPGAN, Hunyuan3D-2 | ✅ présent |" in models_status_md()


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
    md, audio, versions, desc, paroles, fin, modele, _, _ = galerie.details(dossier)
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
    md, audio, versions, *_, modele, image, video = galerie.details(chemin, 2)
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


def test_personnages_recurrents(faux_diffusion):
    from PIL import Image

    from studiovoix import cartes, galerie, personnages

    assert personnages.liste("Mon Jeu") == [] and personnages.choix("Mon Jeu")["choices"] == [personnages.AUCUN]
    with pytest.raises(gr.Error, match="nom"):
        personnages.ajouter("Mon Jeu", " ", ["x.png"])
    with pytest.raises(gr.Error, match="au moins une image"):
        personnages.ajouter("Mon Jeu", "Héroïne", [])
    images = []
    for i in range(5):
        f = cfg.DATA_DIR.parent / f"ref{i}.jpg"
        Image.new("RGB", (2048, 1024), (40 * i, 0, 0)).save(f)
        images.append(str(f))
    msg, gal, liste = personnages.ajouter("Mon Jeu", "Héroïne", images[:2])
    assert "2 image(s) ajoutée(s)" in msg and liste["value"] == "Héroïne" and liste["choices"][1:] == ["Héroïne"]
    with Image.open(gal[0][0]) as ref1:  # fermée : sinon Windows refuse la suppression plus bas
        taille = ref1.size
    assert taille == (1024, 512) and Path(gal[0][0]).name == "ref_1.png"  # réduite, en PNG
    # même nom à la casse près (comme Windows) : même personnage ; 4 références au plus
    msg, gal, _ = personnages.ajouter("Mon Jeu", "HÉROÏNE", images[2:])
    assert "2 image(s) ajoutée(s)" in msg and "1 image(s) ignorée(s)" in msg and len(gal) == 4
    with pytest.raises(gr.Error, match="déjà 4"):
        personnages.ajouter("Mon Jeu", "héroïne", images[:1])
    assert personnages.liste("Mon Jeu") == ["Héroïne"] and personnages.liste("Autre") == []

    # génération avec le personnage : FLUX.2 klein reçoit ses références, 4 pas
    with pytest.raises(gr.Error, match="pas d'image de référence"):
        cartes.generer("Mon Jeu", "x", "a scene", [], "", None, 1, 0, personnage="Inconnu", progress=no_progress)
    msg, imgs, dossier, _ = cartes.generer("Mon Jeu", "Héroïne au combat", "she fights a dragon", ["anime style"], "",
                                           None, 2, 7, personnage="Héroïne", progress=no_progress)
    j = _journal()[-1]
    assert j["action"] == "personnage" and "avec le personnage « Héroïne »" in msg
    assert [Path(r).name for r in j["tache"]["references"]] == ["ref_1.png", "ref_2.png", "ref_3.png", "ref_4.png"]
    assert j["tache"]["etapes"] == 4 and j["tache"]["graines"][0] == 7
    assert j["tache"]["prompt"].startswith("she fights a dragon. " + cartes.MEME_PERSONNAGE + ". Art style: anime")
    infos = galerie.lire(dossier)
    assert infos["personnage"] == "Héroïne" and infos["moteur"] == "FLUX.2 klein 4B"
    assert "personnage **Héroïne**" in galerie.details(dossier, 1)[0]
    galerie.recreer(dossier, 2, progress=no_progress)  # la recréation repasse par le personnage
    assert _journal()[-1]["action"] == "personnage"
    # sans personnage : Z-Image-Turbo, comme avant
    cartes.generer("Mon Jeu", "Décor", "a castle", [], "", None, 1, 0, personnage=personnages.AUCUN,
                   progress=no_progress)
    assert _journal()[-1]["action"] == "image" and galerie.lire(_journal()[-1]["tache"]["sorties"][0].rsplit("variante", 1)[0])["personnage"] is None
    # klein absent : message clair
    import shutil
    shutil.rmtree(diffusion.ckpt_dir("personnages"))
    with pytest.raises(gr.Error, match="FLUX.2 klein"):
        cartes.generer("Mon Jeu", "x", "a scene", [], "", None, 1, 0, personnage="Héroïne", progress=no_progress)
    # suppression (annulée, puis confirmée)
    assert "annulée" in personnages.supprimer("Mon Jeu", None)[0]
    msg, gal, liste = personnages.supprimer("Mon Jeu", "héroïne")
    assert "supprimé" in msg and gal == [] and personnages.liste("Mon Jeu") == [] and liste["value"] == personnages.AUCUN
    assert len(galerie.lister("Illustrations")) == 3  # les illustrations faites avec lui restent


def test_photos(faux_diffusion):
    from PIL import Image

    from studiovoix import galerie, photos

    with pytest.raises(gr.Error, match="photo"):
        photos.traiter(None, photos.ACTION_DEFAUT, "×2", False, True, 0.7, photos.FOND_DEFAUT, None)
    source = cfg.DATA_DIR.parent / "vacances.jpg"
    Image.new("RGB", (40, 30), (200, 100, 50)).save(source)
    (cfg.DATA_DIR.parent / "notes.txt").write_text("x")
    with pytest.raises(gr.Error, match="Format"):
        photos.traiter(str(cfg.DATA_DIR.parent / "notes.txt"), photos.ACTION_DEFAUT, "×2", False, True, 0.7,
                       photos.FOND_DEFAUT, None)
    # amélioration ×4 avec visages
    msg, (avant, apres), fichiers, dossier = photos.traiter(str(source), photos.ACTION_DEFAUT, "×4", True, True, 0.6,
                                                           photos.FOND_DEFAUT, None, progress=no_progress)
    t = _journal()[-1]
    assert t["action"] == "ameliorer" and t["tache"]["echelle"] == 4 and t["tache"]["rapide"] is True
    assert t["tache"]["force"] == 0.6 and t["cwd"] == str(cfg.DIFFUSION_DIR)
    assert "160×120" in msg and "1 visage(s) restauré(s)" in msg
    assert Path(avant).name == "originale.jpg" and Path(apres).name == "resultat.png" and fichiers == [apres]
    # détourage sur une couleur au choix (le sélecteur peut renvoyer « rgba(…) »), puis personne isolée
    msg, _, fichiers, d2 = photos.traiter(apres, "✂️ Détourer (objet, animal, personne…)", "×2", False, True, 0.7,
                                          "Couleur au choix", "rgba(255, 0, 16, 1)", "chat", progress=no_progress)
    t = _journal()[-1]["tache"]
    assert t["modele"] == "general" and t["fond"] == "#ff0010" and "Photo détourée" in msg
    assert [Path(f).name for f in fichiers] == ["resultat.png", "masque.png"]
    assert Path(photos.continuer(d2)).name == "resultat.png"
    _, _, _, d3 = photos.traiter(str(source), "🧍 Isoler une personne", "×2", False, True, 0.7,
                                 "Fond d'origine flouté (effet portrait)", None, progress=no_progress)
    assert _journal()[-1]["tache"]["modele"] == "personne" and _journal()[-1]["tache"]["fond"] == "flou"
    vide = cfg.DATA_DIR.parent / "vide.png"
    Image.new("RGB", (8, 8)).save(vide)
    assert "Presque rien" in photos.traiter(str(vide), "🧍 Isoler une personne", "×2", False, True, 0.7,
                                            photos.FOND_DEFAUT, None, progress=no_progress)[0]
    assert photos.maj_action("🧍 Isoler une personne") == (gr.update(visible=False), gr.update(visible=True))
    # galerie : image affichée, nouveau traitement avec les mêmes réglages
    assert len(galerie.lister("Photos")) == 4
    md, audio, *_, image, _ = galerie.details(d2)
    assert "🖼️ Photo" in md and "détourée (fond : Couleur au choix)" in md and image["value"].endswith("resultat.png")
    msg, nouveau = galerie.recreer(dossier, progress=no_progress)
    t = _journal()[-1]["tache"]
    assert nouveau != dossier and t["echelle"] == 4 and t["rapide"] is True and t["entree"].startswith(nouveau)
    # modèles absents : message clair
    import shutil
    shutil.rmtree(diffusion.dossier_photos())
    with pytest.raises(gr.Error, match="Real-ESRGAN"):
        photos.traiter(str(source), photos.ACTION_DEFAUT, "×2", False, True, 0.7, photos.FOND_DEFAUT, None,
                       progress=no_progress)


def test_videos(faux_diffusion):
    from PIL import Image

    from studiovoix import galerie, videos

    assert videos.preparer("un dragon s'envole", progress=no_progress) == "a red potion bottle"
    assert _journal()[-1]["tache"] == {"mode": "video", "texte": "un dragon s'envole"}
    # avec une image de départ, Qwen la voit (sinon il inventait les couleurs)
    videos.prompt_pret("un dragon s'envole", "", "depart.png", progress=no_progress)
    assert _journal()[-1]["tache"] == {"mode": "video", "texte": "un dragon s'envole", "image": "depart.png"}
    with pytest.raises(gr.Error, match="prompt"):
        videos.generer(" ", None, videos.AUTO, "5 s", 30, 0, progress=no_progress)
    # texte → vidéo : format automatique sans image = paysage 720p ; 5 s = 121 images
    msg, fichier, dossier = videos.generer("a dragon flies", None, videos.AUTO, "5 s", 30, 7, "dragon", "un dragon",
                                           progress=no_progress)
    t = _journal()[-1]
    assert t["action"] == "video" and t["tache"]["image"] is None
    assert (t["tache"]["largeur"], t["tache"]["hauteur"], t["tache"]["images"], t["tache"]["etapes"]) == (1280, 704, 121, 30)
    assert t["tache"]["graine"] == 7 and Path(fichier).name == "video.mp4" and "à partir du texte" in msg
    # image → vidéo : format d'après l'image (portrait), copie de l'image de départ ; enchaînement par la dernière image
    depart = cfg.DATA_DIR.parent / "carte.png"
    Image.new("RGB", (600, 900), (1, 2, 3)).save(depart)
    msg, _, d2 = videos.generer("she smiles", str(depart), videos.AUTO, "2 s", 20, 0, progress=no_progress)
    t = _journal()[-1]["tache"]
    assert (t["largeur"], t["hauteur"], t["images"]) == (704, 1280, 49) and t["graine"] > 0
    assert Path(t["image"]).parent == Path(d2) and Path(t["image"]).name == "image_depart.png"
    assert "à partir de l'image" in msg and Path(videos.continuer(d2)).name == "derniere_image.png"
    assert videos.format_pour("Carré (960×960)", str(depart)) == (960, 960)
    # galerie : lecteur vidéo, recréation avec la même graine et la même image
    md, audio, *_, modele, image, video = galerie.details(d2)
    assert "🎬 Vidéo" in md and "à partir d'une image" in md and video["visible"] and video["value"].endswith("video.mp4")
    assert not image["visible"] and audio is None and len(galerie.lister("Vidéos")) == 2
    graine = galerie.lire(d2)["versions"][0]["graine"]
    _, nouveau = galerie.recreer(d2, progress=no_progress)
    t = _journal()[-1]["tache"]
    assert t["graine"] == graine and (t["largeur"], t["hauteur"], t["images"]) == (704, 1280, 49)
    assert Path(t["image"]).parent == Path(nouveau)
    import shutil
    shutil.rmtree(diffusion.ckpt_dir("video"))
    with pytest.raises(gr.Error, match="Wan 2.2"):
        videos.generer("x", None, videos.AUTO, "2 s", 20, 0, progress=no_progress)


def test_essai_des_liens_avant_telechargement(tmp_path, monkeypatch):
    """Course de huggingface_hub sous Windows sans mode développeur (WinError 1314 constatée) : l'essai des liens
    symboliques est fait avant snapshot_download, sur le dossier du dépôt, et son résultat est gardé."""
    import os

    from huggingface_hub import constants
    from huggingface_hub import file_download as fd

    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(constants, "HF_HUB_DISABLE_SYMLINKS_WARNING", True)

    def refus(*a, **k):
        raise OSError(22, "[WinError 1314] Le client ne dispose pas d'un privilège nécessaire")

    monkeypatch.setattr(os, "symlink", refus)
    assert moteur._tester_liens("ZhengPeng7/BiRefNet_HR-matting") is False
    dossier = str((tmp_path / "models--ZhengPeng7--BiRefNet_HR-matting").resolve())
    assert fd._are_symlinks_supported_in_dir[dossier] is False  # les fils de snapshot_download liront « non »


def test_reprises_apres_coupure_reseau():
    """Coupure constatée chez l'utilisateur (VAE de Wan) : ChunkedEncodingError causée par une ProtocolError ;
    le téléchargement est relancé (huggingface_hub reprend le fichier partiel), jamais un refus d'accès."""
    import requests
    import urllib3

    def coupure():
        try:
            raise urllib3.exceptions.ProtocolError("Connection broken: IncompleteRead")
        except urllib3.exceptions.ProtocolError as e:
            raise requests.exceptions.ChunkedEncodingError(e) from e

    def http(statut):
        reponse = requests.models.Response()
        reponse.status_code = statut
        return requests.exceptions.HTTPError(response=reponse)

    for exc in (requests.exceptions.ConnectionError(), TimeoutError(), http(503)):
        assert moteur._erreur_reseau(exc)
    for exc in (http(401), http(404), ValueError("fichier corrompu"), KeyError("x")):
        assert not moteur._erreur_reseau(exc)
    try:
        coupure()
    except Exception as e:  # noqa: BLE001
        assert moteur._erreur_reseau(e)

    essais, pauses = [], []

    def deux_coupures():
        essais.append(1)
        if len(essais) <= 2:
            coupure()
        return "fini"

    assert moteur._avec_reprises(deux_coupures, "vae", pause=pauses.append) == "fini"
    assert len(essais) == 3 and pauses == [5, 10]
    with pytest.raises(requests.exceptions.HTTPError):  # refus d'accès : pas de nouvel essai
        moteur._avec_reprises(lambda: (_ for _ in ()).throw(http(401)), "x", pause=pauses.append)
    assert pauses == [5, 10]
    with pytest.raises(requests.exceptions.ChunkedEncodingError):  # coupures sans fin : abandon après 6 essais
        moteur._avec_reprises(coupure, "x", pause=pauses.append)
    assert pauses == [5, 10, 5, 10, 20, 40, 60]


def test_photo_modele_pour_les_cartes(faux_diffusion):
    from PIL import Image

    from studiovoix import cartes, galerie, personnages

    photo = cfg.DATA_DIR.parent / "moi.jpg"
    Image.new("RGB", (300, 400), (90, 60, 40)).save(photo)
    with pytest.raises(gr.Error, match="introuvable"):
        cartes.generer("Mon Jeu", "x", "a scene", [], "", None, 1, 0, photo=str(photo) + "x", progress=no_progress)
    # sujet de la photo → FLUX.2 klein avec la photo (copiée dans la création) comme seule référence
    msg, _, dossier, _ = cartes.generer("Mon Jeu", "Paladin", "a paladin on a castle wall", ["oil painting"], "", None,
                                        1, 5, photo=str(photo), progress=no_progress)
    j = _journal()[-1]
    assert j["action"] == "personnage" and [Path(r).name for r in j["tache"]["references"]] == ["photo_modele.jpg"]
    assert Path(j["tache"]["references"][0]).parent == Path(dossier)
    assert cartes.PHOTO_SUJET in j["tache"]["prompt"] and "d'après le sujet de la photo modèle" in msg
    infos = galerie.lire(dossier)
    assert infos["photo_modele"] == "photo_modele.jpg" and infos["usage_photo"] == "sujet"
    assert "d'après une photo modèle" in galerie.details(dossier)[0]
    # recréation : même photo (la copie), même usage
    galerie.recreer(dossier, progress=no_progress)
    j = _journal()[-1]
    assert j["action"] == "personnage" and cartes.PHOTO_SUJET in j["tache"]["prompt"]
    # composition, et avec un personnage : la photo s'ajoute après ses références et ne sert qu'à la pose
    cartes.generer("Mon Jeu", "x", "a scene", [], "", None, 1, 0, photo=str(photo),
                   usage_photo="Garder la composition de la photo (pose, cadrage)", progress=no_progress)
    assert cartes.PHOTO_COMPOSITION in _journal()[-1]["tache"]["prompt"]
    refs = [cfg.DATA_DIR.parent / f"r{i}.png" for i in range(4)]
    for r in refs:
        Image.new("RGB", (64, 64)).save(r)
    personnages.ajouter("Mon Jeu", "Héros", [str(r) for r in refs])
    cartes.generer("Mon Jeu", "x", "a scene", [], "", None, 1, 0, personnage="Héros", photo=str(photo),
                   progress=no_progress)
    t = _journal()[-1]["tache"]
    assert [Path(r).name for r in t["references"]] == ["ref_1.png", "ref_2.png", "ref_3.png", "photo_modele.jpg"]
    assert cartes.MEME_PERSONNAGE in t["prompt"] and cartes.PHOTO_COMPOSITION in t["prompt"]
    assert cartes.PHOTO_SUJET not in t["prompt"]
    # sans photo ni personnage : Z-Image-Turbo, comme avant
    cartes.generer("Mon Jeu", "x", "a scene", [], "", None, 1, 0, progress=no_progress)
    assert _journal()[-1]["action"] == "image"


def test_prompt_prepare_automatiquement(faux_diffusion):
    """« Générer » sans « Préparer le prompt » (constaté : « Il manque le prompt ») : le prompt est préparé d'abord ;
    un prompt déjà là n'est jamais remplacé."""
    from studiovoix import bruitages, cartes, modele3d, videos

    n = len(_journal()) if (cfg.DATA_DIR.parent / "journal.jsonl").exists() else 0
    assert videos.prompt_pret("un dragon s'envole", " ", progress=no_progress) == "a red potion bottle"
    assert _journal()[-1]["tache"] == {"mode": "video", "texte": "un dragon s'envole"}
    assert cartes.prompt_pret("un chevalier", "", progress=no_progress) == "a red potion bottle"
    assert _journal()[-1]["tache"]["mode"] == "carte"
    assert bruitages.prompt_pret("une porte qui grince", None, None, progress=no_progress).startswith("wooden door")
    assert modele3d.prompt_pret("une potion", "", progress=no_progress) == "a red potion bottle"
    nb = len(_journal())
    assert nb == n + 4
    # prompt déjà préparé ou retouché : gardé tel quel, sans appel au moteur
    assert videos.prompt_pret("autre chose", " my own prompt ", progress=no_progress) == "my own prompt"
    assert cartes.prompt_pret("", "kept", progress=no_progress) == "kept"
    assert len(_journal()) == nb
    with pytest.raises(gr.Error, match="Décris la vidéo"):
        videos.prompt_pret("", "", progress=no_progress)
    # constaté : des plans tapés dans la case du prompt, envoyés tels quels à Wan (vidéo sans aucun rapport)
    with pytest.raises(gr.Error, match="Plusieurs plans à la suite"):
        videos.prompt_pret("", "aventurers are fighting\nA kreken apears\nThey are fighting it with magic.",
                           progress=no_progress)
    assert len(_journal()) == nb
    # prompt court tapé à la main, sans description : enrichi par Qwen d'abord
    assert videos.prompt_pret("", "aventurers are fighting", progress=no_progress) == "a red potion bottle"
    assert _journal()[-1]["tache"] == {"mode": "video", "texte": "aventurers are fighting"}
    long_ = " ".join(["word"] * 40)  # prompt détaillé écrit à la main : gardé
    assert videos.prompt_pret("", long_, progress=no_progress) == long_


def test_grande_photo_agrandie_dans_la_limite(faux_diffusion, monkeypatch):
    from PIL import Image

    from studiovoix import photos

    source = cfg.DATA_DIR.parent / "appareil.jpg"
    Image.new("RGB", (60, 90), (1, 2, 3)).save(source)
    # le moteur a limité l'agrandissement (photo de 24 Mpx : ×2 donnerait 6912×10368) : le message le dit
    monkeypatch.setattr(diffusion, "ameliorer", lambda *a, **k: {"sortie": a[1], "largeur": 5461, "hauteur": 8192,
                                                                "visages": 0, "echelle_obtenue": 1.58})
    msg = photos.traiter(str(source), photos.ACTION_DEFAUT, "×2", False, False, 0.7, photos.FOND_DEFAUT, None,
                         progress=no_progress)[0]
    assert "5461×8192 (agrandie ×1,58 au lieu de ×2 : 8192 pixels de côté au plus)" in msg


def test_modele_3d_qualite_maximale(faux_diffusion):
    """Qualité « Maximale » : modèle de forme complet de Hunyuan3D-2 (non distillé), 50 étapes."""
    im = faux_diffusion / "tasse.png"
    im.write_bytes(b"png")
    *_, dossier = modele3d.generer(str(im), "Tasse", "Maximale (modèle complet, 50 étapes, la plus détaillée)", False,
                                   0, ["glb"], progress=no_progress)
    t = _journal()[-1]["tache"]
    assert t["sous_dossier"] == "hunyuan3d-dit-v2-0" and t["etapes"] == 50
    infos = json.loads((Path(dossier) / "creation.json").read_text(encoding="utf-8"))
    assert infos["complet"] is True and infos["etapes"] == 50
    modele3d.generer(str(im), "Tasse", "Normale", False, 0, ["glb"], progress=no_progress)
    assert "sous_dossier" not in _journal()[-1]["tache"]  # turbo par défaut


def test_videos_plans_a_la_suite(faux_diffusion):
    """Plusieurs plans : prompts préparés d'abord, chaque clip part de la dernière image du précédent, assemblage."""
    from PIL import Image

    from studiovoix import galerie, videos

    depart = cfg.DATA_DIR.parent / "pont.png"
    Image.new("RGB", (1200, 700), (1, 2, 3)).save(depart)
    plans = "les aventuriers avancent sur le pont\n\n  un kraken surgit de l'eau  \nle mage lance une boule de feu"
    msg, fichier, dossier = videos.generer_suite(plans, str(depart), videos.AUTO, "2 s", 20, 5, "Kraken",
                                                 progress=no_progress)
    j = _journal()
    assert [e["tache"]["mode"] for e in j if e["action"] == "decrire"] == ["video"] * 3  # avant toute vidéo
    # chaque plan est écrit avec toute l'histoire et le prompt du précédent (sinon : décors et personnages changeaient)
    suites = [e["tache"]["suite"] for e in j if e["action"] == "decrire"]
    assert [s_["indice"] for s_ in suites] == [1, 2, 3] and suites[0]["precedent"] is None
    assert suites[2]["plans"] == ["les aventuriers avancent sur le pont", "un kraken surgit de l'eau",
                                  "le mage lance une boule de feu"]
    prompts = [p["prompt"] for p in galerie.lire(dossier)["plans"]]
    assert suites[1]["precedent"] == prompts[0] and suites[2]["precedent"] == prompts[1]
    assert " en " in msg and " s, dans " in msg  # temps de génération affiché
    assert [e["action"] for e in j][-4:] == ["video", "video", "video", "assembler"]
    clips = [e["tache"] for e in j if e["action"] == "video"]
    d = Path(dossier)
    assert Path(clips[0]["image"]) == d / "image_depart.png"
    assert Path(clips[1]["image"]) == d / "plan_1" / "derniere_image.png"  # dernière image du plan précédent
    assert Path(clips[2]["image"]) == d / "plan_2" / "derniere_image.png"
    assert [c["graine"] for c in clips] == [5, 6, 7] and {(c["largeur"], c["hauteur"]) for c in clips} == {(1280, 704)}
    assert j[-1]["tache"]["clips"] == [str(d / f"plan_{i}" / "video.mp4") for i in (1, 2, 3)]
    assert Path(fichier) == d / "video.mp4" and (d / "derniere_image.png").exists() and "3 plan(s)" in msg
    infos = galerie.lire(dossier)
    assert infos["type"] == "video" and [p["plan"] for p in infos["plans"]][1] == "un kraken surgit de l'eau"
    assert infos["image_depart"] == "image_depart.png"
    # recréer depuis la galerie : mêmes prompts (pas de nouvelle préparation), même graine
    avant = len([e for e in _journal() if e["action"] == "decrire"])
    _, nouveau = galerie.recreer(dossier, progress=no_progress)
    assert len([e for e in _journal() if e["action"] == "decrire"]) == avant
    assert [e["tache"]["graine"] for e in _journal() if e["action"] == "video"][-3:] == [5, 6, 7]
    # un plan qui échoue : les précédents sont gardés et assemblés
    msg, _, d3 = videos.generer_suite(None, None, videos.AUTO, "2 s", 20, 1, prompts=["ok", "ECHEC", "jamais"],
                                      progress=no_progress)
    assert "1 plan(s)" in msg and "Arrêt au plan 2" in msg and _journal()[-1]["tache"]["clips"] == [
        str(Path(d3) / "plan_1" / "video.mp4")]
    with pytest.raises(gr.Error, match="une ligne par plan"):
        videos.generer_suite(" \n ", None, videos.AUTO, "2 s", 20, 0, progress=no_progress)
    with pytest.raises(gr.Error, match="au plus"):
        videos.generer_suite("\n".join(["x"] * (videos.PLANS_MAX + 1)), None, videos.AUTO, "2 s", 20, 0,
                             progress=no_progress)


def test_images(faux_diffusion):
    """Onglet « Images » : texte seul → Z-Image ; avec une photo → FLUX.2 klein (modifier, sujet, composition)."""
    from PIL import Image

    from studiovoix import galerie, images

    # texte seul : Qwen en mode « scene », puis Z-Image au format choisi
    assert images.preparer("un phare dans la tempête", progress=no_progress) == "a red potion bottle"
    assert _journal()[-1]["tache"] == {"mode": "scene", "texte": "un phare dans la tempête"}
    msg, gal, dossier = images.generer("a lighthouse in a storm", None, images.USAGE_DEFAUT,
                                       ["photorealistic photograph, natural light, sharp focus, high detail"],
                                       "Paysage 16:9 (1344×768)", 2, 7, "Phare", "un phare", progress=no_progress)
    j = _journal()[-1]
    assert j["action"] == "image" and (j["tache"]["largeur"], j["tache"]["hauteur"]) == (1344, 768)
    assert j["tache"]["prompt"].startswith("a lighthouse in a storm. Style: photorealistic") and len(gal) == 2
    assert j["tache"]["graines"][0] == 7 and "à partir du texte" in msg and " en " in msg
    infos = galerie.lire(dossier)
    assert infos["type"] == "image" and infos["moteur"] == "Z-Image-Turbo" and infos["photo"] is None
    assert any(d == dossier for _, d in galerie.lister("Images"))
    # avec une photo (portrait 3:4), « Modifier » : consigne de retouche, klein, proportions de la photo
    photo = faux_diffusion / "moi.jpg"
    Image.new("RGB", (900, 1200), (5, 5, 5)).save(photo)
    images.preparer("mets-lui un chapeau", str(photo), images.USAGE_DEFAUT, progress=no_progress)
    assert _journal()[-1]["tache"]["mode"] == "retouche" and _journal()[-1]["tache"]["image"] == str(photo)
    msg, _, dossier = images.generer("Add a pirate hat", str(photo), images.USAGE_DEFAUT, [], images.AUTO, 1, 0,
                                     progress=no_progress)
    j = _journal()[-1]
    assert j["action"] == "personnage" and [Path(r).name for r in j["tache"]["references"]] == ["photo.jpg"]
    assert images.GARDER_LE_RESTE in j["tache"]["prompt"]
    w, h = j["tache"]["largeur"], j["tache"]["hauteur"]
    assert w % 16 == 0 and h % 16 == 0 and abs(w / h - 0.75) < 0.02 and 0.9e6 < w * h < 1.2e6
    assert galerie.lire(dossier)["usage"] == "modifier" and "photo modifiée" in msg
    # « Garder le sujet » : Qwen décrit la photo dans une nouvelle scène ; prompt court tapé à la main enrichi
    sujet = next(k for k, v in images.USAGES.items() if v == "sujet")
    images.prompt_pret("", "on the moon", str(photo), sujet, progress=no_progress)
    assert _journal()[-1]["tache"]["mode"] == "scene" and _journal()[-1]["tache"]["image"] == str(photo)
    _, _, d2 = images.generer("me on the moon", str(photo), sujet, [], images.AUTO, 1, 0, progress=no_progress)
    assert images.SUJET in _journal()[-1]["tache"]["prompt"]
    # une consigne de retouche courte est gardée telle quelle (pas de reformulation)
    n = len(_journal())
    assert images.prompt_pret("", "Add a red cape", str(photo), images.USAGE_DEFAUT, progress=no_progress) == "Add a red cape"
    assert len(_journal()) == n
    # recréer depuis la galerie : même photo (copiée), même usage, même graine
    _, nouveau = galerie.recreer(d2, progress=no_progress)
    j = _journal()[-1]
    assert j["action"] == "personnage" and images.SUJET in j["tache"]["prompt"] and nouveau != d2
    with pytest.raises(gr.Error, match="prompt"):
        images.generer(" ", None, images.USAGE_DEFAUT, [], images.AUTO, 1, 0, progress=no_progress)


def test_histoire_en_images_puis_mini_videos(faux_diffusion):
    """Un texte → N scènes (Qwen, mode « histoire ») → une image par scène, les suivantes avec la première comme
    référence (mêmes personnages) → une mini-vidéo par image, assemblées sans retirer de première image."""
    from studiovoix import galerie, images, videos

    texte = "Trois aventuriers quittent leur village et affrontent un dragon."
    scenes = images.decouper(texte, 3, progress=no_progress)
    t = _journal()[-1]["tache"]
    assert t["mode"] == "histoire" and t["nombre"] == 3 and "Number of images: 3" in t["texte"]
    assert len(scenes.splitlines()) == 3  # complété jusqu'au nombre demandé
    assert images.scenes_du_texte("1. a hero leaves home\n\n2) the dragon wakes up\n- the fight") == [
        "a hero leaves home", "the dragon wakes up", "the fight"]
    # réponse réelle de Qwen : scènes numérotées à la suite sur une seule ligne
    assert images.scenes_du_texte("Heroes leave at dawn, 2. they cross a forest 3. a dragon sleeps on gold") == [
        "Heroes leave at dawn,", "they cross a forest", "a dragon sleeps on gold"]
    # scènes déjà là : pas de nouveau découpage
    n = len(_journal())
    assert images.scenes_pretes(texte, "a\nb", 3, progress=no_progress) == "a\nb" and len(_journal()) == n
    msg, gal, dossier = images.generer_histoire("a hero leaves the village\na dragon in the sky\nthe final fight",
                                                None, [], images.AUTO, True, 11, "Dragon", texte, progress=no_progress)
    j = [e for e in _journal() if e["action"] in ("image", "personnage")][-3:]
    assert [e["action"] for e in j] == ["image", "personnage", "personnage"] and len(gal) == 3
    assert [Path(r).name for r in j[1]["tache"]["references"]] == ["scene_1.png"]
    assert images.MEMES_PERSONNAGES in j[2]["tache"]["prompt"]
    infos = galerie.lire(dossier)
    assert [s["graine"] for s in infos["scenes"]] == [11, 12, 13] and "mini-vidéos" in msg
    # mini-vidéos : une par image, chacune part de son image, assemblées (clips indépendants)
    msg, fichier, dv = videos.generer_depuis_images(dossier, "2 s", 12, "", True, "Dragon", progress=no_progress)
    j = _journal()
    clips = [e["tache"] for e in j if e["action"] == "video"][-3:]
    assert [Path(c["image"]).parent.name for c in clips] == ["plan_1", "plan_2", "plan_3"]
    assert clips[1]["prompt"].startswith("a dragon in the sky. ") and videos.MOUVEMENT_DEFAUT in clips[1]["prompt"]
    assert j[-1]["action"] == "assembler" and j[-1]["tache"]["enchaines"] is False
    assert Path(fichier) == Path(dv) / "video.mp4" and "3 plans" in msg and " en " in msg
    v = galerie.lire(dv)
    assert v["depuis_images"] and len(v["plans"]) == 3
    # recréer : mêmes images (copiées dans la création), mêmes prompts
    _, nouveau = galerie.recreer(dv, progress=no_progress)
    clips2 = [e["tache"] for e in _journal() if e["action"] == "video"][-3:]
    assert [c["prompt"] for c in clips2] == [c["prompt"] for c in clips] and nouveau != dv
    # recréer l'histoire : mêmes scènes, même graine de départ
    _, h2 = galerie.recreer(dossier, progress=no_progress)
    assert [s["graine"] for s in galerie.lire(h2)["scenes"]] == [11, 12, 13]
    with pytest.raises(gr.Error, match="histoire"):
        videos.generer_depuis_images(None, progress=no_progress)
