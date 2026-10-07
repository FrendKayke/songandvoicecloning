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
        if t["mode"] == "style": texte = "watercolor painting, soft washes, paper texture"
        elif t.get("image"): texte = "metal clink on a table"
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
        if "ECHEC" in json.dumps(t.get("prompts") or t["prompt"]):
            print("ERREUR : carte graphique saturée", flush=True); sys.exit(2)
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
    elif action == "detourer" and "entrees" in t:  # lot : fond retiré (alpha nul) partout sauf au centre
        from PIL import Image
        n = len(t["entrees"]) + 1
        for i, (e, s) in enumerate(zip(t["entrees"], t["sorties"]), 2):
            print(f"PROGRESSION {i}/{n} x", flush=True)
            if "DETOURAGE_IMPOSSIBLE" in e:
                print("ERREUR : image illisible.", flush=True); sys.exit(3)
            im = Image.open(e).convert("RGBA"); alpha = Image.new("L", im.size, 0)
            alpha.paste(255, (im.width // 4, im.height // 4, 3 * im.width // 4, 3 * im.height // 4))
            im.putalpha(alpha); im.save(s)
        print("RESULTAT " + json.dumps({"sorties": t["sorties"], "couvertures": [0.25] * len(t["sorties"])}), flush=True)
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
    assert set(moteur.CONSIGNES) == {"son", "objet", "bruitage", "image", "carte", "video", "scene", "retouche", "histoire", "style", "musiques"}
    # le texte de l'utilisateur est ajouté à toutes les consignes de reformulation (oubli constaté pour « video »)
    assert moteur.MODES_TEXTE == {"objet", "bruitage", "carte", "video", "scene", "retouche", "histoire", "musiques"}
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
    with pytest.raises(gr.Error, match="Il manque la scène"):
        cartes.generer("Mon Jeu", "x", "", [], "", None, 1, 0, progress=no_progress)
    with pytest.raises(gr.Error, match="Il manque la scène"):  # contexte vide : chaque image doit être décrite
        cartes.generer("Mon Jeu", "x", "", [], "", None, 2, 0, image_1="a ship", progress=no_progress)
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


def test_illustrations_une_description_par_image(faux_diffusion):
    from studiovoix import cartes, galerie

    # contexte commun + une description par image : un prompt par image, envoyé au moteur dans l'ordre des sorties
    contexte = "Four bearded men, all male, pirates in leather coats."
    msg, images, dossier, _ = cartes.generer("Mon Jeu", "Équipage", contexte, ["oil painting"], "", None, 3, 5,
                                             image_1="on the ship deck at dawn", image_2="fighting a kraken.",
                                             image_3="", image_4="ignored", progress=no_progress)
    t = _journal()[-1]["tache"]
    # description de l'image d'abord, contexte ensuite : un texte trop long perd la fin du contexte, pas l'action
    attendu = [f"{d}. Four bearded men, all male, pirates in leather coats. Art style: oil painting. "
               "No text, no letters, no frame." for d in ("on the ship deck at dawn", "fighting a kraken")]
    attendu.append("Four bearded men, all male, pirates in leather coats. Art style: oil painting. "
                   "No text, no letters, no frame.")
    assert t["prompts"] == attendu and t["prompt"] == attendu[0] and len(images) == 3
    infos = galerie.lire(dossier)
    assert infos["images"] == ["on the ship deck at dawn", "fighting a kraken.", ""] and infos["prompts"] == attendu
    assert infos["description"] == contexte
    # recréer la 2e image : même contexte, même description, sa graine
    graine_2 = infos["versions"][1]["graine"]
    assert cartes.description_de_version(infos, graine_2) == "fighting a kraken."
    _, nouveau = galerie.recreer(dossier, 2, progress=no_progress)
    t = _journal()[-1]["tache"]
    assert t["graines"] == [graine_2] and t["prompt"] == attendu[1] and "prompts" not in t
    # sans description par image : un seul prompt (encodé une fois)
    cartes.generer("Mon Jeu", "x", contexte, [], "", None, 2, 1, progress=no_progress)
    assert "prompts" not in _journal()[-1]["tache"]
    # sans contexte, chaque image décrite suffit ; le prompt n'est alors pas préparé depuis le français
    assert cartes.prompt_pret("", "", "a ship", "a kraken", progress=no_progress) == ""
    assert cartes.prompt_pret("", " A group. ", progress=no_progress) == "A group."
    cartes.generer("Mon Jeu", "x", "", [], "", None, 2, 1, image_1="a ship", image_2="a kraken", progress=no_progress)
    assert _journal()[-1]["tache"]["prompts"][1].startswith("a kraken. No text")
    assert [u["visible"] for u in cartes.maj_images(2)] == [True, True, False, False]


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


def test_histoire_contexte_et_une_description_par_image(faux_diffusion):
    """Contexte commun (anglais, tel quel) + un champ par image (jusqu'à 12) : chaque image reçoit « contexte.
    description » ; champ vide = contexte seul ; Qwen reçoit le contexte pour découper ; recréation identique."""
    from studiovoix import galerie, images

    contexte = "Four bearded men, all male, medieval adventurers."
    champs = ["they leave the village at dawn.", "", "they fight a troll"] + [""] * 9
    # champs remplis : gardés tels quels (pas de découpage), seulement les `nombre` premiers
    assert images.scenes_pretes("une histoire", champs, 3, contexte=contexte, progress=no_progress) == champs[:3]
    assert not (faux_diffusion / "journal.jsonl").exists()  # aucun appel au moteur
    # sans texte, contexte seul : 3 images du contexte
    assert images.scenes_pretes("", [""] * 12, 3, contexte=contexte, progress=no_progress) == ["", "", ""]
    # champs vides + texte : découpés par Qwen, qui voit le contexte
    pretes = images.scenes_pretes("Ils partent puis combattent.", [""] * 12, 3, contexte=contexte,
                                  progress=no_progress)
    t = _journal()[-1]["tache"]
    assert len(pretes) == 3 and all(pretes) and contexte in t["texte"] and "do not repeat it" in t["texte"]
    maj = images.champs_images(pretes, 3)
    assert len(maj) == images.NOMBRE_MAX == 12 and [u["visible"] for u in maj[:4]] == [True, True, True, False]
    assert [u["value"] for u in maj[:3]] == pretes and maj[3]["value"] == ""
    assert [u["visible"] for u in images.maj_champs_images(12)] == [True] * 12

    _, gal, dossier = images.generer_histoire(champs[:3], None, [], images.AUTO, True, 5, "Groupe",
                                              contexte=contexte, progress=no_progress)
    j = [e["tache"] for e in _journal() if e["action"] in ("image", "personnage")][-3:]
    # description de l'image, consigne de la référence, puis le contexte (coupé en dernier si trop long)
    assert j[0]["prompt"].startswith("they leave the village at dawn. Four bearded men, all male, medieval adventurers.")
    assert j[1]["prompt"].startswith(images.MEMES_PERSONNAGES + ". Four bearded men, all male, medieval adventurers.")
    assert j[2]["prompt"].startswith("they fight a troll. " + images.MEMES_PERSONNAGES + ". Four bearded men")
    infos = galerie.lire(dossier)
    assert infos["contexte"] == contexte and [s["image"] for s in infos["scenes"]] == [
        "they leave the village at dawn.", "", "they fight a troll"]
    _, h2 = galerie.recreer(dossier, progress=no_progress)
    assert [s["prompt"] for s in galerie.lire(h2)["scenes"]] == [s["prompt"] for s in infos["scenes"]]
    # mini-vidéos : description, mouvement, puis contexte (Wan lit 512 jetons : le mouvement ne doit pas sauter)
    from studiovoix import videos

    videos.generer_depuis_images(dossier, "2 s", 10, "slow pan", False, "g", progress=no_progress)
    clips = [e["tache"]["prompt"] for e in _journal() if e["action"] == "video"][-3:]
    assert clips[0] == "they leave the village at dawn. slow pan. Four bearded men, all male, medieval adventurers."
    assert clips[1] == "slow pan. Four bearded men, all male, medieval adventurers."
    # sans contexte, un champ vide est une erreur (sauf si tous sont vides : rien à générer)
    with pytest.raises(gr.Error, match="Image\\(s\\) 2 sans description"):
        images.generer_histoire(["a", "", "c"], None, [], images.AUTO, progress=no_progress)
    with pytest.raises(gr.Error, match="aucune scène"):
        images.generer_histoire(["", ""], None, [], images.AUTO, progress=no_progress)


def test_histoire_avec_images_de_depart(faux_diffusion):
    """Images de départ (personnages, lieu) : Qwen les voit en planche, toutes les images les reprennent (klein),
    sans l'image 1 de l'histoire en plus ; 3 au plus ; copiées pour la recréation."""
    from PIL import Image

    from studiovoix import galerie, images

    depart = []
    for k, taille in enumerate([(600, 900), (900, 600)], 1):
        depart.append(str(faux_diffusion / f"perso_{k}.png"))
        Image.new("RGB", taille, (k, k, k)).save(depart[-1])
    images.decouper("Deux héros dans une forêt.", 2, depart, progress=no_progress)
    t = _journal()[-1]["tache"]
    with Image.open(t["image"]) as planche:  # les deux images côte à côte, même hauteur
        assert planche.height == 768 and planche.width == 512 + 16 + 1152
    _, _, dossier = images.generer_histoire("two heroes walk in a forest\nthey find a cave", depart, [], images.AUTO,
                                            True, 3, progress=no_progress)
    j = [e["tache"] for e in _journal() if e["action"] in ("image", "personnage")][-2:]
    assert [[Path(r).name for r in e["references"]] for e in j] == [
        ["depart_1.png", "depart_2.png"], ["depart_1.png", "depart_2.png"]]  # pas l'image 1 : pose recopiée
    assert images.DEPART in j[0]["prompt"] and images.DEPART in j[1]["prompt"]
    w, h = j[0]["largeur"], j[0]["hauteur"]
    assert h > w  # format automatique d'après la première image de départ (portrait)
    assert galerie.lire(dossier)["depart"] == ["depart_1.png", "depart_2.png"]
    _, nouveau = galerie.recreer(dossier, progress=no_progress)
    assert [Path(r).name for r in _journal()[-1]["tache"]["references"]] == ["depart_1.png", "depart_2.png"]
    # la photo de l'onglet sert d'image de départ quand le volet n'en a pas
    assert images.images_de_depart(None, depart[0]) == [depart[0]]
    with pytest.raises(gr.Error, match="3 au plus"):
        images.images_de_depart(depart * 2)


def test_images_plusieurs_photos(faux_diffusion):
    """Onglet « Images » avec plusieurs photos : Qwen voit une planche numérotée, klein reçoit les photos dans
    l'ordre, consignes qui désignent « image 1 », 4 photos au plus, recréation avec toutes les photos."""
    from PIL import Image

    from studiovoix import galerie, images

    photos = []
    for k, taille in enumerate([(800, 1000), (1200, 800), (600, 600)], 1):
        photos.append(str(faux_diffusion / f"p{k}.jpg"))
        Image.new("RGB", taille, (k * 40, 0, 0)).save(photos[-1])
    liste = images.photos_de(photos[0], photos[1:])
    assert liste == photos
    sujet = next(k for k, v in images.USAGES.items() if v == "sujet")
    images.preparer("la femme de la photo 1 dans le château de la photo 2", liste, sujet, progress=no_progress)
    t = _journal()[-1]["tache"]
    assert t["mode"] == "scene" and t["photos"] == 3
    with Image.open(t["image"]) as planche:
        assert planche.height == 768 and planche.getpixel((10, 10)) == (0, 0, 0)  # numéro sur fond noir
    msg, _, dossier = images.generer("the woman from image 1 in the castle from image 2", liste, sujet, [],
                                     images.AUTO, 1, 4, progress=no_progress)
    j = _journal()[-1]["tache"]
    assert [Path(r).name for r in j["references"]] == ["photo.jpg", "photo_2.jpg", "photo_3.jpg"]
    assert images.SUJET_N in j["prompt"] and images.SUJET not in j["prompt"] and "3 photos combinées" in msg
    assert j["hauteur"] > j["largeur"]  # format automatique d'après la photo 1 (portrait)
    assert galerie.lire(dossier)["autres_photos"] == ["photo_2.jpg", "photo_3.jpg"]
    _, nouveau = galerie.recreer(dossier, progress=no_progress)
    assert [Path(r).name for r in _journal()[-1]["tache"]["references"]] == ["photo.jpg", "photo_2.jpg", "photo_3.jpg"]
    # « Modifier » avec 2 photos : retouche de la photo 1 avec un élément de la photo 2
    images.preparer("mets-lui la veste de la photo 2", photos[:2], images.USAGE_DEFAUT, progress=no_progress)
    assert _journal()[-1]["tache"]["mode"] == "retouche" and _journal()[-1]["tache"]["photos"] == 2
    images.generer("Put the jacket from image 2 on the person of image 1", photos[:2], images.USAGE_DEFAUT, [],
                   images.AUTO, 1, 0, progress=no_progress)
    assert images.GARDER_LE_RESTE_N in _journal()[-1]["tache"]["prompt"]
    # une seule photo : rien ne change (pas de planche, consignes d'origine)
    images.preparer("mets-lui un chapeau", photos[0], images.USAGE_DEFAUT, progress=no_progress)
    assert _journal()[-1]["tache"]["image"] == photos[0] and "photos" not in _journal()[-1]["tache"]
    with pytest.raises(gr.Error, match="4 au plus"):
        images.photos_de(photos[0], photos * 2)


def test_longueur_du_texte_lu_par_le_generateur(capsys):
    """Z-Image / klein coupent le texte à 512 jetons par défaut (sans prévenir) : un prompt plus long est lu jusqu'à
    TEXTE_MAX (multiple de 64), au-delà un avertissement est écrit ; un prompt court garde 512."""
    from types import SimpleNamespace

    class Tokeniseur:  # un jeton par mot, plus 10 pour le gabarit de conversation
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            return list(range(len(messages[0]["content"].split()) + 10))

    pipe = SimpleNamespace(tokenizer=Tokeniseur())
    assert moteur._longueur_texte(pipe, "a cat") == moteur.TEXTE_DEFAUT == 512
    assert moteur._longueur_texte(pipe, "w " * 600) == 640 and "AVERTISSEMENT" not in capsys.readouterr().out
    assert moteur._longueur_texte(pipe, "w " * 2000 + "fin") == moteur.TEXTE_MAX == 1024
    assert "AVERTISSEMENT : prompt trop long (2011 jetons)" in capsys.readouterr().out


def test_images_en_serie_depuis_un_tableau_excel(faux_diffusion):
    """Excel : feuille non vide choisie, colonnes devinées (prompts, noms, contexte), contexte d'une case + écrit,
    paquets envoyés au moteur, copies réduites, lot.csv pour Excel, archive."""
    import csv as csv_
    import zipfile

    import openpyxl

    from studiovoix import serie

    f = cfg.DATA_DIR.parent / "icones.xlsx"
    classeur = openpyxl.Workbook()
    classeur.active.title = "Vide"
    ws = classeur.create_sheet("Icônes")
    ws.append(["Nom", "Prompt", "Contexte", "", "Contexte global"])
    ws.append(["Panier", "a shopping cart", "", "", "cooking app icons"])
    for i in range(2, 11):
        ws.append([f"Icône {i}" if i != 5 else "Panier", f"icon number {i}", "red" if i == 3 else None])
    ws.append(["sans prompt", None])
    classeur.save(f)

    feuille, col_p, col_n, col_c, apercu, msg = serie.analyser(str(f))
    assert feuille["value"] == "Icônes" and feuille["visible"] and feuille["choices"] == ["Vide", "Icônes"]
    # « Contexte » (C) n'est remplie que sur une ligne sur dix : pas proposée comme contexte par ligne (choisie à la
    # main plus bas) ; une case « contexte global » seule ne doit pas être répétée dans chaque image
    assert (col_p["value"], col_n["value"], col_c["value"]) == ("B", "A", serie.AUCUNE) and "10 case(s)" in msg
    assert ("B — Prompt", "B") in col_p["choices"] and apercu["value"]["headers"][0] == "A Nom"
    assert serie.prompt_image({"prompt": "x", "contexte_ligne": "Cooking app icons"}, "cooking app icons", []) == \
        "x. cooking app icons."
    assert serie.valeur_case(str(f), "Icônes", "e2") == "cooking app icons"
    assert serie.valeur_case(str(f), "Vide", "Icônes!E2") == "cooking app icons"
    with pytest.raises(gr.Error, match="vide"):
        serie.valeur_case(str(f), "Icônes", "Z9")
    with pytest.raises(gr.Error, match="comme B2"):
        serie.valeur_case(str(f), "Icônes", "deux")

    lot = serie.entrees(str(f), "Icônes", True, "B", "A", "C")
    # la colonne des noms donne exactement le nom du fichier (accents gardés) ; un doublon reçoit « _2 »
    assert len(lot) == 10 and lot[0]["fichier"] == "Panier.png" and lot[4]["fichier"] == "Panier_2.png"
    assert lot[4]["doublon"] and lot[1]["fichier"] == "Icône 2.png" and lot[2]["contexte_ligne"] == "red"
    assert serie.entrees(str(f), "Icônes", True, "B", "A", "C", numeroter=True)[0]["fichier"] == "001_Panier.png"
    assert serie.entrees(str(f), "Icônes", True, "B", None, None)[0]["fichier"] == "001_a_shopping_cart.png"
    tableau, info = serie.apercu_prompts(str(f), "Icônes", True, "B", "A", "C", "E2", "", "flat colors",
                                         [serie.STYLES[0][1]], None)
    assert "10 image(s)" in info and "Z-Image" in info and "1 nom(s) de fichier en double" in info
    style = serie.STYLES[0][1]  # le style en tête et rappelé à la fin
    assert tableau["data"][2][2] == f"Art style: {style}. icon number 3. red. cooking app icons. flat colors. " \
                                    f"Art style: {style}."

    msg, galerie, archive, dossier = serie.generer(str(f), "Icônes", True, "B", "A", "C", "E2", "", "flat colors",
                                                  [serie.STYLES[0][1]], None, "Carré 1:1 (1024×1024)", [32, 512],
                                                  7, True, "Mes icônes", 0, progress=no_progress)
    taches = [e["tache"] for e in _journal() if e["action"] == "image"]
    assert [len(t["sorties"]) for t in taches] == [8, 2]  # paquets de serie.PAQUET
    assert "icon number 3. red. cooking app icons" in taches[0]["prompts"][2] and taches[0]["graines"] == [7] * 8
    d = Path(dossier)
    assert d.parent == cfg.SERIES_DIR and d.name.endswith("_Mes_icones") and "10/10" in msg and len(galerie) == 10
    assert (d / "32px" / "Panier.png").exists() and (d / "512px" / "Icône 10.png").exists()
    with zipfile.ZipFile(archive) as z:
        noms = z.namelist()
    assert "Panier.png" in noms and "32px/Panier.png" in noms and "Icône 10.png" in noms and "lot.csv" in noms
    texte_csv = (d / "lot.csv").read_text(encoding="utf-8-sig")
    lignes = list(csv_.reader(texte_csv.splitlines(), delimiter=";"))
    assert lignes[0][:3] == ["numero", "ligne", "nom"] and lignes[1][2] == "Panier" and lignes[1][6] == "oui"
    assert (d / "lot.csv").read_bytes().startswith(b"\xef\xbb\xbf")  # BOM : Excel lit les accents


def test_images_en_serie_liste_csv_style_et_reprise(faux_diffusion, monkeypatch):
    """Liste collée, CSV en cp1252 avec « ; », images de style données à FLUX.2 klein, paquet en échec puis reprise."""
    from PIL import Image

    from studiovoix import serie

    monkeypatch.setattr(serie, "PAQUET", 2)
    c = cfg.DATA_DIR.parent / "liste.csv"
    c.write_bytes("titre;texte\nun;épée dorée\ndeux;ECHEC bouclier\ntrois;arc\n".encode("cp1252"))
    feuille, col_p, *_ = serie.analyser(str(c))
    assert feuille["value"] == "CSV" and not feuille["visible"] and col_p["value"] == "B"
    style = cfg.DATA_DIR.parent / "style.png"
    Image.new("RGB", (40, 40), "blue").save(style)
    msg, galerie, archive, dossier = serie.generer(str(c), "CSV", True, "B", "A", None, "", "", "", [], [str(style)],
                                                  "Carré 1:1 (1024×1024)", [], 0, False, "", 0, avec_klein=True,
                                                  progress=no_progress)
    j = [e for e in _journal() if e["action"] == "personnage"]
    assert j[0]["tache"]["references"][0].endswith("style_1.png") and serie.STYLE_DES_IMAGES in j[0]["tache"]["prompts"][0]
    assert "1/3" in msg and "⚠️ images 1–2" in msg and "Reprendre" in msg and len(galerie) == 1
    # le prompt est corrigé dans lot.json (ou la cause de l'échec a disparu) : seules les images manquantes repartent
    lot = Path(dossier) / "lot.json"
    lot.write_text(lot.read_text(encoding="utf-8").replace("ECHEC ", ""), encoding="utf-8")
    n = len(_journal())
    msg, galerie, archive, _ = serie.reprendre(dossier, progress=no_progress)
    nouvelles = [e["tache"] for e in _journal()[n:]]
    assert len(nouvelles) == 1 and len(nouvelles[0]["sorties"]) == 2 and "3/3" in msg and archive.endswith("serie.zip")
    # liste collée, sans fichier ; graines qui se suivent sans « même graine »
    serie.generer(None, None, True, None, None, None, "", "a\n\nb\n", "", [], None, "Carré 1:1 (1024×1024)", [],
                  5, False, "x", 0, progress=no_progress)
    assert _journal()[-1]["tache"]["graines"] == [5, 6] and _journal()[-1]["tache"]["prompts"] == ["a.", "b."]
    with pytest.raises(gr.Error, match="Aucun prompt"):
        serie.entrees(liste=" \n ")
    with pytest.raises(gr.Error, match="Aucun lot"):
        serie.reprendre(None, progress=no_progress)


def test_style_des_images_en_serie(faux_diffusion):
    """Le style est respecté : rien de coché par défaut qui le contredise, « dessin » ou « aquarelle » tapés en
    français traduits, style en tête du prompt ; les images de style sont décrites en mots par Qwen (mode « style »)
    et Z-Image génère chaque image seule (FLUX.2 klein recopiait des éléments des images) ; graines différentes."""
    from PIL import Image

    from studiovoix import serie, styles

    assert "hand-drawn" in styles.image(["dessin"]) and "watercolor" in styles.image("Aquarelle")
    assert styles.image(["dessin animé"]).startswith("cartoon") and styles.image(["portrait"]) == "portrait"
    assert styles.image(["Aquarelle"], [("Aquarelle", "watercolor X")]) == "watercolor X"  # libellé du catalogue
    assert serie.texte_du_style(["dessin"]).startswith("hand-drawn")
    p_ = serie.prompt_image({"prompt": "a cat"}, "", ["dessin"])
    assert p_.startswith("Art style: hand-drawn") and p_.endswith("sketchy drawing style.") and "a cat." in p_
    # images d'aquarelle : style décrit par Qwen puis ajouté à chaque prompt, sans les donner au générateur
    aquarelle = cfg.DATA_DIR.parent / "aquarelle.png"
    Image.new("RGB", (40, 40), "teal").save(aquarelle)
    assert "watercolor" in serie.lire_style([str(aquarelle)], progress=no_progress)
    *_, dossier = serie.generer(None, None, True, None, None, None, "", "a fox\na bear\n", "", [], [str(aquarelle)],
                                "Carré 1:1 (1024×1024)", [], 7, False, "x", 0, progress=no_progress)
    j = _journal()
    assert [e["tache"]["mode"] for e in j if e["action"] == "decrire"] == ["style", "style"]
    taches = [e["tache"] for e in j if e["action"] == "image"]
    assert not [e for e in j if e["action"] == "personnage"] and taches[-1]["graines"] == [7, 8]
    assert all(p_.startswith("Art style: watercolor painting") for p_ in taches[-1]["prompts"])
    infos = json.loads((Path(dossier) / "lot.json").read_text(encoding="utf-8"))
    assert infos["style_lu"].startswith("watercolor") and infos["images_style"] == ["style_1.png"]
    assert infos["references"] == [] and (Path(dossier) / "style_1.png").exists()
    # style déjà lu (et retouché) : Qwen n'est pas relancé
    n = len(_journal())
    serie.generer(None, None, True, None, None, None, "", "a fox", "", [], [str(aquarelle)], "Carré 1:1 (1024×1024)",
                  [], 7, False, "x", 0, style_lu="ink wash, grey tones", progress=no_progress)
    assert [e["action"] for e in _journal()[n:]] == ["image"]
    assert _journal()[-1]["tache"]["prompts"][0].startswith("Art style: ink wash, grey tones.")
    tableau, info = serie.apercu_prompts(None, None, True, None, None, None, "", "a fox", "", [], [str(aquarelle)])
    assert "pas encore lu" in info


def test_textes_coupes_par_les_moteurs(capsys):
    """Stable Audio (T5, 128 jetons) et Wan (umT5, 512) coupent sans prévenir : avertissement ; réponse de Qwen
    arrêtée par sa limite : phrase inachevée retirée."""
    from types import SimpleNamespace

    tokeniseur = lambda texte: SimpleNamespace(input_ids=texte.split())  # noqa: E731 - un jeton par mot
    assert moteur._avertir_si_coupe(tokeniseur, "a door creaks", 128, "Stable Audio") == 3
    assert "AVERTISSEMENT" not in capsys.readouterr().out
    moteur._avertir_si_coupe(tokeniseur, "w " * 600 + "camera pans left", 512, "Wan 2.2")
    sortie = capsys.readouterr().out
    assert "AVERTISSEMENT : prompt trop long pour Wan 2.2 (603 jetons" in sortie and "camera pans left" in sortie
    assert moteur._phrases_completes("A knight rides. The dragon roars. The sky tur", "video") == \
        "A knight rides. The dragon roars."
    assert moteur._phrases_completes("1. a\n2. b\n3. c is cu", "histoire") == "1. a\n2. b"
    assert moteur.JETONS_QWEN >= 320


def test_noms_de_fichiers_des_images_en_serie(faux_diffusion):
    """Le nom écrit dans le tableau devient le nom du fichier : caractères interdits par Windows remplacés,
    extension .jpg / .webp respectée (format de l'image), noms réservés de Windows évités."""
    from PIL import Image

    from studiovoix import serie

    assert serie.nom_de_fichier("icone maison") == "icone maison.png"
    assert serie.nom_de_fichier("  ic/ône:1?  ") == "ic_ône_1_.png"
    assert serie.nom_de_fichier("fond.JPEG") == "fond.jpg" and serie.nom_de_fichier("logo.webp") == "logo.webp"
    assert serie.nom_de_fichier("v1.2") == "v1.2.png" and serie.nom_de_fichier("CON") == "CON_.png"
    assert serie.nom_de_fichier(" ... ") == "" and serie.nom_de_fichier("") == ""
    c = cfg.DATA_DIR.parent / "noms.csv"
    c.write_text("fichier;prompt\nfond.jpg;a sunset\nLOGO;a fox\nlogo;a cat\n;a dog\n", encoding="utf-8")
    *_, dossier = serie.generer(str(c), "CSV", True, "B", "A", None, "", "", "", [], None, "Carré 1:1 (1024×1024)",
                                [64], 0, True, "", 0, progress=no_progress)
    d = Path(dossier)
    assert sorted(p.name for p in d.glob("*.*") if p.suffix in (".png", ".jpg")) == [
        "004_a_dog.png", "LOGO.png", "fond.jpg", "logo_2.png"]
    with Image.open(d / "fond.jpg") as im:
        assert im.format == "JPEG"
    assert (d / "64px" / "fond.jpg").exists()


def test_images_en_serie_a_fond_transparent(faux_diffusion, monkeypatch):
    """Fond transparent : fond uni demandé dans le prompt, image générée dans opaque/, puis détourée (un appel de
    BiRefNet par paquet) vers le fichier final ; .jpg → .png ; la reprise ne refait que ce qui manque."""
    import zipfile

    from PIL import Image

    from studiovoix import serie

    monkeypatch.setattr(serie, "PAQUET", 2)
    c = cfg.DATA_DIR.parent / "icones.csv"
    c.write_text("fichier;prompt\nfond.jpg;a sunset\ncoeur;a heart\nlogo.webp;a fox\n", encoding="utf-8")
    tableau, info = serie.apercu_prompts(str(c), "CSV", True, "B", "A", None, "", "", "", [], None, 0, False, True)
    assert [ligne[1] for ligne in tableau["data"]] == ["fond.png", "coeur.png", "logo.webp"]
    assert serie.FOND_UNI in tableau["data"][0][2] and "BiRefNet" in info and "1 nom(s) en .jpg" in info
    assert serie.entrees(liste="a", transparent=False)[0]["en_png"] is False
    # deux noms qui ne diffèrent que par l'extension : même image dans opaque/, donc renommés
    c2 = cfg.DATA_DIR.parent / "doubles.csv"
    c2.write_text("fichier;prompt\nx.png;a\nx.webp;b\n", encoding="utf-8")
    assert [e["fichier"] for e in serie.entrees(str(c2), "CSV", True, "B", "A", transparent=True)] == \
        ["x.png", "x_2.webp"]
    assert [e["fichier"] for e in serie.entrees(str(c2), "CSV", True, "B", "A")] == ["x.png", "x.webp"]

    msg, galerie, archive, dossier = serie.generer(str(c), "CSV", True, "B", "A", None, "", "", "", [], None,
                                                  "Carré 1:1 (1024×1024)", [32], 0, True, "", 0, False, True,
                                                  progress=no_progress)
    d = Path(dossier)
    j = _journal()
    images = [e["tache"] for e in j if e["action"] == "image"]
    lots = [e["tache"] for e in j if e["action"] == "detourer"]
    assert all(Path(s).parent.name == serie.OPAQUE for t in images for s in t["sorties"])
    assert [len(t["entrees"]) for t in lots] == [2, 1] and lots[0]["modele"] == "general"
    assert "3/3" in msg and "Fond transparent" in msg and "BiRefNet" in msg
    for nom in ("fond.png", "coeur.png", "logo.webp", "32px/fond.png"):
        with Image.open(d / nom) as im:
            assert im.mode == "RGBA" and im.getpixel((0, 0))[3] == 0, nom
    assert not (d / "fond.jpg").exists() and (d / serie.OPAQUE / "fond.png").exists()
    with zipfile.ZipFile(archive) as z:
        noms = z.namelist()
    assert "fond.png" in noms and not any(n.startswith(serie.OPAQUE) for n in noms)
    assert json.loads((d / "lot.json").read_text(encoding="utf-8"))["transparent"] is True
    # reprise : une image détourée perdue est seulement redétourée (pas regénérée)
    (d / "coeur.png").unlink()
    n = len(_journal())
    msg, *_ = serie.reprendre(dossier, progress=no_progress)
    nouvelles = _journal()[n:]
    assert [e["action"] for e in nouvelles] == ["detourer"] and nouvelles[0]["tache"]["sorties"] == [str(d / "coeur.png")]
    assert "3/3" in msg


def test_reviser_un_modele_3d(faux_diffusion):
    """Révision : l'image est corrigée d'après un texte (Qwen « retouche » puis FLUX.2 klein, le reste gardé), une vue
    de dos est dessinée d'après la face ; avec elle, la forme passe par Hunyuan3D-2mv (dos dans la tâche, 5 pas)."""
    from PIL import Image

    from studiovoix import galerie
    from studiovoix.onglets import modeles_3d

    face = faux_diffusion / "archer.png"
    Image.new("RGB", (600, 900), "white").save(face)
    msg, glb, _, _, dossier = modele3d.generer(str(face), "Archer", "Normale", True, 0, ["glb"], progress=no_progress)
    image, dos, graine = modele3d.reprendre_images(dossier)
    assert dos is None and graine == 555 and Path(image).name == "image.png"
    assert modeles_3d._graine_du_modele(dossier) == (555, None, None)
    # correction de la face : consigne de Qwen + « garder le reste », proportions de l'image
    corrigee, msg, note = modele3d.corriger_image(image, "refais l'arc", progress=no_progress)
    j = _journal()
    assert j[-2]["action"] == "decrire" and j[-2]["tache"]["mode"] == "retouche" and j[-2]["tache"]["image"] == image
    t = j[-1]["tache"]
    assert j[-1]["action"] == "personnage" and t["references"] == [image] and modele3d.GARDER_OBJET in t["prompt"]
    assert t["hauteur"] > t["largeur"] and Path(corrigee).name == "revision.png" and note == "refais l'arc"
    # vue de dos (avec une précision), puis correction de la vue de dos depuis l'onglet
    vue_dos, msg = modele3d.creer_vue_de_dos(corrigee, "l'arc bien visible", progress=no_progress)
    assert modele3d.VUE_DE_DOS in _journal()[-1]["tache"]["prompt"] and Path(vue_dos).name == "dos.png"
    face2, dos2, msg, note, de = modeles_3d._corriger("La vue de dos", corrigee, vue_dos, "corde visible", dossier,
                                                     progress=no_progress)
    assert face2 == corrigee and Path(dos2).name == "revision.png" and de == dossier
    with pytest.raises(gr.Error, match="vue de dos"):
        modeles_3d._corriger("La vue de dos", corrigee, None, "x", dossier, progress=no_progress)
    with pytest.raises(gr.Error, match="Écris ce qu'il faut corriger"):
        modele3d.corriger_image(image, " ", progress=no_progress)
    # nouveau modèle : face corrigée + dos, même graine, révision notée ; forme multi-vues (5 pas, même en Maximale)
    msg, glb, _, _, d2 = modele3d.generer(face2, "Archer", "Maximale (modèle complet, 50 étapes, la plus détaillée)",
                                          True, graine, ["glb"], image_dos=dos2, revision="refais l'arc",
                                          revision_de=dossier, progress=no_progress)
    t = _journal()[-1]["tache"]
    assert t["dos"] == str(Path(d2) / "image_dos.png") and t["etapes"] == 5 and "sous_dossier" not in t
    assert t["graine"] == 555 and "face et le dos" in msg
    infos = galerie.lire(d2)
    assert infos["image_dos"] == "image_dos.png" and infos["revision"] == "refais l'arc" and infos["revision_de"] == dossier
    assert "révision : « refais l'arc »" in galerie.details(d2, 1)[0]
    assert modele3d.reprendre_images(d2)[1] == str(Path(d2) / "image_dos.png")
    _, d3 = galerie.recreer(d2, 1, progress=no_progress)  # la recréation garde la vue de dos
    assert _journal()[-1]["tache"]["dos"] == str(Path(d3) / "image_dos.png")


def test_ckpt_de_hunyuan_retires(tmp_path, monkeypatch, capsys):
    """Les .ckpt de Hunyuan3D (mêmes poids que les .safetensors, jamais chargés) sont supprimés avec leur blob ;
    ceux du peintre et ceux sans .safetensors à côté sont gardés."""
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    depot = tmp_path / "models--tencent--Hunyuan3D-2"
    blobs, snap = depot / "blobs", depot / "snapshots" / "abc"
    blobs.mkdir(parents=True)
    fichiers = {"hunyuan3d-dit-v2-0-turbo/model.fp16.ckpt": 3000, "hunyuan3d-dit-v2-0-turbo/model.fp16.safetensors": 10,
                "hunyuan3d-vae-v2-0-turbo/model.fp16.ckpt": 2000, "hunyuan3d-vae-v2-0-turbo/model.fp16.safetensors": 10,
                "hunyuan3d-paint-v2-0-turbo/model.ckpt": 5, "hunyuan3d-paint-v2-0-turbo/x.safetensors": 5,
                "hunyuan3d-dit-v2-0/model.ckpt": 7}
    for i, (nom, taille) in enumerate(fichiers.items()):
        (blobs / f"b{i}").write_bytes(b"x" * taille)
        (snap / nom).parent.mkdir(parents=True, exist_ok=True)
        try:
            (snap / nom).symlink_to(blobs / f"b{i}")
        except OSError:  # Windows sans mode développeur : copies, comme le fait huggingface_hub
            (snap / nom).write_bytes(b"x" * taille)
    moteur._retirer_ckpt_inutiles()
    restants = sorted(str(p.relative_to(snap)).replace("\\", "/") for p in snap.rglob("*") if p.is_file())
    assert restants == ["hunyuan3d-dit-v2-0-turbo/model.fp16.safetensors", "hunyuan3d-dit-v2-0/model.ckpt",
                        "hunyuan3d-paint-v2-0-turbo/model.ckpt", "hunyuan3d-paint-v2-0-turbo/x.safetensors",
                        "hunyuan3d-vae-v2-0-turbo/model.fp16.safetensors"]
    assert "Go libérés" in capsys.readouterr().out
    assert moteur.MODELES["forme3d"][0][1] == [
        "hunyuan3d-dit-v2-0-turbo/config.yaml", "hunyuan3d-dit-v2-0-turbo/model.fp16.safetensors",
        "hunyuan3d-vae-v2-0-turbo/config.yaml", "hunyuan3d-vae-v2-0-turbo/model.fp16.safetensors",
        "hunyuan3d-dit-v2-0/config.yaml", "hunyuan3d-dit-v2-0/model.fp16.safetensors"]
