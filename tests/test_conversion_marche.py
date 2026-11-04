"""Feuilles de marche : bandes de Diamant / Perle / Platine, légendes dans le budget de CLIP, conversion en masse d'un
style à l'autre (mise en page seule, puis redessin avec un faux SDXL)."""
import json
import zipfile
from pathlib import Path

import gradio as gr
import numpy as np
import pytest
from PIL import Image, ImageDraw

from conftest import no_progress
from studiovoix import config as cfg
from studiovoix import conversion_marche as cm
from studiovoix import diffusion, dresseur, modele_sprites

GBA, DS = list(dresseur.STYLES)
COULEURS = {"bas": (200, 40, 40), "gauche": (40, 160, 40), "droite": (40, 40, 200), "haut": (220, 200, 40)}


def _feuille_gba(k=2):
    """Feuille GBA (4×4 cases de 16×24) : une couleur par direction, personnage de 12×20, un pixel plus haut au pas
    (colonnes 1 et 3), un œil noir à gauche du visage pour reconnaître le miroir."""
    im = Image.new("RGBA", (64, 96), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for r, sens in enumerate(dresseur.STYLES[GBA]["marche"]["lignes"]):
        for c in range(4):
            haut = r * 24 + 3 - (c % 2)
            d.rectangle((c * 16 + 2, haut, c * 16 + 13, haut + 19), fill=COULEURS[sens] + (255,))
            d.point((c * 16 + 4 + (c == 3), haut + 3), fill=(0, 0, 0, 255))  # autre pas : œil décalé
    return im.resize((128, 192), Image.Resampling.NEAREST) if k == 2 else im


def test_bande_diamant_perle_platine(tmp_path):
    """32×512 = 16 images (haut, bas, gauche, droite × debout, pas, debout, autre pas) → 3×4 cases de 32×32."""
    bande = np.zeros((512, 32, 4), np.uint8)
    for i in range(16):
        bande[i * 32 + 10:i * 32 + 30, 8:24] = (10 * i, 100, 200, 255)
    f = tmp_path / "Pt_npc_test.png"
    Image.fromarray(bande, "RGBA").save(f)
    a = modele_sprites.lire_feuille(f, (3, 4), (32, 32))
    assert a.shape == (128, 96, 4)
    for r in range(4):
        for c, i in enumerate((0, 1, 3)):
            assert a[r * 32 + 20, c * 32 + 16, 0] == 10 * (r * 4 + i)
    assert modele_sprites.verifier_feuilles([str(f)], DS)[1][0][3] == "✅ bonne"


def test_legende_dans_le_budget_de_clip():
    longue = ", ".join(["a young man", "light skin", "short spiky brown hair", "red cap", "blue zip jacket over a "
                        "white shirt", "black cargo pants", "red sneakers", "yellow backpack"] + [f"detail {i}" for i in
                                                                                              range(20)])
    prompt = modele_sprites.prompt_feuille(longue, (3, 4))
    assert modele_sprites.jetons_estimes(prompt) <= modele_sprites.JETONS_CLIP
    assert prompt.startswith("pixelsheet, walking sprite sheet, 3 by 4 grid, a young man, light skin")
    assert "yellow backpack" in prompt and "detail 19" not in prompt and prompt.endswith("on a magenta background")
    courte = "a girl, brown hair"
    assert modele_sprites.prompt_feuille(courte) == modele_sprites._legende_texte(courte)
    # réponse réelle de Qwen avec la première consigne par cases (photo de l'utilisateur, 04/11)
    qwen = ("woman, young, light skin, dark brown curly hair, none, light blue dress, light blue skirt, none, gold "
            "pendant necklace, gold earrings, no bag, no held object")
    assert modele_sprites.sans_absences(qwen) == ("woman, young, light skin, dark brown curly hair, light blue dress, "
                                                  "light blue skirt, gold pendant necklace, gold earrings")
    assert modele_sprites.sans_absences("a man, nose ring, notebook, without hat.") == "a man, nose ring, notebook"


def test_mise_en_page_gba_vers_ds():
    avant = np.array(_feuille_gba(1))
    page = cm.mettre_en_page(avant, GBA, DS)
    assert page.shape == (128, 96, 4)
    for r, sens in enumerate(dresseur.STYLES[DS]["marche"]["lignes"]):  # haut, bas, gauche, droite
        for c in range(3):
            case = page[r * 32:(r + 1) * 32, c * 32:(c + 1) * 32]
            y, x = np.where(case[..., 3] > 0)
            assert tuple(case[y[0], x[0], :3]) in (COULEURS[sens], (0, 0, 0))
            assert y.max() - y.min() + 1 == 25  # 20 px (21 avec le pas) agrandis à la hauteur du style DS
            assert y.max() == (30 if c == 0 else 29) and abs((x.min() + x.max()) / 2 - 15.5) <= 1  # pas gardé
    # colonne 2 = « autre pas » (colonne 3 d'origine : œil décalé), pas la colonne 2 d'origine (debout)
    noir = lambda case: np.argwhere((case[..., 3] > 0) & (case[..., :3].sum(-1) == 0))[0][1]  # noqa: E731
    assert noir(page[32:64, 64:96]) > noir(page[32:64, 0:32])
    miroir = cm.miroir_gauche(page, DS)
    assert np.array_equal(miroir[96:128, 0:32], page[64:96, 0:32][:, ::-1])
    assert np.array_equal(miroir[:96], page[:96])


def test_conversion_sans_modele(env, tmp_path):
    fichiers = []
    for i in range(3):
        f = tmp_path / f"trchar{i:03d}.png"
        _feuille_gba().save(f)
        fichiers.append(str(f))
    vide = tmp_path / "vide.png"
    Image.new("RGBA", (128, 192)).save(vide)
    msg, planches, archive, dossier = cm.convertir(fichiers + [str(vide)], GBA, DS, progress=no_progress)
    assert "3 feuille(s) convertie(s) sur 4" in msg and "vide.png" in msg and "mise en page seulement" in msg
    assert len(planches) == 3
    with zipfile.ZipFile(archive) as z:
        assert sorted(z.namelist()) == sorted([f"trchar{i:03d}{s}.png" for i in range(3) for s in ("", "_x2")])
    a = np.array(Image.open(Path(dossier) / "resultats" / "trchar000.png"))
    assert a.shape == (128, 96, 4)
    infos = json.loads((Path(dossier) / "lot.json").read_text(encoding="utf-8"))
    assert infos["modele"] is None and len(infos["feuilles"]) == 3
    with pytest.raises(gr.Error, match="Dépose"):
        cm.convertir([], GBA, DS, progress=no_progress)


def test_conversion_avec_un_modele_entraine(env, tmp_path, monkeypatch):
    """Faux modèle de feuilles DS : SDXL reçoit la mise en page ×8 sur magenta, un masque blanc entier et la part
    redessinée ; il rend l'image en éclaircissant le personnage ; le résultat reprend les couleurs d'origine."""
    lora = cfg.LORAS_DIR / "marche_ds" / "marche_ds.safetensors"
    lora.parent.mkdir(parents=True)
    lora.write_bytes(b"x")
    (lora.parent / "modele.json").write_text(json.dumps({"style": DS, "type": "marche"}), encoding="utf-8")
    appels = []

    def sprite_dresseur(prompts, sorties, graines, largeur=1024, hauteur=1024, etapes=30, guidage=7.0, negatif=None,
                        lora_echelle=1.0, lora=None, progress=None, image=None, masque=None, force=0.85):
        appels.append({"prompt": prompts, "taille": (largeur, hauteur), "lora": lora, "force": force,
                       "masque": np.array(Image.open(masque)), "graines": graines})
        a = np.array(Image.open(image).convert("RGB")).astype(int)
        perso = np.abs(a - np.array(modele_sprites.FOND)).sum(-1) > 120
        a[perso] = np.minimum(255, a[perso] + 12)  # SDXL rend des couleurs proches, pas exactes
        Image.fromarray(a.astype(np.uint8), "RGB").save(sorties[0])

    monkeypatch.setattr(diffusion, "sprite_dresseur", sprite_dresseur)
    monkeypatch.setattr(diffusion, "verifier", lambda *n: None)
    monkeypatch.setattr(diffusion, "present", lambda nom: nom == "qwen")
    monkeypatch.setattr(diffusion, "decrire", lambda mode, texte, image=None, progress=None:
                        "a young man, light skin, short brown hair, red jacket.")
    assert cm.modeles_de_marche() == [cm.SANS_IA, f"marche_ds ({DS})"]
    fichiers = []
    for i in range(2):
        f = tmp_path / f"ancien_{i}.png"
        _feuille_gba().save(f)
        fichiers.append(str(f))
    msg, planches, archive, dossier = cm.convertir(fichiers, GBA, DS, f"marche_ds ({DS})", 0.6, graine=5,
                                                   progress=no_progress)
    assert "2 feuille(s) convertie(s)" in msg and "marche_ds" in msg
    assert [a["taille"] for a in appels] == [(768, 1024)] * 2 and appels[0]["lora"] == str(lora)
    assert appels[0]["force"] == 0.6 and appels[0]["masque"].min() == 255 and [a["graines"] for a in appels] == [[5], [6]]
    assert appels[0]["prompt"].startswith("pixelsheet, walking sprite sheet, 3 by 4 grid, a young man, light skin")
    a = np.array(Image.open(Path(dossier) / "resultats" / "ancien_0.png"))
    opaques = {tuple(p) for p in a[a[..., 3] > 0][:, :3]}
    assert opaques <= set(COULEURS.values()) | {(0, 0, 0), tuple(modele_sprites.px.CONTOUR_DEFAUT)}
    assert tuple(a[16 + 32, 16, :3]) == COULEURS["bas"]  # ligne 2 (bas) du style DS
    with pytest.raises(gr.Error, match="a appris le style"):
        cm.convertir(fichiers, DS, GBA, f"marche_ds ({DS})", progress=no_progress)
