"""Kit de dresseur : sprites d'exemple recolorés aux couleurs de la personne (faux klein qui recolore)."""
import json
from pathlib import Path

import gradio as gr
import numpy as np
import pytest
from PIL import Image, ImageDraw

from conftest import no_progress
from studiovoix import diffusion, dresseur

GBA, DS = list(dresseur.STYLES)
PEAU, CHEVEUX, CHEVEUX_OMBRE, HABIT, HABIT_OMBRE = (240, 200, 160), (120, 70, 30), (80, 45, 20), (200, 40, 40), \
    (140, 20, 20)


def _bonhomme(d, x, y, cw, ch):
    d.rectangle((x + cw // 4, y + ch // 3, x + 3 * cw // 4 - 1, y + ch - 1), fill=HABIT + (255,))
    d.rectangle((x + cw // 4, y + 2 * ch // 3, x + 3 * cw // 4 - 1, y + ch - 1), fill=HABIT_OMBRE + (255,))
    d.ellipse((x + cw // 3, y + 1, x + 2 * cw // 3, y + ch // 3), fill=PEAU + (255,))
    d.rectangle((x + cw // 3, y + 1, x + 2 * cw // 3, y + ch // 8), fill=CHEVEUX + (255,))
    d.rectangle((x + cw // 3, y + 1, x + cw // 2, y + ch // 16), fill=CHEVEUX_OMBRE + (255,))
    d.rectangle((x + cw // 4, y + ch - 1, x + 3 * cw // 4 - 1, y + ch - 1), fill=(0, 0, 0, 255))  # contour


def _feuille(cols, rows, cw, ch, k, fond=None):
    """Sprites d'exemple : un bonhomme par case (cheveux, peau, habit avec ombre, contour), agrandis ×k."""
    im = Image.new("RGBA", (cols * cw, rows * ch), fond or (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for r in range(rows):
        for c in range(cols):
            _bonhomme(d, c * cw + (c % 3) - 1, r * ch, cw, ch)  # poses un peu différentes d'une case à l'autre
    return im.resize((im.width * k, im.height * k), Image.Resampling.NEAREST)


NOUVELLES = {PEAU: (250, 220, 200), CHEVEUX: (230, 200, 60), CHEVEUX_OMBRE: (190, 160, 40), HABIT: (60, 120, 220),
             HABIT_OMBRE: (60, 120, 220)}  # deux nuances de l'habit peintes pareil : l'ombrage doit rester


@pytest.fixture
def faux_klein(env, monkeypatch, tmp_path):
    appels = []

    def personnage(prompt, references, sorties, graines, largeur, hauteur, progress=None):
        appels.append({"prompt": prompt, "references": [str(r) for r in references], "taille": (largeur, hauteur)})
        if "Recolor" not in prompt:  # sprite de face dessiné d'après la photo
            im = Image.new("RGB", (largeur, hauteur), (255, 255, 255))
            d = ImageDraw.Draw(im)
            d.rectangle((300, 300, 470, 1000), fill=(30, 60, 200))
            d.ellipse((320, 80, 450, 300), fill=(240, 190, 150))
        else:  # recoloriage : chaque couleur connue remplacée, fond et contours gardés (dessin un peu flou)
            a = np.array(Image.open(references[0]).convert("RGB"))
            b = a.copy()
            for ancienne, nouvelle in NOUVELLES.items():
                b[(a == ancienne).all(-1)] = nouvelle
            im = Image.fromarray(b, "RGB").resize((largeur, hauteur), Image.Resampling.NEAREST)
        for s in sorties:
            im.save(s)

    def sprite_dresseur(prompts, sorties, graines, largeur=1024, hauteur=1024, etapes=30, guidage=7.0, negatif=None,
                        lora_echelle=1.0, progress=None):
        appels.append({"prompt": prompts, "sorties": [str(s) for s in sorties], "graines": list(graines)})
        for s, g in zip(sorties, graines):
            im = Image.new("RGB", (largeur, hauteur), (255, 255, 255))
            d = ImageDraw.Draw(im)
            d.rectangle((380, 420, 640, 960), fill=(200, 40, 40))
            d.ellipse((400, 200, 620, 440), fill=(240, 200, 160))
            d.rectangle((400, 180, 620, 280), fill=(120 + g, 70, 30))
            im.save(s)

    monkeypatch.setattr(diffusion, "sprite_dresseur", sprite_dresseur)
    monkeypatch.setattr(diffusion, "personnage", personnage)
    monkeypatch.setattr(diffusion, "verifier", lambda *n: None)
    monkeypatch.setattr(diffusion, "present", lambda nom: False)
    photo = tmp_path / "photo.jpg"
    Image.new("RGB", (300, 400), (90, 120, 90)).save(photo)
    return appels, photo, tmp_path


def _bibliotheque(tmp, style=GBA):
    face, lancer, marche = tmp / "trainer108.png", tmp / "trback002.png", tmp / "trchar113.png"
    _feuille(1, 1, 40, 56, 2).save(face)
    _feuille(5, 1, 64, 64, 2).save(lancer)
    _feuille(4, 4, 16, 24, 2, fond=(120, 160, 200, 255)).save(marche)  # fond uni, pas de transparence
    for feuille, f in ((dresseur.FACE, face), (dresseur.LANCER, lancer), (dresseur.MARCHE, marche)):
        msg, *listes = dresseur.ajouter_gabarits(style, feuille, [str(f)])
        assert "✅" in msg
    return listes


def test_kit_recolore_les_exemples(faux_klein):
    appels, photo, tmp = faux_klein
    _bibliotheque(tmp)
    noms = {f: dresseur.liste_gabarits(GBA, f) for f in ("face", "lancer", "marche")}
    assert noms == {"face": ["trainer108"], "lancer": ["trback002"], "marche": ["trchar113"]}
    msg, apercus, fichiers, dossier, face, couleurs, _ = dresseur.creer_kit(
        GBA, str(photo), "la femme", "hair: blond; top: blue", dresseur.FEUILLES, dresseur.RECOLORER,
        "trainer108", "trback002", "trchar113", 7, progress=no_progress)
    d = Path(dossier)
    assert len(appels) == 1 and "hair: blond" in appels[0]["prompt"]  # une seule image pour toutes les feuilles
    for f, nom in (("face", "trainer108"), ("lancer", "trback002"), ("marche", "trchar113")):
        g = dresseur.lire_gabarit(dresseur.chemin_gabarit(GBA, f, nom), GBA, f)
        a = np.array(Image.open(d / f"{f}.png"))
        assert a.shape == g.shape and (a[..., 3] == g[..., 3]).all()  # silhouette exacte de l'exemple
        op = g[..., 3] >= 128
        anciennes = {tuple(p) for p in g[..., :3][op]}
        nouvelles = {tuple(p) for p in a[..., :3][op]}
        assert len(nouvelles) == len(anciennes)  # ombrage gardé, même si klein peint deux nuances pareil
        assert (a[..., :3][(g[..., :3] == 0).all(-1) & op] == 0).all()  # contours noirs gardés
        cheveux = a[..., :3][(g[..., :3] == CHEVEUX).all(-1) & op].astype(int)
        assert (np.abs(cheveux - NOUVELLES[CHEVEUX]).sum(-1) < 60).all()  # teinte peinte, ombrage du gabarit
    assert Image.open(d / "lancer.png").size == (320, 64) and Image.open(d / "marche.png").size == (64, 96)
    assert Image.open(face).size == (64, 64)
    noms_fichiers = {Path(p).name for p in fichiers}
    assert {"face_x2.png", "lancer_x2.gif", "marche_bas_x2.gif", "marche_haut_x2.gif"} <= noms_fichiers
    with Image.open(d / "lancer_x2.gif") as gif:
        assert gif.n_frames == 5
    infos = json.loads((d / "kit.json").read_text(encoding="utf-8"))
    assert infos["feuilles"]["marche"]["exemple"] == "trchar113" and couleurs == "hair: blond; top: blue"


def test_face_par_le_modele_de_sprites(faux_klein, monkeypatch):
    appels, photo, tmp = faux_klein
    _bibliotheque(tmp)
    with pytest.raises(gr.Error, match="décris"):
        dresseur.creer_kit(GBA, None, "", "", [dresseur.FACE], dresseur.LORA, None, None, None, 1, progress=no_progress)
    monkeypatch.setattr(diffusion, "present", lambda nom: nom == "qwen")
    monkeypatch.setattr(diffusion, "decrire", lambda mode, texte, image=None, progress=None:
                        "a woman with long curly brown hair wearing a red jacket." if mode == "sprite" else "x")
    msg, apercus, fichiers, dossier, face, couleurs, _ = dresseur.creer_kit(
        GBA, str(photo), "la femme", "", dresseur.FEUILLES, dresseur.LORA, None, "trback002", "trchar113", 5,
        variantes=3, progress=no_progress)
    d = Path(dossier)
    assert appels[0]["prompt"] == ("a pixel image of a woman with long curly brown hair wearing a red jacket, gen3, "
                                   "full body, standing, simple background")
    assert appels[0]["graines"] == [5, 6, 7] and len(appels[0]["sorties"]) == 3
    assert Image.open(face).size == (64, 64) and (d / "face_2.png").exists() and (d / "face_3_x2.png").exists()
    assert couleurs == "x" and "face_2_x2.png" in {Path(p).name for p in apercus}  # lues pour le recoloriage
    # les feuilles sont recolorées d'après ce sprite (image 2 du recoloriage), couleurs lues sur le sprite
    assert len(appels) == 2 and len(appels[1]["references"]) == 2
    infos = json.loads((d / "kit.json").read_text(encoding="utf-8"))
    assert infos["feuilles"]["face"] == {"mode": "lora", "prompt": appels[0]["prompt"], "variantes": 3}
    import zipfile
    noms = zipfile.ZipFile(d / f"{d.name}.zip").namelist()
    assert "face_2.png" in noms and not any(n.endswith("_source.png") for n in noms)
    # style DS : étiquette gen5, pas de photo mais une description
    _, _, _, dossier, face, _, _ = dresseur.creer_kit(DS, None, "un homme barbu à lunettes", "", [dresseur.FACE],
                                                   dresseur.LORA, None, None, None, 2, variantes=1, progress=no_progress)
    assert "gen5" in appels[-1]["prompt"] and Image.open(face).size == (80, 80)


def test_face_dessinee_et_couleurs_lues(faux_klein, monkeypatch):
    appels, photo, tmp = faux_klein
    _bibliotheque(tmp)
    with pytest.raises(gr.Error, match="couleurs"):
        dresseur.creer_kit(GBA, None, "", "", [dresseur.MARCHE], dresseur.RECOLORER, None, None, "trchar113", 1,
                           progress=no_progress)
    monkeypatch.setattr(diffusion, "present", lambda nom: nom == "qwen")
    lu = []
    monkeypatch.setattr(diffusion, "decrire", lambda mode, texte, image=None, progress=None: lu.append(mode) or
                        ("hair: black; skin: tan" if mode == "couleurs" else "a woman"))
    _, _, _, dossier, face, couleurs, _ = dresseur.creer_kit(
        GBA, str(photo), "la femme", "", dresseur.FEUILLES, dresseur.DESSINER, None, "trback002", "trchar113", 3,
        progress=no_progress)
    assert couleurs == "hair: black; skin: tan" and lu == ["couleurs", "sprite"]
    # face dessinée d'après la photo, puis donnée au recoloriage comme référence des couleurs
    assert len(appels) == 2 and len(appels[1]["references"]) == 2 and "image 2" in appels[1]["prompt"]
    assert Image.open(face).size == (64, 64) and not (Path(dossier) / "face_source.png").name in str(
        __import__("zipfile").ZipFile(Path(dossier) / f"{Path(dossier).name}.zip").namelist())



def test_kit_d_apres_une_simple_description(faux_klein, monkeypatch):
    """Sans photo : la description française est traduite par Qwen3-VL (texte seul), affichée et modifiable ; la
    description modifiée fait foi pour le sprite de face, les couleurs du recoloriage et la feuille de marche."""
    appels, photo, tmp = faux_klein
    _bibliotheque(tmp)
    monkeypatch.setattr(diffusion, "present", lambda nom: nom == "qwen")
    lu = []

    def decrire(mode, texte, image=None, progress=None):
        lu.append((mode, texte, image))
        return {"sprite_texte": "a young woman with long red hair, red jacket and blue shorts.",
                "couleurs_texte": "hair: red; top: green", "sprite": "a man from the photo."}.get(mode, "?")

    monkeypatch.setattr(diffusion, "decrire", decrire)
    with pytest.raises(gr.Error, match="décris"):
        dresseur.preparer_description(None, "  ", progress=no_progress)
    texte = dresseur.preparer_description(None, "une dresseuse rousse, veste rouge, short bleu", progress=no_progress)
    assert texte == "a young woman with long red hair, red jacket and blue shorts."
    assert lu[-1] == ("sprite_texte", "une dresseuse rousse, veste rouge, short bleu", None)
    assert dresseur.preparer_description(str(photo), "", progress=no_progress) == "a man from the photo."
    assert lu[-1][0] == "sprite" and lu[-1][2] == str(photo)
    # description modifiée par l'utilisateur (veste verte) : Qwen ne la réécrit pas, elle part telle quelle à SDXL
    lu.clear()
    modifiee = "a young woman with long red hair, green jacket and blue shorts"
    msg, apercus, fichiers, dossier, face, couleurs, description = dresseur.creer_kit(
        GBA, None, "une dresseuse rousse", "", dresseur.FEUILLES, dresseur.LORA, None, "trback002", "trchar113", 4,
        variantes=1, description=modifiee, progress=no_progress)
    assert appels[0]["prompt"] == f"a pixel image of {modifiee}, gen3, full body, standing, simple background"
    assert couleurs == "hair: red; top: green" and lu == [("couleurs_texte", modifiee, None)]
    assert description == modifiee and Image.open(face).size == (64, 64)
    infos = json.loads((Path(dossier) / "kit.json").read_text(encoding="utf-8"))
    assert infos["description"] == modifiee
    # sans description préparée ni photo : la demande française est traduite avant SDXL
    lu.clear()
    *_, description = dresseur.creer_kit(GBA, None, "une dresseuse rousse", "", [dresseur.FACE], dresseur.LORA,
                                         None, None, None, 4, variantes=1, progress=no_progress)
    assert lu == [("sprite_texte", "une dresseuse rousse", None)]
    assert "a young woman with long red hair" in appels[-1]["prompt"] and description.startswith("a young woman")
    # couleurs lues sans photo : d'après la description, sinon d'après la demande
    assert dresseur.lire_couleurs(None, "une rousse", "", progress=no_progress) == "hair: red; top: green"
    assert lu[-1] == ("couleurs_texte", "une rousse", None)
    dresseur.lire_couleurs(str(photo), "la femme", modifiee, progress=no_progress)
    assert lu[-1] == ("couleurs_texte", modifiee, None)
    with pytest.raises(gr.Error, match="klein"):
        dresseur.creer_kit(GBA, None, "x", "", [dresseur.FACE], dresseur.DESSINER, None, None, None, 1,
                           description=modifiee, progress=no_progress)


def test_ds_lancer_du_gba_et_erreurs(faux_klein):
    appels, photo, tmp = faux_klein
    _bibliotheque(tmp)
    marche_ds, mauvaise = tmp / "marche_ds.png", tmp / "mauvaise.png"
    _feuille(3, 4, 32, 32, 1).save(marche_ds)
    _feuille(3, 3, 30, 30, 1).save(mauvaise)
    msg, *_ = dresseur.ajouter_gabarits(DS, dresseur.MARCHE, [str(mauvaise), str(marche_ds)])
    assert "3 colonnes et 4 lignes" in msg and "1 exemple(s)" in msg
    assert dresseur.liste_gabarits(DS, "lancer") == ["gba:trback002"]  # celui du GBA sert à la DS
    with pytest.raises(gr.Error, match="Pas de sprite d'exemple"):
        dresseur.creer_kit(DS, str(photo), "", "hair: red", dresseur.FEUILLES, dresseur.RECOLORER, None,
                           "gba:trback002", "marche_ds", 1, progress=no_progress)
    _, _, _, dossier, face, _, _ = dresseur.creer_kit(DS, str(photo), "", "hair: red", [dresseur.LANCER, dresseur.MARCHE],
                                                   dresseur.RECOLORER, None, "gba:trback002", "marche_ds", 1,
                                                   progress=no_progress)
    d = Path(dossier)
    assert face is None and Image.open(d / "lancer.png").size == (400, 80)  # cases agrandies à 80×80
    assert Image.open(d / "marche.png").size == (96, 128) and (d / "marche_droite_x2.gif").exists()
    assert dresseur.apercu_gabarit(DS, "lancer", "gba:trback002").size == (800, 160)


def test_nuances_gardees():
    proches = dresseur._garder_les_nuances({(200, 40, 40): (60, 120, 220), (140, 20, 20): (62, 121, 221),
                                            (0, 0, 0): (0, 0, 0)})
    clair, fonce = np.array(proches[(200, 40, 40)]), np.array(proches[(140, 20, 20)])
    assert dresseur._lum(fonce) < dresseur._lum(clair) - 0.1 and proches[(0, 0, 0)] == (0, 0, 0)


def test_photo_du_kit_envoyee_avec_son_chemin_complet(faux_klein, monkeypatch):
    """Un Path a un attribut `.name` (le nom seul) : il était pris pour l'objet fichier de Gradio et le moteur
    recevait « photo.jpg » (FileNotFoundError chez l'utilisateur, kit avec un modèle de feuilles entraîné)."""
    from studiovoix.outils import chemin_fichier

    appels, photo, tmp = faux_klein
    _bibliotheque(tmp)
    assert chemin_fichier(photo) == photo and chemin_fichier(str(photo)) == str(photo)
    assert chemin_fichier(type("Fichier", (), {"name": "/tmp/gradio/a.png"})()) == "/tmp/gradio/a.png"
    monkeypatch.setattr(diffusion, "present", lambda nom: nom == "qwen")
    vus = []

    def decrire(mode, texte, image=None, progress=None):
        if image is not None:
            assert Path(image).is_absolute() and Path(image).is_file(), image
            vus.append(mode)
        return "hair: red; top: blue" if "couleurs" in mode else "a woman with red hair"

    monkeypatch.setattr(diffusion, "decrire", decrire)
    dresseur.creer_kit(GBA, str(photo), "la femme", "", dresseur.FEUILLES, dresseur.LORA, None, "trback002",
                       "trchar113", 3, variantes=1, progress=no_progress)
    assert vus == ["couleurs", "sprite"]


def test_supprimer_un_exemple(faux_klein):
    appels, photo, tmp = faux_klein
    _bibliotheque(tmp)
    assert dresseur.supprimer_gabarit(GBA, None, "marche") == ("", gr.update())  # confirmation refusée
    msg, maj = dresseur.supprimer_gabarit(GBA, "trchar113", "marche")
    assert "retiré" in msg and maj["choices"] == [] and maj["value"] is None
    assert dresseur.liste_gabarits(GBA, "marche") == []
    with pytest.raises(gr.Error, match="introuvable"):
        dresseur.supprimer_gabarit(GBA, "trchar113", "marche")
    with pytest.raises(gr.Error, match="appartient au style"):  # lancer GBA prêté à la DS
        dresseur.supprimer_gabarit(DS, "gba:trback002", "lancer")
    assert dresseur.liste_gabarits(GBA, "lancer") == ["trback002"]


def test_une_teinte_par_zone_du_gabarit():
    """Les nuances d'une même matière prennent une seule teinte (feuille de lancer de l'utilisateur, 31/10 : veste à
    moitié orange, à moitié kaki), avec l'ombrage du gabarit ; deux matières voisines de teintes différentes restent
    séparées."""
    clair, moyen, ombre = (90, 140, 230), (60, 110, 200), (30, 70, 150)  # veste bleue, trois nuances
    peau = (240, 200, 160)
    g = np.zeros((12, 12, 4), np.uint8)
    g[:4, :, :3] = peau
    g[4:8, :, :3] = clair
    g[8:10, :, :3] = moyen
    g[10:, :, :3] = ombre
    g[..., 3] = 255
    zones = dresseur.materiaux(g)
    assert {frozenset(z) for z in zones} == {frozenset({clair, moyen, ombre}), frozenset({peau})}
    # klein peint la veste claire en orange, l'ombre en kaki : toute la veste devient orange (la plus peinte)
    peint = g[..., :3].copy()
    peint[4:10] = (230, 140, 60)
    peint[10:] = (120, 120, 50)
    peint[:4] = (250, 215, 190)
    k = 4
    image = Image.fromarray(np.repeat(np.repeat(peint, k, 0), k, 1), "RGB")
    n = dresseur.correspondances(g, image, k)
    import colorsys
    teintes = [colorsys.rgb_to_hls(*(np.array(n[c]) / 255))[0] for c in (clair, moyen, ombre)]
    assert max(teintes) - min(teintes) < 0.02 and 0.04 < teintes[0] < 0.12  # une seule teinte, orange
    lum = [dresseur._lum(n[c]) for c in (clair, moyen, ombre)]
    assert lum[0] > lum[1] > lum[2]  # ombrage du gabarit gardé
    assert np.abs(np.array(n[peau]) - (250, 215, 190)).sum() < 30
