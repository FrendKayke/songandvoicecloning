"""Onglet « Corriger un modèle 3D » : l'image d'un modèle est corrigée d'après un texte (FLUX.2 klein), une vue de dos
est créée ou importée, puis le modèle est refait avec la même graine (Hunyuan3D-2 / 2mv). Nouvelle création."""
import gradio as gr

from .. import modele3d
from ..outils import open_folder
from .commun import espace_de_noms
from .outils_3d import glb, infos, maj_liste


def _charger(dossier):
    """Images, graine et réglages du modèle choisi."""
    if not dossier:
        return None, None, 0, "modele", modele3d.QUALITE_DEFAUT, True, ["glb"], None, ""
    i = infos(dossier)
    try:
        image, dos, graine = modele3d.reprendre_images(dossier)
    except gr.Error:
        return None, None, 0, "modele", modele3d.QUALITE_DEFAUT, True, ["glb"], None, "Ce dossier n'est pas un modèle 3D."
    qualite = i.get("qualite") if i.get("qualite") in modele3d.QUALITES else modele3d.QUALITE_DEFAUT
    note = (f"**{i.get('nom', 'modele')}** : graine {graine}, qualité « {qualite} »"
            + (", avec texture" if i.get("texture") else ", sans texture")
            + (", vue de dos présente" if dos else "")
            + (f" ; révision précédente : « {i['revision']} »" if i.get("revision") else "") + ".")
    return (image, dos, graine, i.get("nom", "modele"), qualite, bool(i.get("texture", True)),
            i.get("formats") or ["glb"], i.get("projet"), note)


def _corriger(vue, image, dos, revision, progress=gr.Progress()):
    """Corrige la vue choisie (face ou dos). Renvoie (image, dos, message, révision notée)."""
    if modele3d.VUES.get(vue) == "dos":
        if not dos:
            raise gr.Error("Il n'y a pas encore de vue de dos : clique sur « Créer la vue de dos » (ou importe-la).")
        nouvelle, msg, note = modele3d.corriger_image(dos, revision, progress=progress)
        return image, nouvelle, msg, note
    nouvelle, msg, note = modele3d.corriger_image(image, revision, progress=progress)
    return nouvelle, dos, msg, note


def _refaire(dossier, image, dos, graine, nom, qualite, texture, formats, projet, note, progress=gr.Progress()):
    """Refait le modèle avec la même graine, en notant la révision et le modèle d'origine."""
    if not dossier:
        raise gr.Error("Choisis le modèle à corriger (crée-en d'abord un dans « Modèle 3D d'une image »).")
    i = infos(dossier)
    return modele3d.generer(image, nom, qualite, texture, graine, formats, i.get("description"),
                            i.get("description_fr"), i.get("image_graine"), projet, dos, note, dossier,
                            progress=progress)


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    gr.Markdown(
        "**Corrige un modèle 3D.** Hunyuan3D sculpte d'après l'image : pour corriger un modèle, on corrige son image, "
        "puis on refait le modèle **avec la même graine** (reprise automatiquement). Écris ce qui ne va pas "
        "(« refais l'arc : un long arc en bois, entier, avec sa corde »), FLUX.2 klein corrige l'image sans toucher "
        "au reste, tu vérifies, puis « Refaire le modèle 3D ». **Ce qui est dans le dos** (arc, cape, carquois) se "
        "voit mal de face et le modèle l'invente : crée une **vue de dos** (ou importe la tienne), corrige-la si "
        "besoin, et la forme sera sculptée d'après les deux vues (Hunyuan3D-2mv). Le résultat est une nouvelle "
        "création ; l'originale reste dans la galerie."
    )
    with gr.Row():
        rev_modele = gr.Dropdown([], label="Modèle à corriger (le plus récent en premier)", scale=4)
        rev_maj = gr.Button("🔄 Actualiser", scale=0)
    rev_note = gr.Markdown()
    with gr.Row():
        rev_image = gr.Image(type="filepath", label="Image de face (corrigée ici)", height=260)
        rev_dos = gr.Image(type="filepath", label="Vue de dos (facultative : créée ici, ou importe la tienne)",
                           height=260)
    with gr.Row():
        rev_revision = gr.Textbox(label="Ce qu'il faut corriger (français)", lines=2, scale=3,
                                  placeholder="refais l'arc dans le dos : un long arc en bois entier, avec sa corde")
        rev_vue = gr.Radio(list(modele3d.VUES), value=next(iter(modele3d.VUES)), label="Que corriger", scale=1)
    with gr.Row():
        btn_rev_corriger = gr.Button("✏️ Corriger l'image")
        btn_rev_corriger_3d = gr.Button("✏️ Corriger puis refaire le modèle 3D", variant="primary")
    with gr.Row():
        rev_precision_dos = gr.Textbox(label="Vue de dos : ce qui doit s'y voir (facultatif, français)", scale=3,
                                       placeholder="l'arc et le carquois bien visibles dans le dos")
        btn_rev_dos = gr.Button("🔙 Créer la vue de dos", scale=1)
    rev_msg = gr.Markdown()
    with gr.Accordion("Réglages repris du modèle (modifiables)", open=False):
        with gr.Row():
            rev_nom = gr.Textbox(label="Nom", value="modele")
            rev_graine = gr.Number(value=0, precision=0, label="Graine (celle du modèle : seul le nécessaire change)")
            rev_projet = gr.Textbox(label="Projet de jeu (facultatif)")
        with gr.Row():
            rev_qualite = gr.Radio(list(modele3d.QUALITES), value=modele3d.QUALITE_DEFAUT, label="Qualité")
            rev_texture = gr.Checkbox(value=True, label="Peindre la texture")
            rev_formats = gr.CheckboxGroup(modele3d.FORMATS, value=["glb"], label="Formats")
    rev_revision_note = gr.State()
    btn_rev = gr.Button("🧊 Refaire le modèle 3D (même graine)", variant="primary")
    rev_statut = gr.Markdown()
    rev_dossier = gr.State()
    with gr.Row():
        rev_glb = gr.Model3D(label="Modèle 3D", clear_color=(0.92, 0.92, 0.92, 1.0), scale=3)
        with gr.Column(scale=1):
            rev_detouree = gr.Image(label="Image détourée", interactive=False)
            rev_fichiers = gr.File(label="Fichiers produits", file_count="multiple", interactive=False)
            btn_rev_dossier = gr.Button("📂 Ouvrir le dossier")
    return espace_de_noms(locals())


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    charges = [c.rev_image, c.rev_dos, c.rev_graine, c.rev_nom, c.rev_qualite, c.rev_texture, c.rev_formats,
               c.rev_projet, c.rev_note]
    for evt in (c.rev_maj.click, demo.load, c.onglet.select):
        evt(maj_liste, c.rev_modele, c.rev_modele)
    c.rev_modele.change(_charger, c.rev_modele, charges).then(glb, c.rev_modele, c.rev_glb)
    corriger = ([c.rev_vue, c.rev_image, c.rev_dos, c.rev_revision],
                [c.rev_image, c.rev_dos, c.rev_msg, c.rev_revision_note])
    entrees = [c.rev_modele, c.rev_image, c.rev_dos, c.rev_graine, c.rev_nom, c.rev_qualite, c.rev_texture,
               c.rev_formats, c.rev_projet, c.rev_revision_note]
    sorties = [c.rev_statut, c.rev_glb, c.rev_detouree, c.rev_fichiers, c.rev_dossier]
    # après un modèle refait : il devient le modèle à corriger (les révisions s'enchaînent), et il est proposé aux
    # autres onglets (squelette, alléger)
    suite = lambda evt: evt.success(maj_liste, c.rev_dossier, c.rev_modele).then(  # noqa: E731
        lambda d: (gr.update(value=d),) * 2, c.rev_dossier, [o.rigging_3d.rig_modele, o.alleger_3d.al_modele])
    c.btn_rev_corriger.click(_corriger, *corriger)
    suite(c.btn_rev_corriger_3d.click(_corriger, *corriger).success(_refaire, entrees, sorties))
    suite(c.btn_rev.click(_refaire, entrees, sorties))
    c.btn_rev_dos.click(modele3d.creer_vue_de_dos, [c.rev_image, c.rev_precision_dos], [c.rev_dos, c.rev_msg])
    c.btn_rev_dossier.click(lambda d: open_folder(d) if d else None, c.rev_dossier)
