"""Onglet « Personnages récurrents » (groupe Jeu) : images de référence d'un personnage (1 à 4), reprises par FLUX.2
klein pour le montrer avec la même apparence sur toutes les cartes du projet."""
import gradio as gr

from .. import personnages, projets
from .commun import espace_de_noms

CONFIRMER_PERSONNAGE = ("(p, n) => [p, (n && confirm('Supprimer le personnage « ' + n + ' » et ses images de référence ?'))"
                        " ? n : null]")


def construire():
    """Composants de l'onglet (dans l'onglet ouvert par l'appelant)."""
    gr.Markdown(
        "**Le même héros sur plusieurs cartes.** Un personnage = 1 à 4 images de référence : un portrait net du "
        "visage, une image en pied, la tenue… Crée-les d'abord dans « Illustrations de cartes » (illustration sur "
        "fond simple, clique sur la variante réussie, puis ici « Ajouter la variante gardée »), ou importe tes "
        "propres images. Ensuite, dans « Illustrations de cartes », choisis-le dans « Personnage de la scène » : "
        "chaque nouvelle scène le montre avec la même apparence. Décris la pose et le décor dans la scène, pas son "
        "apparence."
    )
    with gr.Row():
        pc_projet = gr.Dropdown(projets.tous() or ["mon-jeu"], value=(projets.tous() or ["mon-jeu"])[0],
                                allow_custom_value=True, label="Projet de jeu")
        perso_nom = gr.Dropdown([], value=None, allow_custom_value=True, label="Personnage (choisis ou tape un nom)")
    perso_images = gr.File(label="Importer des images du personnage", file_count="multiple", file_types=["image"],
                           type="filepath")
    with gr.Row():
        btn_perso_variante = gr.Button("⭐ Ajouter la variante gardée dans « Illustrations de cartes »")
        btn_perso_import = gr.Button("📥 Ajouter les images importées", variant="primary")
        btn_perso_suppr = gr.Button("🗑️ Supprimer ce personnage", variant="stop")
    perso_msg = gr.Markdown()
    perso_galerie = gr.Gallery(label="Images de référence", columns=4, height=260, object_fit="contain")
    return espace_de_noms(locals())


def brancher(c, demo, o):
    """Événements de l'onglet ; o donne accès aux composants des autres onglets."""
    noms_perso = lambda p, n: gr.update(choices=personnages.liste(p), value=n)  # noqa: E731
    # même projet que l'onglet des illustrations (mis à jour dans les deux sens)
    o.illustrations.ill_projet.change(lambda p: gr.update(value=p), o.illustrations.ill_projet, c.pc_projet)
    c.pc_projet.change(lambda p: (gr.update(choices=personnages.liste(p), value=None), []), c.pc_projet,
                       [c.perso_nom, c.perso_galerie])
    demo.load(lambda p: gr.update(choices=personnages.liste(p)), c.pc_projet, c.perso_nom)
    c.perso_nom.change(personnages.afficher, [c.pc_projet, c.perso_nom], c.perso_galerie)
    # après un ajout ou une suppression : liste « Personnage de la scène » des illustrations à jour
    c.btn_perso_variante.click(lambda p, n, i: personnages.ajouter(p, n, [i] if i else []),
                               [c.pc_projet, c.perso_nom, o.illustrations.ill_choisie],
                               [c.perso_msg, c.perso_galerie, o.illustrations.ill_personnage]).then(
        noms_perso, [c.pc_projet, c.perso_nom], c.perso_nom)
    c.btn_perso_import.click(personnages.ajouter, [c.pc_projet, c.perso_nom, c.perso_images],
                             [c.perso_msg, c.perso_galerie, o.illustrations.ill_personnage]).then(
        noms_perso, [c.pc_projet, c.perso_nom], c.perso_nom)
    c.btn_perso_suppr.click(personnages.supprimer, [c.pc_projet, c.perso_nom],
                            [c.perso_msg, c.perso_galerie, o.illustrations.ill_personnage],
                            js=CONFIRMER_PERSONNAGE).then(
        lambda p: gr.update(choices=personnages.liste(p), value=None), c.pc_projet, c.perso_nom)
