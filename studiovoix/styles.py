"""Catalogue des listes déroulantes de l'onglet « Créer une chanson ».

Chaque choix est un couple (libellé affiché en français, termes envoyés à ACE-Step en anglais) :
ACE-Step a appris avec des descriptions en anglais (voir docs/en/Tutorial.md de son dépôt, « Caption »).
Les listes acceptent aussi une saisie libre, envoyée telle quelle.
"""

GENRES = [
    ("Pop", "pop"),
    ("Rock", "rock"),
    ("Pop rock", "pop rock"),
    ("Chanson française", "French chanson"),
    ("Variété française", "French variété pop"),
    ("Hip-hop / rap", "hip-hop, rap"),
    ("Trap", "trap"),
    ("R&B", "R&B"),
    ("Soul", "soul"),
    ("Funk", "funk"),
    ("Disco", "disco"),
    ("Jazz", "jazz"),
    ("Blues", "blues"),
    ("Reggae", "reggae"),
    ("Électro / EDM", "electronic dance music, EDM"),
    ("House", "house"),
    ("Techno", "techno"),
    ("Drum and bass", "drum and bass"),
    ("Dubstep", "dubstep"),
    ("Synthwave", "synthwave, retrowave"),
    ("8-bit / chiptune", "8-bit chiptune, retro video game music"),
    ("Lo-fi hip-hop", "lo-fi hip-hop"),
    ("Ambient", "ambient"),
    ("Metal", "heavy metal"),
    ("Punk", "punk rock"),
    ("Folk", "folk"),
    # Musique ancienne : description positive seulement (une négation ajoute l'instrument chez ACE-Step) ;
    # « medieval » seul est trop vague et retombait sur une production pop (basse et batterie, constaté)
    ("Médiéval / musique ancienne", "medieval folk music, early music, acoustic period instruments, lute, "
                                    "hurdy-gurdy, recorder flute, frame drum"),
    ("Chanson de taverne", "medieval tavern drinking song, rowdy folk sing-along, acoustic, hand claps"),
    ("Celtique", "celtic folk, bodhran, tin whistle, fiddle"),
    ("Country", "country"),
    ("Classique / orchestral", "classical, orchestral"),
    ("Musique de film", "cinematic film score"),
    ("Bossa nova", "bossa nova"),
    ("Latino / salsa", "latin, salsa"),
    ("Afrobeat", "afrobeat"),
    ("Gospel", "gospel"),
    ("K-pop", "K-pop"),
    ("J-pop", "J-pop"),
]

STYLES = [
    ("Années 60", "1960s"),
    ("Années 70", "1970s"),
    ("Années 80", "1980s"),
    ("Années 90", "1990s"),
    ("Années 2000", "2000s"),
    ("Rétro / vintage", "retro, vintage"),
    ("Moderne", "modern"),
    ("Lo-fi", "lo-fi"),
    ("Production studio soignée", "studio-polished, high-fidelity"),
    ("Enregistrement live", "live recording"),
    ("Acoustique", "acoustic"),
    ("Minimaliste", "minimalist"),
    ("Épique", "epic"),
    ("Cinématique", "cinematic"),
    ("Expérimental", "experimental"),
    ("Dansant", "danceable"),
    ("Tempo lent", "slow tempo"),
    ("Tempo rapide", "fast tempo"),
    ("Son de console de jeu (8-bit)", "NES sound chip, square wave, bitcrushed"),
    ("Son années 80 synthétique", "80s synth-pop sound"),
    ("Chambre (intime)", "bedroom pop, intimate"),
    ("Médiéval (instruments d'époque)", "medieval renaissance style, period acoustic instruments, unplugged ensemble"),
]

INSTRUMENTS = [
    ("Guitare acoustique", "acoustic guitar"),
    ("Guitare électrique", "electric guitar"),
    ("Guitare saturée", "distorted electric guitar"),
    ("Basse électrique", "electric bass"),
    ("Contrebasse", "upright bass"),
    ("Piano", "piano"),
    ("Piano électrique (Rhodes)", "Rhodes electric piano"),
    ("Orgue", "organ"),
    ("Synthé lead", "synth lead"),
    ("Nappes de synthé", "synth pads"),
    ("Synthé 8-bit (onde carrée)", "square wave synth, chiptune arpeggios"),
    ("Arpèges de synthé", "synth arpeggios"),
    ("Batterie acoustique", "acoustic drums"),
    ("Boîte à rythmes", "drum machine"),
    ("808", "808 drums"),
    ("Percussions", "percussion"),
    ("Cordes", "strings"),
    ("Violon", "violin"),
    ("Violoncelle", "cello"),
    ("Cuivres", "brass section"),
    ("Trompette", "trumpet"),
    ("Saxophone", "saxophone"),
    ("Flûte", "flute"),
    ("Accordéon", "accordion"),
    ("Harpe", "harp"),
    ("Ukulélé", "ukulele"),
    ("Banjo", "banjo"),
    ("Harmonica", "harmonica"),
    ("Chœurs", "choir"),
    ("Luth", "lute"),
    ("Vielle à roue", "hurdy-gurdy"),
    ("Cornemuse", "bagpipes"),
    ("Flûte à bec", "recorder flute"),
    ("Chalemie", "shawm"),
    ("Tambour sur cadre / tambourin", "frame drum, tambourine"),
    ("Mandoline", "mandolin"),
    ("Violon folk", "folk fiddle"),
    ("Harpe celtique", "celtic harp"),
]

AMBIANCES = [
    ("Joyeuse", "happy, joyful"),
    ("Énergique", "energetic"),
    ("Festive", "festive, upbeat"),
    ("Mélancolique", "melancholic"),
    ("Triste", "sad"),
    ("Nostalgique", "nostalgic"),
    ("Romantique", "romantic"),
    ("Calme / apaisante", "calm, soothing"),
    ("Rêveuse", "dreamy"),
    ("Intime", "intimate"),
    ("Sombre", "dark"),
    ("Mystérieuse", "mysterious"),
    ("Inquiétante", "eerie, ominous"),
    ("Épique / héroïque", "epic, heroic"),
    ("Puissante", "powerful"),
    ("Agressive", "aggressive"),
    ("Lumineuse / optimiste", "bright, uplifting"),
    ("Humoristique", "playful, humorous"),
    ("Taverne festive", "rowdy tavern atmosphere, sing-along, hand claps, foot stomping"),
]

CONSIGNES = [
    ("Refrain accrocheur", "catchy chorus"),
    ("Intro instrumentale", "instrumental intro"),
    ("Pont instrumental", "instrumental bridge"),
    ("Solo de guitare", "guitar solo"),
    ("Solo de synthé", "synth solo"),
    ("Montée en puissance", "building up to a powerful climax"),
    ("Fin en fondu", "fade-out ending"),
    ("Voix douce", "soft vocals"),
    ("Voix puissante", "powerful vocals"),
    ("Voix rauque", "raspy vocals"),
    ("Voix chuchotée", "whispered vocals"),
    ("Voix aiguë", "high-pitched vocals"),
    ("Voix grave", "deep low vocals"),
    ("Chœurs en réponse", "backing vocals"),
    ("Beaucoup de réverbération", "lots of reverb, spacious"),
    ("Son sec et proche", "dry close-miked sound"),
    ("Rythmique très marquée", "strong punchy rhythm"),
]

LISTES = {"genre": GENRES, "style": STYLES, "instruments": INSTRUMENTS, "ambiance": AMBIANCES, "extra": CONSIGNES}


def _normal(texte_):
    """minuscules, sans accents ni ponctuation, espaces simples : « Médiévale ! » → « medievale »."""
    import re
    import unicodedata

    t = (texte_ or "").replace("œ", "oe").replace("Œ", "Oe").replace("æ", "ae").replace("Æ", "Ae")  # « chœur »
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode().lower()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", t).split())


# Saisies libres françaises courantes → termes anglais (ACE-Step comprend mal le français dans la description)
SYNONYMES = [
    (r"^(medi[ea]v|moyen ?age|musique (ancienne|medievale))", "medieval folk music, early music, acoustic period "
                                                               "instruments"),
    (r"^(taverne|chanson a boire)", "medieval tavern drinking song, rowdy folk sing-along"),
    (r"^celt", "celtic folk"),
    (r"^renaissance", "renaissance era early music"),
    (r"^(baroque)", "baroque"),
]


def musique(selection) -> str:
    """Comme texte(), pour la description d'ACE-Step : une saisie libre qui est un libellé du catalogue (accents et
    majuscules ignorés) ou un synonyme français courant est remplacée par ses termes anglais."""
    import re

    if selection is None:
        return ""
    valeurs = [selection] if isinstance(selection, str) else list(selection)
    libelles = {_normal(lib): termes for liste in LISTES.values() for lib, termes in liste}
    anglais = {termes for liste in LISTES.values() for _, termes in liste}
    sortie = []
    for v in valeurs:
        v = (v or "").strip()
        if not v:
            continue
        if v not in anglais:
            n = _normal(v)
            v = libelles.get(n) or next((en for motif, en in SYNONYMES if re.search(motif, n)), v)
        if v not in sortie:
            sortie.append(v)
    return ", ".join(sortie)


def texte(selection) -> str:
    """Une sélection de liste (liste de valeurs, saisies libres comprises) ou un texte → texte unique."""
    if selection is None:
        return ""
    if isinstance(selection, str):
        return selection.strip()
    return ", ".join(s.strip() for s in selection if s and s.strip())


# Styles d'image tapés en français (saisie libre des listes « Style ») → termes anglais. « dessin » passait tel quel
# derrière le style coché par défaut (« flat vector app icon… ») et les images n'avaient rien d'un dessin (constaté
# chez l'utilisateur). Motifs cherchés dans l'ordre dans le texte normalisé (sans accents) ; un passage reconnu est
# retiré avant les motifs suivants (« dessin animé » n'est ni « dessin » ni « animé »).
SYNONYMES_IMAGE = [
    (r"(dessin anime|cartoon)", "cartoon style, bold clean outlines, simple shapes, bright flat colors"),
    (r"aquarel", "watercolor painting, soft washes of color, visible paper texture, bleeding edges"),
    (r"gouache", "gouache painting, flat opaque colors, visible brush strokes"),
    (r"(peinture a l ?huile|huile)", "oil painting, visible brush strokes, rich textured paint"),
    (r"acryl", "acrylic painting, bold brush strokes"),
    (r"pastel", "soft pastel drawing, chalky texture, pastel colors"),
    (r"fusain", "charcoal drawing, smudged shading, rough strokes"),
    (r"(crayon de couleur|crayons de couleur)", "colored pencil drawing, visible pencil strokes, paper texture"),
    (r"(crayon|graphite|croquis|esquisse)", "pencil sketch, hand-drawn graphite lines, cross-hatching shading"),
    (r"(encre|plume)", "ink drawing, bold black ink lines, hatching"),
    (r"(bande dessinee|^bd\b| bd\b|comic)", "comic book illustration, bold ink outlines, flat colors"),
    (r"(manga|anime)", "anime manga style, clean line art, cel shading"),
    (r"(kawaii|mignon)", "cute kawaii style, rounded shapes, pastel colors"),
    (r"(\btraits?\b|contour|line ?art)", "clean line art, uniform outlines"),
    (r"(dessin|dessine|illustration a la main)", "hand-drawn illustration, visible pencil and ink lines, sketchy "
                                                  "drawing style"),
    (r"(icone|icon)", "app icon, single centered subject"),
    (r"(vectoriel|\bplate?s?\b|flat)", "flat vector illustration, simple shapes, solid colors"),
    (r"pixel", "pixel art, crisp pixels, limited palette"),
    (r"low ?poly", "low poly 3D, faceted geometric shapes"),
    (r"(isometrique|isometric)", "isometric view"),
    (r"(^3d|\b3d\b|rendu 3d)", "3D render, soft lighting"),
    (r"(photo|realiste|photorealiste)", "photorealistic photograph, natural light, sharp focus"),
    (r"noir et blanc", "black and white, monochrome"),
    (r"(vitrail)", "stained glass art, lead lines, glowing colored glass"),
    (r"(papier decoupe|papercut)", "paper cut-out art, layered paper, soft shadows"),
    (r"(neon)", "neon glow, dark background, glowing lines"),
    (r"(retro|vintage)", "vintage retro illustration, muted colors, grain"),
]


def image(selection, catalogue=()) -> str:
    """Comme texte(), pour un prompt d'image : une saisie libre égale à un libellé de `catalogue` (couples libellé
    → termes ; accents et casse ignorés) est remplacée par ses termes, et un style tapé en français (« dessin »,
    « aquarelle »…) par ses termes anglais (SYNONYMES_IMAGE). Le reste passe tel quel."""
    import re

    if selection is None:
        return ""
    valeurs = [selection] if isinstance(selection, str) else list(selection)
    libelles = {_normal(lib): termes for lib, termes in catalogue}
    anglais = {termes for _, termes in catalogue}
    sortie = []
    for v in valeurs:
        v = (v or "").strip()
        if not v:
            continue
        if v not in anglais:
            n = _normal(v)
            trouves, reste = [], n
            for motif, en in SYNONYMES_IMAGE:
                m = re.search(motif, reste)
                if m:
                    trouves.append(en)
                    reste = reste[:m.start()] + " " + reste[m.end():]
            v = libelles.get(n) or ", ".join(dict.fromkeys(trouves)) or v
        if v not in sortie:
            sortie.append(v)
    return ", ".join(sortie)
