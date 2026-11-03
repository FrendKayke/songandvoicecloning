# Backlog — fonctionnalités à faire

Idées retenues par l'utilisateur (03/11/2026), pas encore commencées. Chacune deviendra un onglet (une page = une
fonction) ; aucune ne demande de carte graphique sauf mention.

## Audio et vidéo
- [x] **Convertisseur universel audio / vidéo** (fait le 03/11 : groupe « 🔧 Audio / vidéo ») : MP4 → MP3, extraire le son d'une vidéo, couper / recoller, changer
  le format, réduire le poids. PyAV est déjà installé (aucune dépendance nouvelle).
- [x] **Montage vidéo simple** (fait le 03/11 : « 🔧 Audio / vidéo → 🎬 Montage ») : assembler des clips avec des fondus, ajouter une musique de fond, un titre, un logo.
- [x] **Nettoyer le fond d'une vidéo** (fait le 03/11 : « 🎬 Vidéo → ✂️ Retirer le fond ») : BiRefNet image par image (fond vert sans fond vert). Carte graphique.
- [ ] **Karaoké** : paroles affichées au rythme de la chanson (Whisper donne déjà les temps des mots).
- [ ] **Traduction et doublage** : transcrire une vidéo (Whisper), traduire (Qwen3-VL), relire avec ta voix dans
  l'autre langue (Chatterbox), remettre en voix off (onglet existant). Carte graphique.
- [ ] **Images → vidéo diaporama** : effet Ken Burns, musique et voix off.

## Scripts
- [ ] **File de tâches pour la nuit** : un `.bat` qui lance chansons, images, 3D… à partir d'un fichier texte.
- [ ] **Rangeur** : exporte toutes les créations d'un projet dans un dossier propre.

## PDF
- [ ] **OCR** des PDF scannés et des photos de documents (Tesseract ou PaddleOCR, licence Apache) : texte cherchable,
  PDF → Word complet (l'onglet « PDF → texte / Word » refuse aujourd'hui les PDF scannés).
