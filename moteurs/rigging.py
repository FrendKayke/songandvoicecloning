"""Moteur « rigging » — script exécuté DANS l'environnement de rigging (StudioVoix\\rigging\\.venv), depuis le dossier
du code d'UniRig (StudioVoix\\rigging\\UniRig : ses configurations sont relatives à ce dossier).

UniRig (VAST-AI-Research, MIT, SIGGRAPH 2025) : squelette (transformeur autorégressif, 1,4 Go) puis poids de peau
(4,4 Go) d'un modèle 3D, pour l'animer dans un jeu ou dans Blender. Les étapes sont celles des scripts de son
README (launch/inference/*.sh, commit épinglé dans installer.ps1), lancées une à une avec leur code de sortie et
leur fichier de sortie vérifiés :
  1. src.data.extract  : modèle → <travail>/modele/raw_data.npz (Blender en module, bpy 4.2)
  2. run.py squelette  : → squelette.fbx
  3. src.data.extract  : squelette.fbx → <travail>/squelette/raw_data.npz
  4. run.py peau       : → peau.fbx (maillage + squelette + poids)
  5. src.inference.merge : squelette et poids reportés sur le modèle d'origine (textures gardées) → GLB

    python rigging.py animer <tache.json>    {entree, sortie (GLB), fbx (copie du résultat FBX, facultatif),
                                              graine, faces}
    python rigging.py telecharger            poids d'UniRig + configuration d'OPT-350M (lue par le squelette)

Sortie : « PROGRESSION i/n … », « RESULTAT <json> », « ERREUR : message », « TERMINE <fichier> ».
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

UNIRIG = "VAST-AI/UniRig"
# chemins des poids : src/inference/download.py d'UniRig (MAP), configurations quick_inference_*.yaml
POIDS = ["skeleton/articulation-xl_quantization_256/model.ckpt", "skin/articulation-xl/model.ckpt"]
OPT = "facebook/opt-350m"  # configuration du transformeur du squelette (AutoConfig.from_pretrained)
SUFFIXES = "obj,fbx,FBX,dae,glb,gltf,vrm"
TACHE_SQUELETTE = "configs/task/quick_inference_skeleton_articulationxl_ar_256.yaml"
TACHE_PEAU = "configs/task/quick_inference_unirig_skin.yaml"
ETAPES = 5


def _erreur(msg, code=2):
    print(f"ERREUR : {msg}", flush=True)
    sys.exit(code)


def _executer(args, attendu, quoi):
    """Lance une commande d'UniRig (même Python, dossier courant = code d'UniRig) ; ses lignes passent dans le
    journal du moteur. Erreur claire si elle échoue ou ne produit pas son fichier."""
    debut = time.time()
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen([sys.executable, *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", env=env)
    fin = []
    for ligne in proc.stdout:
        ligne = ligne.rstrip()
        fin = (fin + [ligne])[-15:]
        print(f"  {ligne}", flush=True)
    proc.wait()
    if proc.returncode != 0 or not Path(attendu).is_file():
        manque = "" if Path(attendu).is_file() else f" ({Path(attendu).name} absent)"
        if any("OutOfMemoryError" in l_ or "out of memory" in l_ for l_ in fin):
            _erreur(f"{quoi} : mémoire de la carte graphique insuffisante. Ferme les autres programmes qui utilisent la "
                    "carte, ou simplifie le modèle (moins de faces).", 3)
        _erreur(f"{quoi} a échoué (code {proc.returncode}){manque}. Dernières lignes : "
                + " | ".join(l_ for l_ in fin if l_.strip())[-600:])
    print(f"{quoi} : {time.time() - debut:.0f} s.", flush=True)


def _extraire(fichier, travail, faces):
    return ["-m", "src.data.extract", "--config=configs/data/quick_inference.yaml", f"--require_suffix={SUFFIXES}",
            "--force_override=true", "--num_runs=1", "--id=0", f"--time={time.strftime('%Y_%m_%d_%H_%M_%S')}",
            f"--faces_target_count={int(faces)}", f"--input={fichier}", f"--output_dir={travail}"]


def animer(chemin_tache):
    t = json.loads(Path(chemin_tache).read_text(encoding="utf-8"))
    entree, sortie = Path(t["entree"]), Path(t["sortie"])
    if not entree.is_file():
        _erreur(f"modèle introuvable : {entree}")
    if not Path(TACHE_SQUELETTE).is_file():
        _erreur("code d'UniRig introuvable (le moteur doit être lancé depuis son dossier) : relance l'installateur.")
    # Dossier de travail sans virgule (UniRig découpe --input sur les virgules) ni point dans les noms : extract.py
    # range raw_data.npz dans <dossier du fichier>/<nom sans extension>/
    travail = sortie.parent / "rigging"
    shutil.rmtree(travail, ignore_errors=True)
    travail.mkdir(parents=True)
    modele = travail / f"modele{entree.suffix.lower()}"
    shutil.copy(entree, modele)
    squelette, peau = travail / "squelette.fbx", travail / "peau.fbx"
    graine, faces = int(t.get("graine") or 12345), int(t.get("faces") or 50000)

    print(f"PROGRESSION 1/{ETAPES} lecture du modèle", flush=True)
    _executer(_extraire(modele, travail, faces), travail / "modele" / "raw_data.npz", "Lecture du modèle")
    print(f"PROGRESSION 2/{ETAPES} squelette", flush=True)
    _executer(["run.py", f"--task={TACHE_SQUELETTE}", f"--seed={graine}", f"--input={modele}", f"--output={squelette}",
               f"--npz_dir={travail}"], squelette, "Squelette")
    print(f"PROGRESSION 3/{ETAPES} lecture du squelette", flush=True)
    _executer(_extraire(squelette, travail, faces), travail / "squelette" / "raw_data.npz", "Lecture du squelette")
    print(f"PROGRESSION 4/{ETAPES} poids de peau", flush=True)
    _executer(["run.py", f"--task={TACHE_PEAU}", f"--seed={graine}", f"--input={squelette}", f"--output={peau}",
               f"--npz_dir={travail}", "--data_name=raw_data.npz"], peau, "Poids de peau")
    print(f"PROGRESSION 5/{ETAPES} report sur le modèle d'origine", flush=True)
    _executer(["-m", "src.inference.merge", f"--require_suffix={SUFFIXES}", "--num_runs=1", "--id=0",
               f"--source={peau}", f"--target={modele}", f"--output={sortie}"], sortie, "Report sur le modèle")
    if t.get("fbx"):
        shutil.copy(peau, t["fbx"])
    _resultat({"sortie": str(sortie), "fbx": t.get("fbx"), "os": _nombre_d_os(sortie), "graine": graine})
    print(f"TERMINE {sortie}", flush=True)


def _nombre_d_os(glb):
    """Articulations du squelette dans le GLB (en-tête JSON du glTF binaire), 0 si illisible."""
    import struct

    try:
        donnees = Path(glb).read_bytes()
        n = struct.unpack("<I", donnees[12:16])[0]
        gltf = json.loads(donnees[20:20 + n])
        return max((len(s.get("joints", [])) for s in gltf.get("skins", [])), default=0)
    except (OSError, ValueError, struct.error):
        return 0


def _resultat(obj):
    print("RESULTAT " + json.dumps(obj, ensure_ascii=False), flush=True)


def telecharger():
    from huggingface_hub import snapshot_download

    print("PROGRESSION 1/2 téléchargement : poids d'UniRig (~5,8 Go)", flush=True)
    snapshot_download(UNIRIG, allow_patterns=POIDS)
    print("PROGRESSION 2/2 téléchargement : configuration d'OPT-350M", flush=True)
    snapshot_download(OPT, allow_patterns=["config.json"])
    print("Modèles prêts.", flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["telecharger"]:
        telecharger()
    elif len(sys.argv) == 3 and sys.argv[1] == "animer":
        animer(sys.argv[2])
    else:
        _erreur("usage : rigging.py animer <tache.json> | telecharger")
