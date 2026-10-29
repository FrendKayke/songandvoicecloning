"""Moteur « entraînement » — script exécuté DANS l'environnement d'entraînement (StudioVoix\\entrainement\\.venv,
Python 3.10, PyTorch 2.6 CUDA 12.4), depuis le dossier de kohya sd-scripts (StudioVoix\\entrainement\\sd-scripts, commit
épinglé dans installer.ps1).

Entraîne un LoRA SDXL sur des images légendées (feuilles de sprites préparées par l'application) avec
sdxl_train_network.py de kohya, dans les réglages « 8 à 10 Go » de sa documentation (docs/train_SDXL-en.md) :
U-Net seul (--network_train_unet_only), latents et sorties des encodeurs de texte en cache, gradient checkpointing,
AdamW 8 bits, fp16 (le dossier SDXL du cache Hugging Face ne contient que les poids « fp16 », que kohya ne charge
qu'avec --mixed_precision fp16 : library/sdxl_train_util.py, variant = "fp16" if weight_dtype == float16),
--no_half_vae (le VAE de SDXL donne des NaN en fp16).

    python lora_sdxl.py entrainer <tache.json>   {sdxl (dossier diffusers), jeu (dossier avec N_nom/), sortie, nom,
                                                  epoques, dim, graine, largeur, hauteur}
    python lora_sdxl.py telecharger              tokeniseurs CLIP de SDXL (openai/clip-vit-large-patch14,
                                                  laion/CLIP-ViT-bigG-14-laion2B-39B-b160k) dans tokenizers/
    python lora_sdxl.py verifier                 import de torch, CUDA, bitsandbytes

Sortie : « PROGRESSION i/n … », « RESULTAT <json> », « ERREUR : message », « TERMINE <fichier> ».
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

TOKENISEURS = ["openai/clip-vit-large-patch14", "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"]  # sdxl_train_util.py
PAS = re.compile(r"steps:\s+\d+%\|[^|]*\|\s*(\d+)/(\d+)")  # barre tqdm « steps » de train_network.py


def _erreur(msg, code=2):
    print(f"ERREUR : {msg}", flush=True)
    sys.exit(code)


def dossier_tokeniseurs():
    """Cache des tokeniseurs CLIP (--tokenizer_cache_dir de kohya : <id avec _>/), à côté de sd-scripts."""
    return Path(os.environ.get("STUDIOVOIX_TOKENISEURS") or Path.cwd().parent / "tokenizers")


def telecharger():
    from transformers import CLIPTokenizer

    d = dossier_tokeniseurs()
    d.mkdir(parents=True, exist_ok=True)
    for i, nom in enumerate(TOKENISEURS, 1):
        print(f"PROGRESSION {i}/{len(TOKENISEURS)} tokeniseur {nom}", flush=True)
        cible = d / nom.replace("/", "_")
        if (cible / "tokenizer_config.json").is_file():
            continue
        CLIPTokenizer.from_pretrained(nom).save_pretrained(cible)
    print("RESULTAT " + json.dumps({"tokeniseurs": str(d)}), flush=True)
    print(f"TERMINE {d}", flush=True)


def verifier():
    import torch

    if not torch.cuda.is_available():
        _erreur("CUDA indisponible dans l'environnement d'entraînement.")
    import bitsandbytes  # noqa: F401 - AdamW 8 bits

    print(f"Entraînement prêt : torch {torch.__version__}, GPU {torch.cuda.get_device_name(0)}", flush=True)
    print("TERMINE -", flush=True)


def arguments(t):
    """Ligne de commande de sdxl_train_network.py (options lues dans library/args.py, train_network.py et
    library/sdxl_train_util.py au commit épinglé)."""
    pas = max(1, int(t.get("epoques", 12)))
    return [
        "sdxl_train_network.py",
        f"--pretrained_model_name_or_path={t['sdxl']}",
        f"--train_data_dir={t['jeu']}",
        "--caption_extension=.txt",
        f"--resolution={int(t['largeur'])},{int(t['hauteur'])}",
        "--enable_bucket", "--bucket_no_upscale", "--bucket_reso_steps=32",
        f"--min_bucket_reso={min(int(t['largeur']), int(t['hauteur']))}",
        f"--max_bucket_reso={max(int(t['largeur']), int(t['hauteur']))}",
        f"--output_dir={t['sortie']}", f"--output_name={t['nom']}", "--save_model_as=safetensors",
        "--network_module=networks.lora", f"--network_dim={int(t.get('dim', 32))}",
        f"--network_alpha={max(1, int(t.get('dim', 32)) // 2)}", "--network_train_unet_only",
        "--learning_rate=1e-4", "--lr_scheduler=constant", "--optimizer_type=AdamW8bit",
        "--train_batch_size=1", f"--max_train_epochs={pas}",
        "--mixed_precision=fp16", "--save_precision=fp16", "--no_half_vae",
        "--cache_latents", "--cache_latents_to_disk", "--cache_text_encoder_outputs",
        "--cache_text_encoder_outputs_to_disk", "--gradient_checkpointing", "--sdpa",
        f"--seed={int(t.get('graine', 42))}", f"--tokenizer_cache_dir={dossier_tokeniseurs()}",
        "--max_data_loader_n_workers=0",
    ]


def entrainer(chemin_tache):
    t = json.loads(Path(chemin_tache).read_text(encoding="utf-8"))
    if not Path("sdxl_train_network.py").is_file():
        _erreur("code de kohya sd-scripts introuvable (le moteur doit être lancé depuis son dossier) : relance "
                "l'installateur.")
    if not Path(t["sdxl"], "model_index.json").is_file():
        _erreur(f"SDXL introuvable : {t['sdxl']}")
    if not any(Path(t["jeu"]).rglob("*.png")):
        _erreur(f"aucune image dans {t['jeu']}")
    sortie = Path(t["sortie"]) / f"{t['nom']}.safetensors"
    sortie.unlink(missing_ok=True)
    cmd = [sys.executable, "-m", "accelerate.commands.launch", "--num_processes=1", "--num_machines=1",
           "--mixed_precision=fp16", "--dynamo_backend=no", "--num_cpu_threads_per_process=1", *arguments(t)]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    print("PROGRESSION 1/2 chargement de SDXL et mise en cache des images", flush=True)
    debut = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                            errors="replace", env=env, bufsize=1)
    fin, dernier = [], 0
    for ligne in proc.stdout:
        for morceau in ligne.replace("\r", "\n").split("\n"):  # la barre tqdm se réécrit avec des retours chariot
            morceau = morceau.rstrip()
            if not morceau.strip():
                continue
            m = PAS.search(morceau)
            if m:
                i, n = int(m.group(1)), int(m.group(2))
                if i != dernier and n:
                    dernier = i
                    print(f"PROGRESSION {i}/{n} entraînement", flush=True)
                continue
            fin = (fin + [morceau])[-20:]
            print(f"  {morceau}", flush=True)
    proc.wait()
    if proc.returncode != 0 or not sortie.is_file():
        if any("OutOfMemoryError" in l_ or "out of memory" in l_ for l_ in fin):
            _erreur("mémoire de la carte graphique insuffisante : ferme les autres programmes qui utilisent la "
                    "carte, ou baisse la dimension du LoRA.", 3)
        _erreur(f"kohya a échoué (code {proc.returncode}){'' if sortie.is_file() else ' (LoRA absent)'}. "
                "Dernières lignes : " + " | ".join(l_.strip() for l_ in fin if l_.strip())[-800:])
    print("RESULTAT " + json.dumps({"lora": str(sortie), "duree_s": round(time.time() - debut)}), flush=True)
    print(f"TERMINE {sortie}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        _erreur("usage : lora_sdxl.py entrainer <tache.json> | telecharger | verifier")
    if sys.argv[1] == "entrainer":
        entrainer(sys.argv[2])
    elif sys.argv[1] == "telecharger":
        telecharger()
    elif sys.argv[1] == "verifier":
        verifier()
    else:
        _erreur(f"action inconnue : {sys.argv[1]}")
