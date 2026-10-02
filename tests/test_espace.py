"""Espace disque : mesure, retrait (fichiers supprimés + clé notée pour l'installateur), réinstallation, vidage."""
import os
import sys

import gradio as gr
import pytest

from studiovoix import chatterbox, config as cfg, diffusion, espace, retraits


def _fichier(p, octets):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"0" * octets)
    return p


def _modele_hf(depot, chemin, octets, lien=True):
    """Fichier d'un dépôt dans le cache Hugging Face : blob + lien dans snapshots (copie si les liens échouent)."""
    racine = diffusion._dossier_depot(depot)
    blob = _fichier(racine / "blobs" / f"{abs(hash(chemin))}", octets)
    cible = racine / "snapshots" / "abc" / chemin
    cible.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not lien:
            raise OSError
        os.symlink(blob, cible)
    except OSError:  # Windows sans droits de lien symbolique : fichier réel, comme huggingface_hub
        blob.unlink()
        _fichier(cible, octets)
    return cible


@pytest.fixture
def disque(env, monkeypatch):
    eng = env / "StudioVoix"
    monkeypatch.setattr(cfg, "ENG_DIR", eng)
    for nom in ("CHATTERBOX", "NETTOYAGE", "RVC", "DIFFUSION"):
        monkeypatch.setattr(cfg, f"{nom}_DIR", eng / nom.lower())
    monkeypatch.setenv("HF_HOME", str(eng / "hf-home"))
    monkeypatch.setattr(cfg, "DIFFUSION_PYTHON", sys.executable)  # environnement « installé »
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    _fichier(eng / "ace-step" / "checkpoints" / "dit.safetensors", 5000)
    _fichier(cfg.CHATTERBOX_DIR / ".venv" / "torch.dll", 3000)
    _fichier(chatterbox.ckpt_dir() / "snapshots" / "abc" / "t3.safetensors", 2000)
    _fichier(cfg.NETTOYAGE_DIR / "checkpoints" / "m.pt", 1000)
    _fichier(cfg.RVC_DIR / ".venv" / "torch.dll", 3000)
    _fichier(cfg.RVC_DIR / "rvc" / "models" / "pretraineds" / "hifi-gan" / "f0G40k.pth", 700)
    _fichier(cfg.RVC_DIR / "rvc" / "models" / "predictors" / "rmvpe.pt", 100)
    _fichier(cfg.RVC_DIR / "rvc" / "models" / "predictors" / "lisez-moi.txt", 10)  # fourni avec Applio : gardé
    _fichier(cfg.RVC_DIR / "logs" / "ma voix" / "ma voix.pth", 900)  # modèle entraîné : jamais supprimé
    _fichier(cfg.RVC_DIR / "core.py", 50)
    _fichier(cfg.DIFFUSION_DIR / ".venv" / "torch.dll", 3000)
    forme = _modele_hf("tencent/Hunyuan3D-2", "hunyuan3d-dit-v2-0-turbo/model.fp16.safetensors", 400)
    _modele_hf("tencent/Hunyuan3D-2", "hunyuan3d-vae-v2-0-turbo/model.fp16.safetensors", 50)
    _modele_hf("tencent/Hunyuan3D-2", "hunyuan3d-dit-v2-0/model.fp16.safetensors", 400)  # qualité « Maximale »
    peinture = _modele_hf("tencent/Hunyuan3D-2", "hunyuan3d-paint-v2-0-turbo/unet/diffusion_pytorch_model.safetensors", 1600)
    _modele_hf("Tongyi-MAI/Z-Image-Turbo", "vae/diffusion_pytorch_model.safetensors", 300)
    _modele_hf("unsloth/Z-Image-Turbo-GGUF", "z-image-turbo-Q8_0.gguf", 700, lien=False)
    _fichier(eng / "uv-cache" / "wheels" / "torch.whl", 4000)
    return forme, peinture


def test_inventaire(disque):
    md = espace.inventaire()
    assert "| ACE-Step (génération musicale) | 0 Mo | indispensable |" in md
    assert "Hunyuan3D-2 : texture (le plus gros) | 0 Mo | retirable |" in md and "Total :" in md
    assert espace.taille(cfg.CHATTERBOX_DIR) == 3000
    # forme et texture partagent un dépôt : chacune ne compte que ses sous-dossiers
    assert sum(espace.taille(c) for c in espace.chemins("diffusion:texture3d")) == 1600
    assert sum(espace.taille(c) for c in espace.chemins("diffusion:forme3d")) == 850  # turbo + complet + VAE
    assert sum(espace.taille(c) for c in espace.chemins("diffusion:zimage")) == 1000
    # FLUX.2 klein reprend l'encodeur de Z-Image : le retirer ne touche pas au dépôt de Z-Image
    assert [c.name for c in espace.chemins("diffusion:personnages")] == [
        "models--black-forest-labs--FLUX.2-klein-4B", "models--unsloth--FLUX.2-klein-4B-GGUF"]


def test_retirer_puis_reinstaller_un_modele(disque):
    forme, peinture = disque
    msg, md = espace.retirer("diffusion:texture3d")
    assert "libérés" in msg and "Hunyuan3D-2 : texture (le plus gros) | 0 Mo | retiré |" in md
    assert not peinture.exists() and forme.exists()  # seule la texture part
    if forme.is_symlink():
        assert forme.resolve().exists() and len(list(forme.parents[3].glob("blobs/*"))) == 3
    assert retraits.liste() == ["diffusion:texture3d"] and retraits.retire("diffusion:texture3d")
    assert not diffusion.present("texture3d") and diffusion.present("forme3d")
    with pytest.raises(gr.Error, match="retiré pour gagner de la place"):
        diffusion._verifier("forme3d", "texture3d")
    msg, _ = espace.reinstaller("diffusion:texture3d")
    assert "METTRE_A_JOUR.bat" in msg and retraits.liste() == []
    assert "n'est pas retiré" in espace.reinstaller("diffusion:texture3d")[0]


def test_retirer_rvc_garde_les_modeles_entraines(disque):
    espace.retirer("rvc")
    rvc = cfg.RVC_DIR
    assert not (rvc / ".venv").exists() and not (rvc / "rvc/models/pretraineds/hifi-gan").exists()
    assert not (rvc / "rvc/models/predictors/rmvpe.pt").exists()
    assert (rvc / "rvc/models/predictors/lisez-moi.txt").exists() and (rvc / "core.py").exists()
    assert (rvc / "logs" / "ma voix" / "ma voix.pth").read_bytes() == b"0" * 900


def test_retirer_chatterbox_et_conseil(disque, env):
    espace.retirer("chatterbox")
    assert not cfg.CHATTERBOX_DIR.exists() and not chatterbox.ckpt_dir().exists()
    assert "Réinstaller" in retraits.conseil("chatterbox") and "INSTALLER.bat" in retraits.conseil("nettoyage")
    assert "retiré" in next(chatterbox.download())


def test_tout_le_moteur_de_diffusion(disque):
    espace.retirer("diffusion")
    assert not cfg.DIFFUSION_DIR.exists() and not diffusion.present("zimage")
    assert retraits.retire("diffusion:qwen")  # chaque modèle l'est aussi
    with pytest.raises(gr.Error, match="réinstalle d'abord"):
        espace.reinstaller("diffusion:zimage")
    espace.reinstaller("diffusion")
    assert retraits.liste() == []


def test_vider_et_annuler(disque):
    msg, _ = espace.vider("uv-cache")
    assert "libérés" in msg and not (cfg.ENG_DIR / "uv-cache").exists()
    assert espace.retirer(None)[0] == "Retrait annulé." and espace.vider(None)[0] == "Annulé."
    with pytest.raises(gr.Error):
        espace.retirer("ace-step")  # jamais retirable
