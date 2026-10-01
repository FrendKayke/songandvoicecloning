"""Simulation de installer.ps1 sous Linux (PowerShell 7) avec de faux uv.exe / python.exe : ordre des étapes,
marqueurs à signature, reprise après échec, mise à jour d'une étape, adoption des anciens marqueurs.

Ne teste pas les vraies installations (réseau, carte graphique) : seulement la logique de l'installateur.
Ignoré si pwsh est absent (variable PWSH pour indiquer son chemin) ou sous Windows.
"""
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parent.parent
SIMU = Path(__file__).resolve().parent / "installateur"
PWSH = os.environ.get("PWSH") or shutil.which("pwsh")

pytestmark = pytest.mark.skipif(os.name == "nt" or not PWSH, reason="PowerShell 7 (pwsh) sous Linux requis")


@pytest.fixture
def inst(tmp_path):
    """Application copiée dans un dossier temporaire, moteurs dans tmp/StudioVoix, faux uv déjà présent."""
    app = tmp_path / "app"
    app.mkdir()
    shutil.copy(RACINE / "installer.ps1", app)
    shutil.copy(RACINE / "requirements.txt", app)
    shutil.copytree(RACINE / "installation", app / "installation")
    (app / "moteurs").mkdir()
    eng = tmp_path / "StudioVoix"
    (eng / "uv").mkdir(parents=True)
    shutil.copy(SIMU / "faux_outil.sh", eng / "uv" / "uv.exe")
    journal = tmp_path / "journal.log"

    class Inst:
        pass

    i = Inst()
    i.app, i.eng, i.journal = app, eng, journal

    def lancer(echec="", reponse=""):
        journal.write_text("")
        env = {**os.environ, "STUDIOVOIX_MOTEURS": str(eng), "JOURNAL": str(journal), "ECHEC": echec,
               "REPONSE": reponse}
        r = subprocess.run([PWSH, "-NoProfile", "-c",
                            f". '{SIMU / 'simulations.ps1'}'; & '{app / 'installer.ps1'}'; exit $LASTEXITCODE"],
                           cwd=app, env=env, capture_output=True, text=True, timeout=300)
        i.sortie = r.stdout + r.stderr
        i.appels = journal.read_text().splitlines()
        return r.returncode

    i.lancer = lancer
    i.installs = lambda: [a for a in i.appels if " pip install " in a]
    return i


def _marqueurs(eng):
    return {str(p.relative_to(eng)): p.read_text() for p in eng.rglob("*")
            if p.is_file() and (p.name.endswith("-ok") or p.name in ("installe.ok", ".complet"))}


def test_installation_complete_puis_relance(inst):
    assert inst.lancer() == 0, inst.sortie
    assert "Installation terminée" in inst.sortie and (inst.eng / "installation.ok").exists()
    etapes = [int(n) for n in re.findall(r"=== \[(\d+)/18\]", inst.sortie)]
    assert etapes == sorted(etapes) and etapes[0] == 1 and etapes[-1] == 18
    # chaque environnement lit sa liste de dépendances dans installation\
    listes = {Path(a.split(" -r ")[1].split(" ")[0].replace("\\", "/")).name for a in inst.installs() if " -r " in a}
    assert {"seed-vc.txt", "chatterbox.txt", "nettoyage.txt", "diffusion.txt", "requirements.txt"} <= listes
    m = _marqueurs(inst.eng)
    assert re.fullmatch(r"[0-9a-f]{64}", m["diffusion/.env-ok"]) and re.fullmatch(r"[0-9a-f]{64}", m["diffusion/.modeles-ok"])
    assert "Hunyuan3D-2/archive/" in m["diffusion/hunyuan3d/.complet"]
    telechargement = [a for a in inst.appels if "diffusion.py telecharger" in a][0]
    assert "telecharger qwen forme3d texture3d zimage detourage" in telechargement

    assert inst.lancer() == 0, inst.sortie
    assert inst.installs() == [] and "Mise à jour" not in inst.sortie
    assert inst.sortie.count("Déjà fait") >= 14


def test_anciens_marqueurs_vides_adoptes(inst):
    assert inst.lancer() == 0
    for p in list(inst.eng.rglob("*-ok")) + list(inst.eng.rglob(".complet")):
        p.write_text("")  # installation faite avant les signatures
    assert inst.lancer() == 0, inst.sortie
    assert inst.installs() == [] and not any(a.startswith("TELECHARGEMENT") for a in inst.appels)
    assert re.fullmatch(r"[0-9a-f]{64}", (inst.eng / "diffusion" / ".env-ok").read_text())


def test_liste_modifiee_seule_etape_refaite(inst):
    assert inst.lancer() == 0
    liste = inst.app / "installation" / "diffusion.txt"
    liste.write_text(liste.read_text() + "nouveau-paquet\n")
    assert inst.lancer() == 0, inst.sortie
    assert "Mise à jour : cette étape a changé" in inst.sortie
    installs = inst.installs()
    assert installs and all("diffusion" in a for a in installs)
    assert any(a.startswith("uv.exe venv") and "diffusion" in a for a in inst.appels)


def test_commit_epingle_change_code_mis_a_jour_sans_perte(inst):
    assert inst.lancer() == 0
    rvc = inst.eng / "rvc"
    (rvc / "logs" / "ma_voix").mkdir(parents=True)
    (rvc / "logs" / "ma_voix" / "ma_voix.pth").write_text("modèle entraîné")
    script = inst.app / "installer.ps1"
    texte = script.read_bytes().decode("utf-8-sig")
    ancien = re.search(r"\$RvcCommit = '([0-9a-f]+)'", texte).group(1)
    script.write_bytes(b"\xef\xbb\xbf" + texte.replace(ancien, "0" * 40).encode("utf-8"))
    assert inst.lancer() == 0, inst.sortie
    assert any("Applio/archive/" + "0" * 40 in a for a in inst.appels)
    assert (rvc / "logs" / "ma_voix" / "ma_voix.pth").read_text() == "modèle entraîné"  # rien de supprimé
    assert "0" * 40 in (rvc / "version.txt").read_text()  # nouveau code copié par-dessus
    assert any(" -r " in a and "rvc" in a for a in inst.installs())  # environnement RVC refait (signature)


def test_echec_puis_reprise(inst):
    assert inst.lancer(echec="diffusion.txt") == 1
    assert "ERREUR" in inst.sortie and "relance INSTALLER.bat" in inst.sortie
    assert not (inst.eng / "diffusion" / ".env-ok").exists() and (inst.eng / "nettoyage" / ".env-ok").exists()
    assert inst.lancer() == 0, inst.sortie
    assert inst.installs() and all("diffusion" in a or "requirements.txt" in a for a in inst.installs())


def test_jeton_hugging_face(inst):
    assert inst.lancer(reponse="  hf_abcdefghijklmnopqrstuvwxyz  ") == 0, inst.sortie
    jeton = inst.eng / "hf-home" / "token"
    assert jeton.read_bytes() == b"hf_abcdefghijklmnopqrstuvwxyz"  # sans BOM ni fin de ligne
    assert any("telecharger bruitages" in a for a in inst.appels)
