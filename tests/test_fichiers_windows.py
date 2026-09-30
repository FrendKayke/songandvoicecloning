"""Encodages exigés par Windows (voir CLAUDE.md) : .ps1 en UTF-8 avec BOM + CRLF, .bat en ASCII + CRLF."""
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parent.parent


def _crlf_partout(data: bytes) -> bool:
    return data.count(b"\n") == data.count(b"\r\n") > 0


def test_installer_ps1_bom_crlf():
    data = (RACINE / "installer.ps1").read_bytes()
    assert data.startswith(b"\xef\xbb\xbf"), "installer.ps1 doit être en UTF-8 avec BOM (PowerShell 5.1)"
    assert _crlf_partout(data), "installer.ps1 doit avoir des fins de ligne CRLF"
    data[3:].decode("utf-8")


@pytest.mark.parametrize("nom", ["INSTALLER.bat", "lancer.bat"])
def test_bat_ascii_crlf(nom):
    data = (RACINE / nom).read_bytes()
    data.decode("ascii")  # pas d'accents dans les .bat
    assert _crlf_partout(data)


def test_lancer_et_installer_coherents():
    """Les mêmes dossiers doivent être utilisés à l'installation et au lancement."""
    bat = (RACINE / "lancer.bat").read_text(encoding="ascii")
    ps1 = (RACINE / "installer.ps1").read_text(encoding="utf-8-sig")
    for var, bat_val, ps_val in [
        ("UV_CACHE_DIR", r"%ENG%\uv-cache", "Join-Path $Eng 'uv-cache'"),
        ("UV_PYTHON_INSTALL_DIR", r"%ENG%\python", "Join-Path $Eng 'python'"),
        ("TORCH_HOME", r"%ENG%\torch-cache", "Join-Path $Eng 'torch-cache'"),
        ("HF_HOME", r"%ENG%\hf-home", "Join-Path $Eng 'hf-home'"),
        ("PKUSEG_HOME", r"%ENG%\chatterbox\pkuseg", "Join-Path $Cb 'pkuseg'"),
    ]:
        assert f'set "{var}={bat_val}"' in bat, var
        assert f"$env:{var} = {ps_val}" in ps1, var
