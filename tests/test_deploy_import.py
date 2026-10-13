"""Importing the repo-root ``deploy.py`` must not build or upload anything."""

import importlib.util
import subprocess
from pathlib import Path

DEPLOY_PATH = Path(__file__).resolve().parent.parent / "deploy.py"


def test_importing_deploy_has_no_side_effects(monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: calls.append(a))
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: calls.append(a))

    spec = importlib.util.spec_from_file_location("_deploy_under_test", DEPLOY_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert calls == []
    assert callable(module.main)
