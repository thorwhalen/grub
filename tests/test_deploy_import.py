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


def test_a_failed_build_uploads_nothing(monkeypatch):
    """A build that fails must not go on to upload whatever stale files sit in dist/."""
    calls = []

    def run(cmd, *a, check=False, **kw):
        calls.append(cmd)
        if check:
            raise subprocess.CalledProcessError(1, cmd)
        return subprocess.CompletedProcess(cmd, 1)

    spec = importlib.util.spec_from_file_location("_deploy_under_test", DEPLOY_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.subprocess, "run", run)

    try:
        module.main()
    except subprocess.CalledProcessError:
        pass

    assert calls == [module.BUILD_COMMAND]
