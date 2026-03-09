from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import zipfile


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_packaged_layout(package_root: Path) -> None:
    assert (package_root / "main.py").exists()
    assert (package_root / "ai.py").exists()
    assert (package_root / "protocol.py").exists()
    assert (package_root / "common.py").exists()
    assert (package_root / "SDK" / "__init__.py").exists()
    assert (package_root / "tools" / "setup_native.py").exists()


def _run_packaging_script(script_name: str, output_path: Path | None = None) -> Path:
    command = ["bash", f"AI/{script_name}"]
    if output_path is not None:
        command.append(str(output_path))
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(completed.stdout.strip())


def _assert_packaged_main_imports_without_optional_env_dependency(package_root: Path) -> None:
    script = """
import builtins
import sys

package_root = sys.argv[1]
real_import = builtins.__import__

def blocked(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "gymnasium" or name.startswith("gymnasium.") or name == "pettingzoo" or name.startswith("pettingzoo."):
        raise ModuleNotFoundError(f"No module named '{name.split('.')[0]}'")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = blocked
sys.path.insert(0, package_root)
import main
"""
    subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(package_root),
        ],
        check=True,
    )


def test_zip_rand_creates_runnable_layout(tmp_path: Path) -> None:
    package_root = tmp_path / "random-package"
    returned_path = _run_packaging_script("zip_rand.sh", package_root)
    assert returned_path == package_root
    _assert_packaged_layout(package_root)
    sys.path.insert(0, str(package_root))
    try:
        module = _load_module("packaged_random_ai", package_root / "ai.py")
        assert hasattr(module, "AI")
    finally:
        sys.path.remove(str(package_root))


def test_zip_mcts_and_zip_expert_include_expected_support_files(tmp_path: Path) -> None:
    mcts_root = tmp_path / "mcts-package"
    expert_root = tmp_path / "expert-package"
    assert _run_packaging_script("zip_mcts.sh", mcts_root) == mcts_root
    assert _run_packaging_script("zip_expert.sh", expert_root) == expert_root
    _assert_packaged_layout(mcts_root)
    _assert_packaged_layout(expert_root)
    assert not (mcts_root / "ai_greedy.py").exists()
    assert not (mcts_root / "AI" / "AI_expert").exists()
    assert (expert_root / "runtime.py").exists()
    assert (expert_root / "antwar" / "core.py").exists()
    assert not (expert_root / "AI" / "AI_expert").exists()


def test_gitignore_covers_transient_directories() -> None:
    content = Path(".gitignore").read_text()
    for pattern in ("build/", "__pycache__/", ".pytest_cache/"):
        assert pattern in content


def test_main_entrypoint_uses_supplied_ai_class(monkeypatch) -> None:
    import AI.main as entry

    observed = SimpleNamespace(agent=None, session=None)

    class DummyAI:
        pass

    class DummySession:
        pass

    def fake_build_session(agent):
        observed.agent = agent
        return DummySession()

    def fake_run_session(session) -> None:
        observed.session = session

    monkeypatch.setattr(entry, "build_session", fake_build_session)
    monkeypatch.setattr(entry, "run_session", fake_run_session)
    entry.main(DummyAI)
    assert isinstance(observed.agent, DummyAI)
    assert isinstance(observed.session, DummySession)


def test_zip_script_runs_without_explicit_output_dir() -> None:
    archive_path = _run_packaging_script("zip_rand.sh")
    package_root = archive_path.with_suffix("")
    try:
        assert archive_path.exists()
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(package_root)
        _assert_packaged_layout(package_root)
    finally:
        shutil.rmtree(package_root, ignore_errors=True)
        archive_path.unlink(missing_ok=True)


def test_zip_script_accepts_explicit_zip_output_path(tmp_path: Path) -> None:
    archive_path = tmp_path / "custom-random.zip"
    returned_path = _run_packaging_script("zip_rand.sh", archive_path)
    package_root = tmp_path / "custom-random"
    try:
        assert returned_path == archive_path
        assert archive_path.exists()
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(package_root)
        _assert_packaged_layout(package_root)
    finally:
        shutil.rmtree(package_root, ignore_errors=True)


def test_packaged_ais_do_not_require_gymnasium_for_main_import(tmp_path: Path) -> None:
    for script_name in ("zip_rand.sh", "zip_mcts.sh", "zip_expert.sh"):
        package_root = tmp_path / script_name.replace(".sh", "")
        returned_path = _run_packaging_script(script_name, package_root)
        assert returned_path == package_root
        _assert_packaged_main_imports_without_optional_env_dependency(package_root)
