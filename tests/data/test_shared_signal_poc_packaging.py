from __future__ import annotations

import base64
import ast
import io
import json
import tarfile
from pathlib import Path
from pathlib import PurePosixPath

import pytest

from scripts.package_shared_signal_poc import extract_shared_signal_poc, package_shared_signal_poc


def _public_root(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    for name in (
        "green_sm_shared_signal_poc_linked_v1",
        "green_sm_shared_signal_poc_control_paired_v1",
    ):
        path = root / name / "foundation"
        path.mkdir(parents=True)
        (path / "manifest.json").write_text(name, encoding="utf-8")
    private = root / "private_audit"
    private.mkdir()
    (private / "affinity.json").write_text("never package", encoding="utf-8")
    return root


def test_package_round_trip_excludes_private_audit(tmp_path: Path) -> None:
    payload = package_shared_signal_poc(_public_root(tmp_path), tmp_path / "poc.tar.gz.b64")
    extracted = extract_shared_signal_poc(payload, tmp_path / "extracted")

    assert payload.name.endswith(".tar.gz.b64")
    assert (extracted / "green_sm_shared_signal_poc_linked_v1" / "foundation" / "manifest.json").is_file()
    assert not (extracted / "private_audit").exists()


def test_extraction_rejects_path_traversal(tmp_path: Path) -> None:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        info = tarfile.TarInfo("../escape.txt")
        info.size = 1
        archive.addfile(info, io.BytesIO(b"x"))
    payload = tmp_path / "unsafe.tar.gz.b64"
    payload.write_bytes(base64.b64encode(stream.getvalue()))

    with pytest.raises(ValueError, match="unsafe archive member"):
        extract_shared_signal_poc(payload, tmp_path / "out")


@pytest.mark.parametrize("task", ("food", "ads"))
def test_downstream_notebook_extracts_checkpoint_with_directory_entries(tmp_path: Path, task: str) -> None:
    source = tmp_path / "checkpoint"
    source.mkdir()
    (source / "checkpoint_manifest.json").write_text("{}", encoding="utf-8")
    archive_path = tmp_path / "checkpoint.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(source, arcname=".")
    notebook = json.loads(
        (Path(__file__).parents[2] / "notebooks" / f"shared_signal_poc_{task}_kaggle.ipynb").read_text(encoding="utf-8")
    )
    source_code = "".join(notebook["cells"][2]["source"])
    function = next(node for node in ast.parse(source_code).body if isinstance(node, ast.FunctionDef) and node.name == "extract_checkpoint")
    namespace = {"tarfile": tarfile, "PurePosixPath": PurePosixPath}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<notebook>", "exec"), namespace)

    destination = tmp_path / "extracted"
    namespace["extract_checkpoint"](archive_path, destination)
    assert (destination / "checkpoint_manifest.json").read_text(encoding="utf-8") == "{}"


@pytest.mark.parametrize(
    "notebook_name",
    (
        "shared_signal_poc_pretrain_kaggle.ipynb",
        "shared_signal_poc_food_kaggle.ipynb",
        "shared_signal_poc_ads_kaggle.ipynb",
    ),
)
def test_shared_signal_notebooks_use_safe_payloads_and_streamed_logs(notebook_name: str) -> None:
    notebook = Path(__file__).parents[2] / "notebooks" / notebook_name
    content = notebook.read_text(encoding="utf-8")

    assert "extract_shared_signal_poc" in content
    assert "python -u" in content
    assert "checkpoint_manifest.json" in content
    if notebook_name != "shared_signal_poc_pretrain_kaggle.ipynb":
        assert "best.pt" not in content
