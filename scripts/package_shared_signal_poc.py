"""Package the public shared-signal corpus as one safe Kaggle payload."""

from __future__ import annotations

import argparse
import base64
import io
import tarfile
from pathlib import Path, PurePosixPath


PUBLIC_ROOTS = {
    "green_sm_shared_signal_poc_linked_v1",
    "green_sm_shared_signal_poc_control_paired_v1",
}


def _safe_member_name(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe archive member: {name!r}")
    if path.parts[0] not in PUBLIC_ROOTS or "private_audit" in path.parts:
        raise ValueError(f"archive member is outside public PoC payload: {name!r}")
    return path


def package_shared_signal_poc(source_root: str | Path, destination: str | Path) -> Path:
    """Write an opaque base64 tarball containing only public paired corpus roots."""

    source = Path(source_root)
    destination_path = Path(destination)
    if destination_path.suffixes[-2:] != [".tar", ".gz"] and not destination_path.name.endswith(".tar.gz.b64"):
        raise ValueError("destination must end with .tar.gz.b64")
    if not source.is_dir():
        raise ValueError(f"source root does not exist: {source}")
    missing = [name for name in sorted(PUBLIC_ROOTS) if not (source / name).is_dir()]
    if missing:
        raise ValueError(f"source root lacks public corpus directories: {missing}")
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as archive:
        for name in sorted(PUBLIC_ROOTS):
            for path in sorted((source / name).rglob("*")):
                if not path.is_file():
                    continue
                relative = path.relative_to(source).as_posix()
                _safe_member_name(relative)
                archive.add(path, arcname=relative, recursive=False)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_bytes(base64.b64encode(payload.getvalue()))
    return destination_path


def extract_shared_signal_poc(payload_path: str | Path, destination: str | Path) -> Path:
    """Decode a payload without trusting archive paths or creating private artifacts."""

    payload = Path(payload_path)
    target = Path(destination)
    raw = base64.b64decode(payload.read_bytes(), validate=True)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        for member in archive.getmembers():
            path = _safe_member_name(member.name)
            if not member.isfile():
                raise ValueError(f"payload may only contain regular files: {member.name!r}")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"payload member could not be read: {member.name!r}")
            output = target.joinpath(*path.parts)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(stream.read())
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    command = parser.add_subparsers(dest="command", required=True)
    pack = command.add_parser("pack")
    pack.add_argument("source_root")
    pack.add_argument("destination")
    unpack = command.add_parser("extract")
    unpack.add_argument("payload")
    unpack.add_argument("destination")
    args = parser.parse_args()
    result = (
        package_shared_signal_poc(args.source_root, args.destination)
        if args.command == "pack"
        else extract_shared_signal_poc(args.payload, args.destination)
    )
    print(result)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
