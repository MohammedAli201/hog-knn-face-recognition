"""Restore the original ZIP and project assets using only Python's standard library."""
from pathlib import Path
import hashlib
import json
import zipfile
import io
import shutil

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for data in iter(lambda: f.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def restore_archive(config):
    manifest = json.loads((ROOT / "archive/parts/manifest.json").read_text())
    target = ROOT / "archive/original-submission.zip"
    if target.exists():
        if digest(target) != manifest["sha256"]:
            raise RuntimeError("Existing original-submission.zip differs; move it before restoring.")
        return target
    for item in manifest["parts"]:
        part = ROOT / item["path"]
        if part.stat().st_size != item["size"] or digest(part) != item["sha256"]:
            raise RuntimeError(f"Missing or damaged part: {part}")
    temporary = target.with_suffix(".restoring")
    try:
        with temporary.open("wb") as output:
            for item in manifest["parts"]:
                with (ROOT / item["path"]).open("rb") as source:
                    shutil.copyfileobj(source, output)
        if digest(temporary) != manifest["sha256"]:
            raise RuntimeError("Original archive checksum failed.")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def extract_member(archive, source, destination):
    output = ROOT / destination
    output.parent.mkdir(parents=True, exist_ok=True)
    # Assets are restored only into the documented project directories.
    if output.exists():
        with archive.open(source) as f:
            if output.read_bytes() != f.read():
                raise RuntimeError(f"Existing asset differs: {output}")
        return
    with archive.open(source) as f, output.open("wb") as out:
        shutil.copyfileobj(f, out)


def main():
    config = json.loads((ROOT / "archive/restore-config.json").read_text())
    original = restore_archive(config)
    with zipfile.ZipFile(original) as archive:
        if config["kind"] == "ztype":
            with zipfile.ZipFile(io.BytesIO(archive.read(config["source_zip"]))) as source:
                for item in source.infolist():
                    rel = Path(item.filename)
                    if item.is_dir() or ".." in rel.parts or rel.is_absolute():
                        continue
                    if item.filename.startswith("ZType/Resource/") or item.filename.startswith("ZType/source/image/"):
                        extract_member(source, item.filename, "src/" + item.filename)
            for item in archive.infolist():
                if not item.is_dir() and item.filename.startswith(config["appendices"]):
                    extract_member(archive, item.filename, "docs/appendices/" + Path(item.filename).name)
        else:
            for item in archive.infolist():
                if item.is_dir() or not item.filename.startswith(config["data_prefix"]):
                    continue
                relative = item.filename[len(config["data_prefix"]):]
                if ".." in Path(relative).parts or Path(relative).is_absolute() or Path(relative).suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                extract_member(archive, item.filename, "data/Final Training Images/" + relative)
        extract_member(archive, config["video"], "demos/original-presentation.mp4")
    print("Original archive checksum verified; assets and presentation video restored.")
    print("The ZIP retains all submitted files, including historical generated files.")


if __name__ == "__main__":
    main()
