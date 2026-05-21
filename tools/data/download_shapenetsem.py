"""Download and extract the gated ShapeNetSem archive from Hugging Face."""

import argparse
import ssl
import os
import urllib.request
import zipfile
from pathlib import Path


METADATA_URL = (
    "https://www.shapenet.org/solr/models3d/select?"
    "q=isAligned%3Atrue+AND+source%3Awss+AND+category%3A*&rows=100000&"
    "fl=fullId%2Ccategory%2Cwnsynset%2Cwnlemmas%2Cup%2Cfront%2Cunit%2Caligned.dims%2C"
    "isContainerLike%2CsurfaceVolume%2CsolidVolume%2CsupportSurfaceArea%2Cweight%2C"
    "staticFrictionForce%2Cname%2Ctags&wt=csv&indent=true"
)


def read_token(token_file: str | Path | None) -> str:
    if token_file:
        token = Path(token_file).read_text().strip()
    else:
        token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise ValueError("Hugging Face token not found. Pass --token_file or set HF_TOKEN.")
    return token


def download_current_metadata(raw_dir: Path) -> Path:
    metadata_path = raw_dir / "metadata.csv"
    archive_metadata_path = raw_dir / "metadata.archive.csv"
    if metadata_path.exists() and not archive_metadata_path.exists():
        metadata_path.replace(archive_metadata_path)

    print("Downloading current ShapeNetSem metadata from ShapeNet Solr")
    context = ssl._create_unverified_context()
    with urllib.request.urlopen(METADATA_URL, context=context, timeout=120) as src:
        content = src.read()
    metadata_path.write_bytes(content)
    return metadata_path


def download_archive(out_dir: Path, token: str) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required. Install it with pip install huggingface_hub.") from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    return Path(
        hf_hub_download(
            repo_id="ShapeNet/ShapeNetSem-archive",
            repo_type="dataset",
            filename="ShapeNetSem.zip",
            token=token,
            local_dir=str(out_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
    )


def extract_selected(archive_path: Path, raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as zf:
        names = zf.namelist()
        wanted = {
            "metadata.csv",
            "categories.synset.csv",
            "materials.csv",
            "densities.csv",
            "taxonomy.txt",
            "README.txt",
            "DATA.md",
            "models-OBJ.zip",
        }
        for name in names:
            base = Path(name).name
            if base in wanted:
                target = raw_dir / base
                if target.exists() and target.stat().st_size > 0:
                    continue
                print(f"Extracting {base}")
                with zf.open(name) as src, open(target, "wb") as dst:
                    dst.write(src.read())

        obj_dir = raw_dir / "models-OBJ"
        marker = obj_dir / ".extract_complete"
        if not marker.exists():
            obj_entries = [n for n in names if "/models-OBJ/" in n and not n.endswith("/")]
            if obj_entries:
                obj_dir.mkdir(parents=True, exist_ok=True)
                print(f"Extracting {len(obj_entries)} OBJ/MTL entries to {obj_dir}")
                for name in obj_entries:
                    rel = name.split("/models-OBJ/", 1)[1]
                    target = obj_dir / rel
                    if target.exists() and target.stat().st_size > 0:
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(name) as src, open(target, "wb") as dst:
                        dst.write(src.read())
                marker.write_text("ok\n")

    obj_zip = raw_dir / "models-OBJ.zip"
    obj_dir = raw_dir / "models-OBJ"
    marker = obj_dir / ".extract_complete"
    if obj_zip.exists() and not marker.exists():
        obj_dir.mkdir(parents=True, exist_ok=True)
        print(f"Extracting {obj_zip} to {obj_dir}")
        with zipfile.ZipFile(obj_zip) as zf:
            zf.extractall(obj_dir)
        marker.write_text("ok\n")

    download_current_metadata(raw_dir)


def main():
    parser = argparse.ArgumentParser(description="Download/extract ShapeNetSem.")
    parser.add_argument("--data_dir", default=os.path.expanduser("~/shapenetsem_regression/data/shapenetsem_regression"))
    parser.add_argument("--token_file", default=None)
    parser.add_argument("--download_only", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    raw_dir = data_dir / "raw"
    token = read_token(args.token_file)
    archive = download_archive(raw_dir, token)
    print(f"Archive ready: {archive}")
    if not args.download_only:
        extract_selected(archive, raw_dir)
        print(f"Raw data ready: {raw_dir}")


if __name__ == "__main__":
    main()
