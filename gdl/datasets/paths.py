"""Path resolution helpers for dataset manifests."""

from pathlib import Path


def resolve_manifest_path(path_text: str, manifest_dir: str | Path) -> Path:
    """Resolve a path stored in a split manifest.

    Historical manifests contain a mix of absolute paths, repository-relative
    paths such as ``data/shapenetsem_regression/processed/...``, and processed
    corpus paths moved outside tracked artifact metadata. This resolver keeps
    those manifests usable after the repository layout cleanup.
    """

    path = Path(path_text)
    manifest_dir = Path(manifest_dir)
    candidates = [path]

    roots = [manifest_dir, manifest_dir.parent, manifest_dir.parent.parent]
    candidates.extend(root / path for root in roots)

    parts = path.parts
    if "processed" in parts:
        processed_index = parts.index("processed")
        suffix = Path(*parts[processed_index + 1 :])
        candidates.extend(root / "processed" / suffix for root in roots)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return path
