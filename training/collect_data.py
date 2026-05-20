#!/usr/bin/env python
"""
Phase 1 -- Automated Dataset Collection.

Robust automated downloader that collects chess-piece detection datasets from
multiple public sources without any Roboflow dependency.

Download strategies (tried in order per source):
  1. ``requests`` with retry + exponential backoff
  2. ``curl`` subprocess fallback
  3. ``git clone`` fallback (for repository sources)

Features:
  - Retry logic with exponential backoff
  - SSL certificate fallback for Windows (certifi -> unverified)
  - Automatic ZIP / tar.gz extraction
  - Post-download dataset validation (image-label pairs, YOLO format, classes)
  - Broken source skipping with detailed logging
  - Stores everything under ``training/raw_data/<source_name>/``

Usage:
    python collect_data.py                      # try all sources
    python collect_data.py --sources github hf  # specific sources only
    python collect_data.py --list                # show available sources
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import platform
import shutil
import ssl
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

from config import CLASS_NAMES, CLASS_TO_ID, NUM_CLASSES, RAW_DATA_DIR

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("collect_data")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_RETRIES = 3
RETRY_BACKOFF = 2          # seconds, doubled each attempt
DOWNLOAD_TIMEOUT = 300     # 5 minutes per download
CHUNK_SIZE = 8192
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IS_WINDOWS = platform.system() == "Windows"

# ---------------------------------------------------------------------------
# Dataset source registry
# ---------------------------------------------------------------------------
# Each source dict has:
#   name     : unique identifier
#   label    : human-readable description
#   type     : "zip" | "git" | "hf_repo"
#   urls     : list of fallback URLs (tried in order)
#   subpath  : optional relative path within extracted archive to the data
#   branch   : optional branch for git clone
# ---------------------------------------------------------------------------
SOURCES: list[dict[str, Any]] = [
    {
        "name": "roboflow_chess_pieces",
        "label": "Roboflow Chess Pieces (GitHub mirror ZIP)",
        "type": "zip",
        "urls": [
            "https://github.com/ultralytics/assets/releases/download/v0.0.0/chess-pieces-mjzgj.zip",
            "https://universe.roboflow.com/ds/chess-pieces-mjzgj/2?key=public",
        ],
    },
    {
        "name": "github_chess_erdogant",
        "label": "erdogant/chess GitHub dataset",
        "type": "zip",
        "urls": [
            "https://github.com/erdogant/chess/archive/refs/heads/main.zip",
        ],
    },
    {
        "name": "github_chess_yolo_dataset",
        "label": "Chess YOLO Dataset (GitHub ZIP)",
        "type": "zip",
        "urls": [
            "https://github.com/Elucidation/ChessboardFenTensorflowJs/archive/refs/heads/master.zip",
        ],
    },
    {
        "name": "hf_chess_pieces_francesco",
        "label": "HuggingFace Francesco/chess-pieces",
        "type": "hf_repo",
        "urls": [
            "https://huggingface.co/datasets/Francesco/chess-pieces/resolve/main/data/train-00000-of-00001.parquet",
        ],
        "hf_repo_id": "Francesco/chess-pieces",
    },
    {
        "name": "github_chess_vision",
        "label": "Chess Vision Dataset (GitHub)",
        "type": "git",
        "urls": [
            "https://github.com/Rydeen7/chess_piece_detection.git",
        ],
        "branch": "main",
    },
]


# ---------------------------------------------------------------------------
# SSL / TLS helpers (Windows certificate issues)
# ---------------------------------------------------------------------------
def _get_ssl_context() -> ssl.SSLContext | None:
    """Build an SSL context, falling back gracefully on Windows."""
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
        log.debug("Using certifi CA bundle for SSL")
        return ctx
    except ImportError:
        pass

    try:
        ctx = ssl.create_default_context()
        return ctx
    except ssl.SSLError:
        pass

    if IS_WINDOWS:
        log.warning("SSL verification failed -- creating unverified context (Windows fallback)")
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx

    return None


def _requests_session():
    """Build a requests.Session with retry adapter and SSL fallback."""
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    session = requests.Session()
    retry = Retry(
        total=MAX_RETRIES,
        backoff_factor=RETRY_BACKOFF,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    if IS_WINDOWS:
        try:
            import certifi
            session.verify = certifi.where()
        except ImportError:
            session.verify = True

    return session


# ---------------------------------------------------------------------------
# Download strategies
# ---------------------------------------------------------------------------
def _download_with_requests(url: str, dest_file: Path) -> bool:
    """Download *url* to *dest_file* using requests. Returns True on success."""
    try:
        import requests
    except ImportError:
        log.warning("requests not installed -- skipping requests download")
        return False

    session = _requests_session()
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            log.info("  [requests] Attempt %d/%d: %s", attempt, MAX_RETRIES, url)
            resp = session.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT, allow_redirects=True)
            resp.raise_for_status()

            dest_file.parent.mkdir(parents=True, exist_ok=True)
            total = int(resp.headers.get("content-length", 0))
            downloaded = 0

            with open(dest_file, "wb") as f:
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

            if total and downloaded < total * 0.9:
                log.warning("  Incomplete download (%d/%d bytes)", downloaded, total)
                dest_file.unlink(missing_ok=True)
                continue

            size_mb = downloaded / (1024 * 1024)
            log.info("  Downloaded %.1f MB -> %s", size_mb, dest_file.name)
            return True

        except Exception as exc:
            log.warning("  [requests] Attempt %d failed: %s", attempt, exc)
            dest_file.unlink(missing_ok=True)
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF * (2 ** (attempt - 1))
                log.info("  Retrying in %ds...", wait)
                time.sleep(wait)

    # Try without SSL verification on Windows as last resort
    if IS_WINDOWS:
        try:
            log.info("  [requests] Trying with SSL verification disabled (Windows fallback)...")
            import requests as req
            resp = req.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT, verify=False, allow_redirects=True)
            resp.raise_for_status()
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            with open(dest_file, "wb") as f:
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
            log.info("  Downloaded (SSL-unverified) -> %s", dest_file.name)
            return True
        except Exception as exc:
            log.warning("  [requests] SSL-fallback also failed: %s", exc)

    return False


def _download_with_curl(url: str, dest_file: Path) -> bool:
    """Download *url* to *dest_file* using curl subprocess."""
    if shutil.which("curl") is None:
        log.debug("  curl not found on PATH")
        return False

    dest_file.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, MAX_RETRIES + 1):
        log.info("  [curl] Attempt %d/%d: %s", attempt, MAX_RETRIES, url)
        cmd = [
            "curl", "-fSL",
            "--retry", "3",
            "--retry-delay", "2",
            "--connect-timeout", "30",
            "--max-time", str(DOWNLOAD_TIMEOUT),
            "-o", str(dest_file),
            url,
        ]
        if IS_WINDOWS:
            cmd.insert(1, "-k")  # allow insecure on Windows as needed

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=DOWNLOAD_TIMEOUT + 30,
            )
            if result.returncode == 0 and dest_file.exists() and dest_file.stat().st_size > 0:
                size_mb = dest_file.stat().st_size / (1024 * 1024)
                log.info("  [curl] Downloaded %.1f MB -> %s", size_mb, dest_file.name)
                return True
            else:
                log.warning("  [curl] Failed (exit %d): %s", result.returncode, result.stderr.strip()[:200])
                dest_file.unlink(missing_ok=True)
        except subprocess.TimeoutExpired:
            log.warning("  [curl] Timed out")
            dest_file.unlink(missing_ok=True)
        except Exception as exc:
            log.warning("  [curl] Error: %s", exc)
            dest_file.unlink(missing_ok=True)

        if attempt < MAX_RETRIES:
            wait = RETRY_BACKOFF * (2 ** (attempt - 1))
            time.sleep(wait)

    return False


def _git_clone(url: str, dest_dir: Path, branch: str | None = None) -> bool:
    """Clone a git repository to *dest_dir*."""
    if shutil.which("git") is None:
        log.debug("  git not found on PATH")
        return False

    if dest_dir.exists():
        shutil.rmtree(dest_dir, ignore_errors=True)

    dest_dir.parent.mkdir(parents=True, exist_ok=True)

    cmd = ["git", "clone", "--depth", "1"]
    if branch:
        cmd.extend(["-b", branch])
    cmd.extend([url, str(dest_dir)])

    for attempt in range(1, MAX_RETRIES + 1):
        log.info("  [git] Attempt %d/%d: cloning %s", attempt, MAX_RETRIES, url)
        try:
            env = os.environ.copy()
            if IS_WINDOWS:
                env["GIT_SSL_NO_VERIFY"] = "true"

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=DOWNLOAD_TIMEOUT,
                env=env,
            )
            if result.returncode == 0 and dest_dir.exists():
                git_dir = dest_dir / ".git"
                if git_dir.exists():
                    shutil.rmtree(git_dir, ignore_errors=True)
                log.info("  [git] Cloned successfully -> %s", dest_dir.name)
                return True
            else:
                log.warning("  [git] Failed (exit %d): %s", result.returncode, result.stderr.strip()[:200])
                shutil.rmtree(dest_dir, ignore_errors=True)
        except subprocess.TimeoutExpired:
            log.warning("  [git] Timed out")
            shutil.rmtree(dest_dir, ignore_errors=True)
        except Exception as exc:
            log.warning("  [git] Error: %s", exc)
            shutil.rmtree(dest_dir, ignore_errors=True)

        if attempt < MAX_RETRIES:
            wait = RETRY_BACKOFF * (2 ** (attempt - 1))
            time.sleep(wait)

    return False


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
def _extract_archive(archive_path: Path, dest_dir: Path) -> bool:
    """Extract ZIP or tar.gz archive to *dest_dir*."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = archive_path.name.lower()

    try:
        if name.endswith(".zip"):
            log.info("  Extracting ZIP archive...")
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(dest_dir)
            log.info("  Extracted %d entries to %s", len(zipfile.ZipFile(archive_path).namelist()), dest_dir)
            return True

        elif name.endswith((".tar.gz", ".tgz")):
            log.info("  Extracting tar.gz archive...")
            with tarfile.open(archive_path, "r:gz") as tf:
                tf.extractall(dest_dir, filter="data")
            log.info("  Extracted to %s", dest_dir)
            return True

        elif name.endswith(".tar"):
            log.info("  Extracting tar archive...")
            with tarfile.open(archive_path, "r:") as tf:
                tf.extractall(dest_dir, filter="data")
            log.info("  Extracted to %s", dest_dir)
            return True

        else:
            log.warning("  Unknown archive format: %s", archive_path.name)
            return False

    except Exception as exc:
        log.error("  Extraction failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Post-download validation
# ---------------------------------------------------------------------------
def _count_images(directory: Path) -> int:
    """Recursively count image files."""
    count = 0
    for ext in IMAGE_EXTENSIONS:
        count += len(list(directory.rglob(f"*{ext}")))
    return count


def _count_labels(directory: Path) -> int:
    """Recursively count .txt label files (excluding known non-label files)."""
    skip = {"classes.txt", "obj.names", "_classes.txt", "_classes_from_yaml.txt",
            "README.txt", "readme.txt", "notes.txt"}
    count = 0
    for txt in directory.rglob("*.txt"):
        if txt.name.lower() not in skip:
            count += 1
    return count


def _find_classes_in_labels(directory: Path, max_scan: int = 500) -> set[int]:
    """Scan up to *max_scan* label files and return all class IDs found."""
    skip = {"classes.txt", "obj.names", "_classes.txt", "_classes_from_yaml.txt",
            "README.txt", "readme.txt", "notes.txt"}
    class_ids: set[int] = set()
    scanned = 0
    for txt in directory.rglob("*.txt"):
        if txt.name.lower() in skip:
            continue
        try:
            for line in txt.read_text().strip().splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_ids.add(int(parts[0]))
        except (ValueError, UnicodeDecodeError):
            continue
        scanned += 1
        if scanned >= max_scan:
            break
    return class_ids


def validate_source(source_dir: Path, source_name: str) -> dict[str, Any]:
    """Validate a downloaded source directory. Returns a report dict."""
    report: dict[str, Any] = {
        "name": source_name,
        "valid": False,
        "images": 0,
        "labels": 0,
        "class_ids": set(),
        "issues": [],
    }

    if not source_dir.exists():
        report["issues"].append("Directory does not exist")
        return report

    report["images"] = _count_images(source_dir)
    report["labels"] = _count_labels(source_dir)

    if report["images"] == 0:
        report["issues"].append("No image files found")
    if report["labels"] == 0:
        report["issues"].append("No label files found (images-only dataset -- labels may be generated during preparation)")

    report["class_ids"] = _find_classes_in_labels(source_dir)

    if report["images"] > 0:
        report["valid"] = True
    if report["labels"] > 0 and report["images"] > 0:
        pair_ratio = min(report["images"], report["labels"]) / max(report["images"], report["labels"])
        if pair_ratio < 0.5:
            report["issues"].append(
                f"Image-label count mismatch (ratio={pair_ratio:.2f}): "
                f"{report['images']} images, {report['labels']} labels"
            )

    return report


def print_validation_report(reports: list[dict[str, Any]]) -> None:
    """Print a summary validation report for all downloaded sources."""
    log.info("")
    log.info("=" * 65)
    log.info("  DATASET VALIDATION REPORT")
    log.info("=" * 65)

    total_images = 0
    total_labels = 0
    all_class_ids: set[int] = set()
    valid_sources = 0

    for r in reports:
        status = "OK" if r["valid"] else "FAILED"
        log.info("  [%s] %s", status, r["name"])
        log.info("         Images: %d | Labels: %d | Classes: %s",
                 r["images"], r["labels"], sorted(r["class_ids"]) if r["class_ids"] else "N/A")
        for issue in r["issues"]:
            log.info("         ! %s", issue)

        if r["valid"]:
            valid_sources += 1
            total_images += r["images"]
            total_labels += r["labels"]
            all_class_ids |= r["class_ids"]

    log.info("-" * 65)
    log.info("  Valid sources: %d / %d", valid_sources, len(reports))
    log.info("  Total images:  %d", total_images)
    log.info("  Total labels:  %d", total_labels)

    if all_class_ids:
        log.info("  Class IDs found: %s", sorted(all_class_ids))
        if len(all_class_ids) >= NUM_CLASSES:
            log.info("  All %d chess classes represented", NUM_CLASSES)
        else:
            missing = set(range(NUM_CLASSES)) - all_class_ids
            log.info("  Missing class IDs: %s (may be remapped during preparation)", sorted(missing))
    log.info("=" * 65)


# ---------------------------------------------------------------------------
# Source-type handlers
# ---------------------------------------------------------------------------
def _collect_zip_source(source: dict, dest_dir: Path) -> bool:
    """Download and extract a ZIP source."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_archive = Path(tmpdir) / f"{source['name']}.zip"

        for url in source["urls"]:
            log.info("Trying URL: %s", url)

            if _download_with_requests(url, tmp_archive):
                break
            if _download_with_curl(url, tmp_archive):
                break

            log.info("  All download methods failed for this URL, trying next...")
            continue
        else:
            log.error("All URLs exhausted for source '%s'", source["name"])
            return False

        if not tmp_archive.exists() or tmp_archive.stat().st_size == 0:
            log.error("Download produced empty file for '%s'", source["name"])
            return False

        if not _extract_archive(tmp_archive, dest_dir):
            return False

    return True


def _collect_git_source(source: dict, dest_dir: Path) -> bool:
    """Clone a git repository source."""
    branch = source.get("branch")
    for url in source["urls"]:
        log.info("Trying git clone: %s", url)
        if _git_clone(url, dest_dir, branch=branch):
            return True

        # Fallback: try downloading as a ZIP from GitHub
        if "github.com" in url:
            zip_url = url.replace(".git", "") + f"/archive/refs/heads/{branch or 'main'}.zip"
            log.info("  Git clone failed, trying GitHub ZIP fallback: %s", zip_url)
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_archive = Path(tmpdir) / f"{source['name']}.zip"
                if _download_with_requests(zip_url, tmp_archive) or _download_with_curl(zip_url, tmp_archive):
                    if _extract_archive(tmp_archive, dest_dir):
                        return True

    return False


def _collect_hf_source(source: dict, dest_dir: Path) -> bool:
    """Download from HuggingFace using huggingface_hub or direct HTTP."""
    repo_id = source.get("hf_repo_id", "")

    # Strategy 1: Use huggingface_hub to snapshot download
    try:
        from huggingface_hub import snapshot_download
        log.info("  [hf] Downloading %s via huggingface_hub...", repo_id)
        dest_dir.mkdir(parents=True, exist_ok=True)
        local_dir = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(dest_dir),
            ignore_patterns=["*.md", "*.gitattributes"],
        )
        log.info("  [hf] Downloaded to %s", local_dir)
        return True
    except ImportError:
        log.info("  [hf] huggingface_hub not installed, trying direct HTTP...")
    except Exception as exc:
        log.warning("  [hf] huggingface_hub download failed: %s", exc)

    # Strategy 2: Use datasets library
    try:
        from datasets import load_dataset
        log.info("  [hf] Loading %s via datasets library...", repo_id)
        dest_dir.mkdir(parents=True, exist_ok=True)
        img_dir = dest_dir / "images"
        img_dir.mkdir(exist_ok=True)

        ds = load_dataset(repo_id, split="train")
        saved = 0
        for i, sample in enumerate(ds):
            img = sample.get("image")
            if img is not None:
                img.save(img_dir / f"{i:06d}.jpg")
                saved += 1

        log.info("  [hf] Saved %d images from %s", saved, repo_id)
        return saved > 0
    except ImportError:
        log.info("  [hf] datasets library not installed, trying direct URL...")
    except Exception as exc:
        log.warning("  [hf] datasets library failed: %s", exc)

    # Strategy 3: Direct URL download (parquet or zip)
    for url in source.get("urls", []):
        log.info("  [hf] Trying direct URL: %s", url)
        with tempfile.TemporaryDirectory() as tmpdir:
            ext = ".parquet" if "parquet" in url else ".zip"
            tmp_file = Path(tmpdir) / f"{source['name']}{ext}"
            if _download_with_requests(url, tmp_file) or _download_with_curl(url, tmp_file):
                if ext == ".zip":
                    return _extract_archive(tmp_file, dest_dir)
                else:
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(tmp_file, dest_dir / tmp_file.name)
                    log.info("  [hf] Saved %s to %s", tmp_file.name, dest_dir)
                    return True

    return False


# ---------------------------------------------------------------------------
# Collector registry
# ---------------------------------------------------------------------------
TYPE_HANDLERS = {
    "zip": _collect_zip_source,
    "git": _collect_git_source,
    "hf_repo": _collect_hf_source,
}


def collect_source(source: dict, raw_data_root: Path) -> dict[str, Any] | None:
    """Attempt to download and validate a single source. Returns validation report or None."""
    name = source["name"]
    source_type = source["type"]
    dest_dir = raw_data_root / name

    log.info("")
    log.info("=" * 60)
    log.info("  SOURCE: %s", source["label"])
    log.info("  Type: %s | Target: %s", source_type, dest_dir)
    log.info("=" * 60)

    if dest_dir.exists() and any(dest_dir.iterdir()):
        log.info("  Already downloaded -- validating existing data...")
        report = validate_source(dest_dir, name)
        if report["valid"]:
            log.info("  Existing data is valid, skipping re-download.")
            return report
        else:
            log.info("  Existing data invalid, re-downloading...")
            shutil.rmtree(dest_dir, ignore_errors=True)

    handler = TYPE_HANDLERS.get(source_type)
    if handler is None:
        log.error("  Unknown source type: %s", source_type)
        return None

    t0 = time.time()
    try:
        success = handler(source, dest_dir)
    except Exception as exc:
        log.error("  Unhandled error collecting '%s': %s", name, exc)
        success = False
    elapsed = time.time() - t0

    if not success:
        log.error("  FAILED to download '%s' (%.1fs) -- skipping.", name, elapsed)
        shutil.rmtree(dest_dir, ignore_errors=True)
        return None

    log.info("  Download completed in %.1fs. Validating...", elapsed)
    report = validate_source(dest_dir, name)

    if not report["valid"]:
        log.warning("  Validation FAILED for '%s'. Issues:", name)
        for issue in report["issues"]:
            log.warning("    - %s", issue)
        log.info("  Keeping data anyway -- prepare_dataset.py will attempt to use it.")

    return report


# ---------------------------------------------------------------------------
# Full collection report
# ---------------------------------------------------------------------------
def print_collection_summary(reports: list[dict[str, Any]], elapsed: float) -> None:
    """Print end-of-run summary."""
    log.info("")
    log.info("=" * 65)
    log.info("  COLLECTION SUMMARY  (%.1fs total)", elapsed)
    log.info("=" * 65)

    successful = [r for r in reports if r is not None and r["valid"]]
    failed = len(reports) - len(successful)

    for r in reports:
        if r is None:
            continue
        status = "OK" if r["valid"] else "WARN"
        log.info("  [%s] %-35s imgs=%-5d lbls=%-5d", status, r["name"], r["images"], r["labels"])

    log.info("-" * 65)
    log.info("  Downloaded: %d source(s) | Skipped/Failed: %d", len(successful), failed)

    total_imgs = sum(r["images"] for r in reports if r and r["valid"])
    total_lbls = sum(r["labels"] for r in reports if r and r["valid"])
    log.info("  Total images: %d | Total labels: %d", total_imgs, total_lbls)

    if total_imgs == 0:
        log.warning("")
        log.warning("  No images were downloaded!")
        log.warning("  You can manually place YOLO-format datasets in:")
        log.warning("    %s/<dataset_name>/", RAW_DATA_DIR)
        log.warning("  Then run: python prepare_dataset.py --force")
    log.info("=" * 65)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def get_source_names() -> list[str]:
    return [s["name"] for s in SOURCES]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Automated chess dataset collection (no Roboflow dependency).",
    )
    parser.add_argument(
        "--sources", nargs="*", default=None,
        help="Source names to download (default: all). Use --list to see options.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available sources and exit.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download even if data already exists.",
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Only validate existing raw_data/ without downloading.",
    )
    args = parser.parse_args()

    if args.list:
        print("Available dataset sources:")
        for s in SOURCES:
            print(f"  {s['name']:35s}  {s['label']}")
        return

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Raw data directory: %s", RAW_DATA_DIR)

    # Select sources
    if args.sources:
        selected = [s for s in SOURCES if s["name"] in args.sources]
        unknown = set(args.sources) - {s["name"] for s in SOURCES}
        if unknown:
            log.warning("Unknown source names (ignored): %s", unknown)
    else:
        selected = list(SOURCES)

    log.info("Sources to process: %s", [s["name"] for s in selected])

    if args.validate_only:
        reports = []
        for source in selected:
            src_dir = RAW_DATA_DIR / source["name"]
            if src_dir.exists():
                reports.append(validate_source(src_dir, source["name"]))
        print_validation_report(reports)
        return

    if args.force:
        log.info("--force: will re-download all sources")
        for source in selected:
            d = RAW_DATA_DIR / source["name"]
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)

    t0 = time.time()
    reports: list[dict[str, Any] | None] = []
    for source in selected:
        report = collect_source(source, RAW_DATA_DIR)
        reports.append(report)

    elapsed = time.time() - t0
    valid_reports = [r for r in reports if r is not None]
    print_collection_summary(reports, elapsed)
    if valid_reports:
        print_validation_report(valid_reports)


if __name__ == "__main__":
    main()
