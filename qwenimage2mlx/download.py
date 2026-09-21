"""Fault-tolerant download of Qwen-Image-2.1 weights from the HF Hub.

Guarantees:
- Resumable: partially-downloaded files continue where they stopped, complete
  files are skipped (huggingface_hub verifies size + etag).
- Fail-proof: retries each file with exponential backoff, indefinitely by
  default, surviving flaky networks and dropped connections.
- Reuse-aware: on restart it re-checks every needed file and only fetches what
  is missing or incomplete.
- Elegant progress: a clean per-file + overall progress bar in the terminal.
"""
import os
import time

from .config import HF_REPO, NEEDED_DIRS, NEEDED_FILES, ORIGINAL_DIR


def _list_repo_targets(repo, token):
    """Return [(rel_path, size_bytes)] for every needed file in the repo."""
    from huggingface_hub import HfApi

    api = HfApi()
    info = api.repo_info(repo, files_metadata=True, token=token)
    wanted_prefixes = tuple(f"{d}/" for d in NEEDED_DIRS)
    targets = []
    for s in info.siblings:
        rf = s.rfilename
        if rf.startswith(wanted_prefixes) or rf in NEEDED_FILES:
            targets.append((rf, s.size or 0))
    return targets


def _needs_download(dest, rel, size):
    path = os.path.join(dest, rel)
    if not os.path.exists(path):
        return True
    if size and os.path.getsize(path) != size:
        return True
    return False


def _human(n):
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PiB"


def ensure_original(repo=HF_REPO, dest=ORIGINAL_DIR, max_retries=None, verbose=True):
    """Download all needed files, resuming and retrying until complete.

    max_retries: per-file retry cap; None = retry forever (until success).
    Returns the local directory once every needed file is present and complete.
    """
    from huggingface_hub import hf_hub_download
    from tqdm import tqdm

    try:
        from huggingface_hub import get_token
        token = get_token()
    except Exception:
        try:
            from huggingface_hub.utils import HfFolder
            token = HfFolder.get_token()
        except Exception:
            token = None
    os.makedirs(dest, exist_ok=True)

    # Discover targets (with a couple of retries — the metadata call can flake too).
    targets = None
    for attempt in range(5):
        try:
            targets = _list_repo_targets(repo, token)
            break
        except Exception as e:
            if verbose:
                print(f"[download] repo listing failed ({e}); retry {attempt+1}/5", flush=True)
            time.sleep(2 ** attempt)
    if targets is None:
        raise RuntimeError(f"Could not list repo {repo} after retries.")

    pending = [(rel, sz) for rel, sz in targets if _needs_download(dest, rel, sz)]
    done = [(rel, sz) for rel, sz in targets if not _needs_download(dest, rel, sz)]
    total_bytes = sum(sz for _, sz in targets)
    have_bytes = sum(sz for _, sz in done)

    if not pending:
        if verbose:
            print(f"[download] complete — {len(targets)} files, {_human(total_bytes)} present.", flush=True)
        return dest

    if verbose:
        print(f"[download] {len(done)}/{len(targets)} files present "
              f"({_human(have_bytes)}/{_human(total_bytes)}); "
              f"fetching {len(pending)} remaining.", flush=True)

    overall = tqdm(total=total_bytes, initial=have_bytes, unit="B", unit_scale=True,
                   unit_divisor=1024, desc="Qwen-Image-2.1", disable=not verbose,
                   dynamic_ncols=True)

    for rel, sz in pending:
        attempt = 0
        while True:
            try:
                # hf_hub_download resumes partial files and skips complete ones.
                hf_hub_download(
                    repo_id=repo, filename=rel, local_dir=dest,
                    token=token, force_download=False,
                )
                overall.update(sz)
                break
            except KeyboardInterrupt:
                overall.close()
                raise
            except Exception as e:
                attempt += 1
                if max_retries is not None and attempt > max_retries:
                    overall.close()
                    raise RuntimeError(f"Failed to download {rel} after {max_retries} retries: {e}")
                wait = min(60, 2 ** min(attempt, 6))
                if verbose:
                    tqdm.write(f"[download] {rel} failed ({e}); retry {attempt} in {wait}s")
                time.sleep(wait)
    overall.close()

    # Final verification pass — anything still incomplete gets one more loop.
    still = [(rel, sz) for rel, sz in targets if _needs_download(dest, rel, sz)]
    if still:
        if verbose:
            print(f"[download] {len(still)} files still incomplete; re-running.", flush=True)
        return ensure_original(repo, dest, max_retries, verbose)

    if verbose:
        print(f"[download] complete — {len(targets)} files, {_human(total_bytes)}.", flush=True)
    return dest
