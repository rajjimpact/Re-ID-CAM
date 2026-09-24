"""
scripts/calibrate_threshold.py  —  Phase 4: Threshold Calibration Tool.

Usage
─────
1. Collect a few crops of SAME person across cameras -> calibration/same/
2. Collect a few crops of DIFFERENT people          -> calibration/different/
3. Run: python scripts/calibrate_threshold.py

The tool prints the cosine similarity distribution and recommends a threshold.
Optionally patches config.py automatically.

Folder structure
────────────────
calibration/
  same/       <-- images of the SAME person from different cameras/angles
  different/  <-- images of DIFFERENT people (one person per sub-folder, or mixed)

Any .jpg/.png/.jpeg file is accepted.
"""
from __future__ import annotations
import os
import sys
import itertools
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

try:
    import cv2
except ImportError:
    print("ERROR: OpenCV not installed. Run: pip install opencv-python")
    sys.exit(1)


def load_images(folder: str):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    paths = [p for p in Path(folder).rglob("*") if p.suffix.lower() in exts]
    images = []
    for p in sorted(paths):
        img = cv2.imread(str(p))
        if img is not None:
            images.append((str(p), img))
    return images


def compute_embeddings(images, embedder):
    results = []
    for path, img in images:
        emb = embedder.embed(img)
        results.append((path, emb))
    return results


def cosine_pairs(embeddings_a, embeddings_b=None):
    """
    If embeddings_b is None: all pairs within embeddings_a.
    Otherwise: cross-product of a x b.
    Returns list of cosine scores.
    """
    scores = []
    if embeddings_b is None:
        for (_, e1), (_, e2) in itertools.combinations(embeddings_a, 2):
            scores.append(float(np.dot(e1, e2)))
    else:
        for (_, e1) in embeddings_a:
            for (_, e2) in embeddings_b:
                scores.append(float(np.dot(e1, e2)))
    return scores


def main():
    same_dir = Path("calibration/same")
    diff_dir = Path("calibration/different")

    if not same_dir.exists() or not diff_dir.exists():
        print(
            "\n[calibrate] Calibration folders not found.\n"
            "Create the following structure:\n\n"
            "  calibration/\n"
            "    same/       <- crops of the SAME person (any number)\n"
            "    different/  <- crops of DIFFERENT people\n\n"
            "Then re-run this script.\n"
        )
        sys.exit(1)

    # Load embedder
    from config import CONFIG
    from edge.reid_embedder import build_embedder
    print("[calibrate] Loading embedder (this may take a moment)...")
    embedder = build_embedder(
        embedding_dim=CONFIG.embedding_dim,
        checkpoint_path=CONFIG.reid_checkpoint_path,
    )
    print("[calibrate] Embedder ready.\n")

    # Load images
    same_imgs = load_images(str(same_dir))
    diff_imgs = load_images(str(diff_dir))
    print(f"[calibrate] Found {len(same_imgs)} same-person images, {len(diff_imgs)} different-person images.")

    if len(same_imgs) < 2:
        print("ERROR: Need at least 2 images in calibration/same/ to compute similarity pairs.")
        sys.exit(1)
    if len(diff_imgs) < 2:
        print("ERROR: Need at least 2 images in calibration/different/.")
        sys.exit(1)

    # Compute embeddings
    print("[calibrate] Embedding same-person images...")
    same_embs = compute_embeddings(same_imgs, embedder)
    print("[calibrate] Embedding different-person images...")
    diff_embs = compute_embeddings(diff_imgs, embedder)

    # Compute cosine similarities
    same_scores = cosine_pairs(same_embs)
    diff_scores = cosine_pairs(diff_embs)

    same_mean = np.mean(same_scores)
    same_std  = np.std(same_scores)
    diff_mean = np.mean(diff_scores)
    diff_std  = np.std(diff_scores)

    print("\n" + "=" * 60)
    print("  CALIBRATION RESULTS")
    print("=" * 60)
    print(f"  Same-person  pairs: n={len(same_scores):3d}  mean={same_mean:.4f}  std={same_std:.4f}  min={min(same_scores):.4f}  max={max(same_scores):.4f}")
    print(f"  Diff-person  pairs: n={len(diff_scores):3d}  mean={diff_mean:.4f}  std={diff_std:.4f}  min={min(diff_scores):.4f}  max={max(diff_scores):.4f}")

    # Recommended threshold = midpoint between diff_mean+diff_std and same_mean-same_std
    recommended = (diff_mean + diff_std + same_mean - same_std) / 2.0
    recommended = max(0.50, min(0.99, recommended))

    print(f"\n  Current config.py threshold : {CONFIG.similarity_threshold:.4f}")
    print(f"  RECOMMENDED threshold       : {recommended:.4f}")
    print()

    if same_mean <= diff_mean:
        print("  WARNING: Same-person scores are NOT higher than different-person scores.")
        print("  This suggests the model may not be loaded correctly, or your crops are too noisy.")
    else:
        gap = same_mean - diff_mean
        print(f"  Separation gap (same_mean - diff_mean) = {gap:.4f}")
        if gap > 0.10:
            print("  [GOOD] Clear separation — the trained model is working well.")
        elif gap > 0.04:
            print("  [OK]   Moderate separation — threshold calibration will help.")
        else:
            print("  [WARN] Small separation — consider collecting cleaner crops.")

    print("=" * 60)

    # Ask to patch config.py
    try:
        answer = input(f"\nPatch config.py to set similarity_threshold = {recommended:.4f}? [y/N] ").strip().lower()
    except EOFError:
        answer = "n"

    if answer == "y":
        config_path = Path("config.py")
        text = config_path.read_text(encoding="utf-8")
        import re
        new_text = re.sub(
            r"(similarity_threshold\s*:\s*float\s*=\s*)[\d.]+",
            f"\\g<1>{recommended:.4f}",
            text,
        )
        config_path.write_text(new_text, encoding="utf-8")
        print(f"[calibrate] config.py updated: similarity_threshold = {recommended:.4f}")
    else:
        print("[calibrate] config.py not changed.")


if __name__ == "__main__":
    main()
