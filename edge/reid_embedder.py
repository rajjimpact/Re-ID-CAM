"""
edge/reid_embedder.py — Person appearance embedding.

Primary  : ResNet50Embedder  — loads your trained checkpoint, outputs 512-d
           L2-normalised cosine-ready vectors.
Fallback : HistogramFallbackEmbedder — colour histogram, no PyTorch needed.

§10.3 checkpoint key contract
─────────────────────────────
The checkpoint file must contain:
    "backbone_state_dict" → weights for the ResNet50 trunk (fc replaced by Identity)
    "projection"          → numpy array of shape (2048, 512)  [already float32]
    "bnneck_state_dict"   → state_dict for nn.BatchNorm1d(512)  [optional but recommended]

The critical log line that confirms the real model is active:
    [reid_embedder] Loaded trained Re-ID weights from <path>  (eval results at export time: ...)

BNNeck note
───────────
The training notebook applies BatchNorm1d after the linear projection before
computing rank-1/mAP.  Production must use the same post-BNNeck L2-normalised
vector; otherwise the embedding space doesn't match what was evaluated in Colab.
"""
from __future__ import annotations
import logging
import os
from typing import Optional
import numpy as np

log = logging.getLogger(__name__)

# ── Availability flags ────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore
    _HAS_TORCH = False

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


# ─────────────────────────────────────────────────────────────────────────────
# Base class — defines the interface both embedders honour
# ─────────────────────────────────────────────────────────────────────────────

class BaseEmbedder:
    embedding_dim: int

    def embed(self, crop: np.ndarray) -> np.ndarray:
        """
        Args:
            crop: BGR uint8 image of a person bounding-box crop.
        Returns:
            1-D float32 numpy array, L2-normalised, length = self.embedding_dim.
        """
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# Primary: ResNet50 + BNNeck + trained projection (§10.3)
# ─────────────────────────────────────────────────────────────────────────────

class ResNet50Embedder(BaseEmbedder):
    """
    ResNet50 backbone with the FC layer replaced by Identity, followed by a
    (2048 → 512) linear projection and a BNNeck (BatchNorm1d) trained on
    Market-1501.

    The embedding returned by embed() matches the post-BNNeck L2-normalised
    space that was used to compute rank-1/mAP during evaluation in Colab.
    Skipping BNNeck would produce a different embedding space and degrade
    cross-camera matching accuracy.

    Requires PyTorch + torchvision. If those are unavailable, build_embedder()
    will never instantiate this class — it falls back to HistogramFallbackEmbedder.

    Checkpoint keys (exactly as exported by the training notebook):
        "backbone_state_dict" : state_dict for the truncated ResNet50
        "projection"          : numpy float32 array, shape (2048, 512)
        "bnneck_state_dict"   : state_dict for nn.BatchNorm1d(512)  ← recommended
        "eval_results"        : optional string displayed on load
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        device: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        if not _HAS_TORCH:
            raise RuntimeError("PyTorch is required for ResNet50Embedder.")

        self.embedding_dim = embedding_dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # ── Build backbone ────────────────────────────────────────────────────
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        backbone.fc = nn.Identity()          # strip classification head → 2048-d output
        self.backbone = backbone.to(self.device).eval()

        # ── BNNeck — must be applied post-projection at inference ─────────────
        # Initialise to near-identity defaults; real running stats are loaded
        # from the checkpoint below (if the key is present).
        self.bnneck = nn.BatchNorm1d(embedding_dim).to(self.device)
        self.bnneck.eval()   # use stored running_mean/running_var, NOT batch stats

        # ── Load checkpoint (§10.3 key contract) ─────────────────────────────
        if checkpoint_path and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device,
                              weights_only=False)

            # backbone_state_dict ← exact key from training notebook
            if "backbone_state_dict" in ckpt:
                self.backbone.load_state_dict(ckpt["backbone_state_dict"])
            else:
                print(
                    f"[reid_embedder] WARNING: 'backbone_state_dict' key missing "
                    f"in {checkpoint_path}. Keys found: {list(ckpt.keys())}"
                )

            # projection ← exact key from training notebook (2048, 512) numpy array
            if "projection" in ckpt:
                proj = np.array(ckpt["projection"], dtype=np.float32)
            else:
                print(
                    f"[reid_embedder] WARNING: 'projection' key missing "
                    f"in {checkpoint_path}. Keys found: {list(ckpt.keys())}"
                )
                proj = self._random_projection(embedding_dim)

            # bnneck_state_dict ← BatchNorm1d running stats from training
            if "bnneck_state_dict" in ckpt:
                self.bnneck.load_state_dict(ckpt["bnneck_state_dict"])
                self.bnneck.eval()   # re-call after load_state_dict to be explicit
            else:
                # Checkpoint pre-dates BNNeck export — warn but do not crash.
                # The layer will act near-identity (weight=1, bias=0, running
                # stats=default), which is much less damaging than skipping it
                # entirely or crashing the whole pipeline.
                print(
                    f"[reid_embedder] WARNING: 'bnneck_state_dict' key missing "
                    f"in {checkpoint_path}. BNNeck will use default (near-identity) "
                    f"stats. Re-export the checkpoint from Colab to fix this."
                )

            # ── THE KEY LOG LINE — confirms real model is active ──────────────
            eval_results = ckpt.get("eval_results", "n/a")
            print(
                f"[reid_embedder] Loaded trained Re-ID weights from {checkpoint_path} "
                f"(eval results at export time: {eval_results})"
            )
        else:
            if checkpoint_path:
                print(
                    f"[reid_embedder] WARNING: checkpoint not found at "
                    f"'{checkpoint_path}'. Using random projection (untrained)."
                )
            else:
                print("[reid_embedder] No checkpoint path supplied. Using random projection.")
            proj = self._random_projection(embedding_dim)

        # Store as a non-trained tensor (no grad needed at inference)
        self.projection = torch.from_numpy(proj).to(self.device)   # (2048, 512)

        # ── Image pre-processing (Market-1501 standard) ───────────────────────
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    @staticmethod
    def _random_projection(embedding_dim: int) -> np.ndarray:
        """Reproducible random projection used as untrained baseline."""
        rng = np.random.default_rng(seed=42)
        proj = rng.normal(size=(2048, embedding_dim)).astype(np.float32)
        proj /= np.linalg.norm(proj, axis=0, keepdims=True)
        return proj

    def embed(self, crop: np.ndarray) -> np.ndarray:
        """
        BGR uint8 crop → 512-d L2-normalised float32 vector.

        Pipeline:
          1. Preprocess (resize, normalise)
          2. ResNet50 backbone → 2048-d features
          3. Linear projection → 512-d
          4. BNNeck (BatchNorm1d in eval mode) → normalised features
          5. L2 normalise → cosine-ready vector

        Step 4 is critical: the embedding space used for rank-1/mAP evaluation
        in Colab is post-BNNeck.  Skipping it produces a different space and
        degrades cross-camera matching.
        """
        if crop is None or crop.size == 0:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        with torch.no_grad():
            # BGR → RGB for torchvision
            rgb = crop[:, :, ::-1].copy()
            tensor = self.preprocess(rgb).unsqueeze(0).to(self.device)  # (1, 3, 256, 128)

            features = self.backbone(tensor)            # (1, 2048)
            projected = features @ self.projection      # (1, 512)
            bn_out = self.bnneck(projected)             # (1, 512) — post-BNNeck
            embedding = bn_out.squeeze(0)               # (512,)

            # L2 normalise → cosine similarity = dot product (FAISS IndexFlatIP)
            norm = embedding.norm(p=2)
            if norm > 0:
                embedding = embedding / norm

            return embedding.cpu().numpy().astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Fallback: colour histogram (no PyTorch needed)
# ─────────────────────────────────────────────────────────────────────────────

class HistogramFallbackEmbedder(BaseEmbedder):
    """
    Dependency-light fallback when PyTorch isn't available.

    Computes an HSV histogram (H: 32 bins, S: 8 bins, V: 8 bins → 2048 values)
    and projects it down to embedding_dim with a fixed random matrix.
    Not suitable for real re-identification — exists only to let the rest of
    the pipeline run end-to-end without ML dependencies.
    """

    _BINS = (32, 8, 8)   # → 2048 features before projection

    def __init__(self, embedding_dim: int = 512) -> None:
        self.embedding_dim = embedding_dim
        rng = np.random.default_rng(seed=0)
        raw_dim = self._BINS[0] * self._BINS[1] * self._BINS[2]
        proj = rng.normal(size=(raw_dim, embedding_dim)).astype(np.float32)
        proj /= np.linalg.norm(proj, axis=0, keepdims=True)
        self._projection = proj

    def embed(self, crop: np.ndarray) -> np.ndarray:
        if crop is None or crop.size == 0 or not _HAS_CV2:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        import cv2
        resized = cv2.resize(crop, (64, 128))
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv], [0, 1, 2], None,
            list(self._BINS),
            [0, 180, 0, 256, 0, 256],
        ).flatten().astype(np.float32)

        norm = np.linalg.norm(hist)
        if norm > 0:
            hist /= norm

        vec = hist @ self._projection   # (embedding_dim,)
        norm2 = np.linalg.norm(vec)
        if norm2 > 0:
            vec /= norm2
        return vec.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Factory (§10.4)
# ─────────────────────────────────────────────────────────────────────────────

def build_embedder(
    embedding_dim: int = 512,
    checkpoint_path: Optional[str] = None,
) -> BaseEmbedder:
    """
    Return the best available embedder.

    Priority:
      1. ResNet50Embedder with trained checkpoint  (requires torch + torchvision)
      2. HistogramFallbackEmbedder                 (requires only numpy + cv2)

    Checkpoint failures are logged with a full traceback — they can never be
    silently swallowed.  If ResNet50Embedder.__init__ raises for any reason
    (bad weights, missing keys, CUDA OOM, etc.), the error is printed AND
    recorded by logging.exception() before falling back.
    """
    if _HAS_TORCH:
        try:
            return ResNet50Embedder(
                embedding_dim=embedding_dim,
                checkpoint_path=checkpoint_path,
            )
        except Exception as e:
            print(f"[reid_embedder] ERROR: ResNet50Embedder init failed: {e}")
            log.exception(
                "ResNet50Embedder init failed — falling back to HistogramFallbackEmbedder. "
                "Re-ID quality will be severely degraded. Fix the checkpoint issue above."
            )
    print("[reid_embedder] Using HistogramFallbackEmbedder (no PyTorch or init error).")
    return HistogramFallbackEmbedder(embedding_dim=embedding_dim)
