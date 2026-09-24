"""
tests/test_reid_engine.py — Unit tests for GlobalIdentityRegistry.

Tests cover:
  - New identity registration below threshold
  - Same-identity merge above threshold
  - Cross-camera match detection
  - Zone-visit path building (Entrance → Electronics → Checkout)
  - TTL expiry closes open zone visit and marks identity inactive
  - Identity persistence through brief occlusion (TTL not yet expired)
"""
import sys
import os
import time
import unittest
import numpy as np

# Ensure reid_system root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from central.reid_engine import GlobalIdentityRegistry
from central.schemas import TrackPayload


# ── Helper: make a unit-length embedding ─────────────────────────────────────

def make_embedding(dim: int = 512, seed: int = 0) -> list:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(dim,)).astype(np.float32)
    v /= np.linalg.norm(v)
    return v.tolist()


def similar_embedding(original: list, noise: float = 0.02) -> list:
    """Return an embedding very close to original (should match above threshold)."""
    v = np.array(original, dtype=np.float32)
    rng = np.random.default_rng(seed=999)
    v += rng.normal(scale=noise, size=v.shape).astype(np.float32)
    v /= np.linalg.norm(v)
    return v.tolist()


def make_payload(camera_id: str, zone: str, embedding: list,
                 local_track_id: int = 1, timestamp: float = None) -> TrackPayload:
    return TrackPayload(
        camera_id=camera_id,
        local_track_id=local_track_id,
        timestamp=timestamp or time.time(),
        bbox=[10.0, 10.0, 80.0, 160.0],
        embedding=embedding,
        zone=zone,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestNewIdentityRegistration(unittest.TestCase):

    def setUp(self):
        self.registry = GlobalIdentityRegistry(
            embedding_dim=512,
            similarity_threshold=0.78,
            inactive_ttl_seconds=300.0,
        )

    def test_first_payload_creates_new_identity(self):
        emb = make_embedding(seed=1)
        gid, cross, _ = self.registry.ingest(make_payload("cam_entrance", "Entrance", emb))
        self.assertTrue(gid.startswith("GSI-"))
        self.assertFalse(cross)
        self.assertEqual(self.registry.identity_count(), 1)

    def test_dissimilar_embeddings_create_separate_identities(self):
        """Two very different embeddings (opposite vectors) → 2 identities."""
        emb1 = make_embedding(seed=1)
        emb2 = [-x for x in emb1]   # anti-parallel → cosine similarity ≈ -1
        norm2 = np.linalg.norm(emb2)
        emb2 = [x / norm2 for x in emb2]

        self.registry.ingest(make_payload("cam_entrance", "Entrance", emb1))
        self.registry.ingest(make_payload("cam_entrance", "Entrance", emb2))
        self.assertEqual(self.registry.identity_count(), 2)


class TestIdentityMerge(unittest.TestCase):

    def setUp(self):
        self.registry = GlobalIdentityRegistry(
            embedding_dim=512,
            similarity_threshold=0.78,
            inactive_ttl_seconds=300.0,
        )

    def test_similar_embeddings_same_camera_merge(self):
        emb1 = make_embedding(seed=10)
        emb2 = similar_embedding(emb1, noise=0.01)
        now = time.time()

        gid1, _, _ = self.registry.ingest(make_payload("cam_entrance", "Entrance", emb1, timestamp=now))
        gid2, _, _ = self.registry.ingest(make_payload("cam_entrance", "Entrance", emb2, timestamp=now + 1))
        self.assertEqual(gid1, gid2)
        self.assertEqual(self.registry.identity_count(), 1)

    def test_similar_embeddings_cross_camera_merge_and_flag(self):
        emb1 = make_embedding(seed=20)
        emb2 = similar_embedding(emb1, noise=0.01)
        now = time.time()

        gid1, cross1, _ = self.registry.ingest(make_payload("cam_entrance", "Entrance", emb1, timestamp=now))
        gid2, cross2, _ = self.registry.ingest(make_payload("cam_electronics", "Electronics", emb2, timestamp=now + 5))

        self.assertEqual(gid1, gid2, "Same person should resolve to same Global ID")
        self.assertFalse(cross1, "First registration is never a cross-camera match")
        self.assertTrue(cross2, "Second payload from different camera should flag cross-camera match")

    def test_no_cross_camera_flag_when_same_camera(self):
        emb = make_embedding(seed=30)
        now = time.time()
        self.registry.ingest(make_payload("cam_entrance", "Entrance", emb, timestamp=now))
        _, cross, _ = self.registry.ingest(make_payload("cam_entrance", "Entrance",
                                                        similar_embedding(emb), timestamp=now + 2))
        self.assertFalse(cross)


class TestZonePathBuilding(unittest.TestCase):

    def setUp(self):
        self.registry = GlobalIdentityRegistry(
            embedding_dim=512,
            similarity_threshold=0.78,
            inactive_ttl_seconds=300.0,
        )

    def test_zone_path_builds_correctly(self):
        emb = make_embedding(seed=40)
        now = time.time()

        zones = ["Entrance", "Electronics", "Checkout"]
        cameras = ["cam_entrance", "cam_electronics", "cam_checkout"]
        gid = None

        for i, (zone, cam) in enumerate(zip(zones, cameras)):
            close_emb = similar_embedding(emb, noise=0.01) if i > 0 else emb
            gid, _, _ = self.registry.ingest(
                make_payload(cam, zone, close_emb, timestamp=now + i * 30)
            )

        identity = self.registry.get_by_id(gid)
        self.assertIsNotNone(identity)
        self.assertEqual(identity.path, zones,
                         f"Expected path {zones}, got {identity.path}")
        self.assertEqual(len(identity.zone_visits), 3)

    def test_same_zone_does_not_add_duplicate_visit(self):
        emb = make_embedding(seed=50)
        now = time.time()

        gid, _, _ = self.registry.ingest(make_payload("cam_entrance", "Entrance", emb, timestamp=now))
        _, _, _ = self.registry.ingest(make_payload("cam_entrance", "Entrance",
                                                    similar_embedding(emb, 0.01), timestamp=now + 1))
        identity = self.registry.get_by_id(gid)
        self.assertEqual(len(identity.zone_visits), 1,
                         "Staying in the same zone should not create a new visit")


class TestTTLExpiry(unittest.TestCase):

    def test_stale_identity_gets_marked_inactive(self):
        registry = GlobalIdentityRegistry(
            embedding_dim=512,
            similarity_threshold=0.78,
            inactive_ttl_seconds=2.0,   # very short for testing
        )
        emb = make_embedding(seed=60)
        past = time.time() - 10.0   # 10 seconds ago

        gid, _, _ = registry.ingest(make_payload("cam_entrance", "Entrance", emb, timestamp=past))

        # Trigger a new ingest at "now" — the TTL expiry fires during this call
        new_emb = make_embedding(seed=999)   # different person
        _ = registry.ingest(make_payload("cam_entrance", "Entrance", new_emb))

        identity = registry.get_by_id(gid)
        self.assertFalse(identity.active, "Identity should be inactive after TTL")
        # Visit should be closed
        self.assertIsNotNone(
            identity.zone_visits[-1].exit_time,
            "Open zone visit should be closed on TTL expiry"
        )

    def test_identity_within_ttl_stays_active(self):
        registry = GlobalIdentityRegistry(
            embedding_dim=512,
            similarity_threshold=0.78,
            inactive_ttl_seconds=300.0,
        )
        emb = make_embedding(seed=70)
        gid, _, _ = registry.ingest(make_payload("cam_entrance", "Entrance", emb))
        identity = registry.get_by_id(gid)
        self.assertTrue(identity.active)


class TestAnalyticsIntegration(unittest.TestCase):
    """Smoke-test that analytics functions don't crash on real registry data."""

    def test_store_analytics_smoke(self):
        from central.analytics import compute_store_analytics
        registry = GlobalIdentityRegistry(embedding_dim=512, similarity_threshold=0.78)

        emb = make_embedding(seed=80)
        now = time.time()
        registry.ingest(make_payload("cam_entrance", "Entrance", emb, timestamp=now))
        registry.ingest(make_payload("cam_electronics", "Electronics",
                                     similar_embedding(emb, 0.01), timestamp=now + 5))

        analytics = compute_store_analytics(registry.get_all())
        self.assertIn("unique_shoppers", analytics)
        self.assertIn("zone_popularity", analytics)
        self.assertGreaterEqual(analytics["unique_shoppers"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
