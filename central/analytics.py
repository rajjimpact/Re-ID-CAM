"""
central/analytics.py — On-demand analytics computed from the identity registry.

Nothing is precomputed or cached — recalculated fresh from the registry each
call. Fast enough at prototype scale; add a cache layer if you hit latency issues.
"""
from __future__ import annotations
import time
from typing import Dict, List

from central.schemas import GlobalIdentity


def compute_store_analytics(identities: List[GlobalIdentity]) -> dict:
    """
    Store-wide summary metrics.

    Returns
    -------
    {
        "unique_shoppers"      : int,
        "currently_active"     : int,
        "avg_dwell_seconds"    : float,
        "zone_popularity"      : {zone: {"visit_count": int, "avg_dwell_seconds": float}},
        "computed_at"          : float   (unix timestamp)
    }
    """
    now = time.time()
    zone_visits: Dict[str, List[float]] = {}   # zone → list of dwell durations
    currently_active = 0

    for identity in identities:
        is_active = identity.active and (now - identity.last_seen) < 300
        if is_active:
            currently_active += 1
        for visit in identity.zone_visits:
            zone = visit.zone
            if zone not in zone_visits:
                zone_visits[zone] = []
            zone_visits[zone].append(visit.dwell_seconds)

    total_dwell_values = [
        v.dwell_seconds
        for identity in identities
        for v in identity.zone_visits
    ]
    avg_dwell = (sum(total_dwell_values) / len(total_dwell_values)
                 if total_dwell_values else 0.0)

    zone_popularity = {}
    for zone, dwells in zone_visits.items():
        zone_popularity[zone] = {
            "visit_count": len(dwells),
            "avg_dwell_seconds": round(sum(dwells) / len(dwells), 1) if dwells else 0.0,
        }

    return {
        "unique_shoppers": len(identities),
        "currently_active": currently_active,
        "avg_dwell_seconds": round(avg_dwell, 1),
        "zone_popularity": zone_popularity,
        "computed_at": now,
    }


def compute_shopper_summary(identities: List[GlobalIdentity]) -> List[dict]:
    """
    Per-shopper summary list, newest-active-first.
    """
    summaries = [i.to_summary_dict() for i in identities]
    summaries.sort(key=lambda s: (-int(s["active"]), -s["last_seen"]))
    return summaries
