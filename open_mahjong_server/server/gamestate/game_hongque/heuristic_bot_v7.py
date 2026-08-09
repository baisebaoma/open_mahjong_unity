"""Hongque heuristic v7 bot (user_id=3): v3 + D3 claim suppression only.

Experimental change over ``heuristic_bot_v3`` (see
HONGQUE_HEURISTIC_RESEARCH.md §10):

- **D3: claim suppression under threat** — when threat is high, a claim forces
  a discard; if even the safest discard stays dangerous (danger > cap), pass
  instead of claiming.  v5 bundled this with D1 (meld-weighted threat) and
  lost to v3; v7 isolates D3 using v3's *original* threat level to measure
  the component alone.

Everything else (discard, folding, supplements, kong) is identical to v3 and
re-exported from it.  This module is deliberately import/mutation-free.
"""
from __future__ import annotations

from typing import Optional, Sequence

from .heuristic_bot import (
    V2Value,
    _claim_advances,
    _meld_from_candidate,
    _remove_tiles,
    evaluate_hand,
)
from .heuristic_bot_v3 import (
    _DEFENSE_THREAT,
    _WALL_START,
    OpponentView,
    _threat_level,
    _tile_danger,
    choose_discard,
    choose_turn_plan,
)
from .win_check import is_winning_hand

# D3: under threat, claiming exposes the hand and forces a discard that may
# deal in.  We only suppress a claim when the forced discard stays dangerous
# after the folding logic has already picked its safest option.
_CLAIM_DANGER_CAP = 2.0


def choose_claim_plan(
    hand: Sequence[str],
    open_melds: Sequence[dict],
    candidates: Sequence[dict],
    visible_codes: Sequence[str],
    *,
    opponents: Sequence[OpponentView] = (),
    wall_count: int = _WALL_START,
) -> dict:
    """Choose an authoritative candidate id or pass after another discard.

    D3: when threat is high, evaluate the discard a claim would force with the
    real opponent views; if even the safest option stays dangerous, pass.
    """
    for candidate in candidates:
        if candidate.get("kind") == "win":
            return {"action": "claim", "candidate_id": candidate.get("id")}

    before = evaluate_hand(hand, open_melds, visible_codes)
    threat = _threat_level(opponents, wall_count)
    best_candidate: Optional[dict] = None
    best_rank: Optional[tuple] = None

    for candidate in candidates:
        if candidate.get("kind") not in {"sequence", "triplet", "rainbow"}:
            continue
        after = _remove_tiles(hand, candidate.get("hand_tiles", ()))
        if after is None:
            continue
        melds_after = list(open_melds) + [_meld_from_candidate(candidate)]
        priority = int(candidate.get("priority", 0) or 0)
        if is_winning_hand(after, melds_after):
            value = V2Value(0, 126, 1_000_000, 1_000_000)
            forced_danger = 0.0
        elif after:
            forced_tile, value = choose_discard(
                after, melds_after, visible_codes, None,
                opponents=opponents, wall_count=wall_count,
            )
            forced_danger = _tile_danger(forced_tile, opponents) if forced_tile else 0.0
        else:
            continue
        if not _claim_advances(before, value, priority):
            continue
        if threat >= _DEFENSE_THREAT and forced_danger > _CLAIM_DANGER_CAP:
            continue  # D3: claiming would force a dangerous discard under threat
        rank = (-value.distance, value.ukeire, value.points, value.style,
                priority, len(candidate.get("tiles", ())))
        if best_rank is None or rank > best_rank:
            best_candidate, best_rank = candidate, rank

    if best_candidate is None:
        return {"action": "pass"}
    return {"action": "claim", "candidate_id": best_candidate.get("id")}
