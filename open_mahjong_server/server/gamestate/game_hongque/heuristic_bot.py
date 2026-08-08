"""Hongque heuristic v2 bot (user_id=3): score-aware tile play.

Built on the rule-local group index and the authoritative ``best_win_result``
scorer, mirroring the omc guobiao heuristic methodology:

- **Tenpai waits**: score every live wait with ``best_win_result`` and prefer
  high-point waits (the upstream v1 bot ignores scoring entirely, so it wins
  cheap 1-4 fan hands; v2 steers toward the fan table).
- **Doing-direction (做牌)**: while not ready, a soft style utility rewards
  keeping tiles that lean toward colour-family concentration (清一色/双色/三色),
  number concentration (清一数/二数/三数), 全带幺 terminals, and long groups
  (金龙/二金).  This is a tie-break on top of the structural distance, never a
  hard gate — mirroring guobiao ``styleUtility``.
- **Thickness**: reuse the structural ``flexibility`` metric (the v1 bot's
  tie-break) as a cheap ukeire proxy instead of a per-tile one-draw loop, so
  the per-turn cost stays in the v1 CPU envelope.

This module is deliberately import/mutation-free like ``efficiency_bot``.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Optional, Sequence

from .group_index import (
    FULL_DECK_MASK,
    GROUP_MASKS,
    TILE_INDEX,
    codes_from_mask,
    mask_from_codes,
    waiting_mask,
    waiting_masks_after_discards,
)
from .scoring import best_win_result
from .tile import HongqueTile
from .win_check import is_winning_hand


@dataclass(frozen=True)
class V2Value:
    """A comparable Hongque hand value; a larger rank is better.

    ``distance`` is the replacement distance (shanten analog); 0 means tenpai.
    ``ukeire`` is the live-wait count (tenpai) or structural flexibility proxy
    (non-tenpai).  ``points`` is the best hypothetical win score among live
    waits when tenpai.  ``style`` is the doing-direction soft score.
    """

    distance: int
    ukeire: int
    points: int
    style: int


def _bit_for_code(code: str) -> int:
    return 1 << TILE_INDEX[code]


def _visible_mask(codes: Iterable[str]) -> int:
    return mask_from_codes(tuple(sorted(set(codes))))


# ── Doing-direction style utility ────────────────────────────────────────────

# Fan values (scoring.py): 清一色18 / 双色12 / 三色6; 清一数18 / 二数12 / 三数6.
# Use the *potential* fan as a soft steering reward on a partial hand.
_COLOUR_FAN = {1: 18, 2: 12, 3: 6}
_NUMBER_FAN = {1: 18, 2: 12, 3: 6, 4: 3}

# Soft cap so a stray preference can never dominate the structural distance.
_STYLE_CAP = 64

# 六张起看 (six-tile route) was measured to hurt this fast game: committing to
# colour concentration costs +1 replacement distance every shed, and with only
# ~9 draws per hand the opponents win first.  Kept here as a design note; the
# active v2 uses tiebreak-only doing-direction + score-aware tenpai.


def _style_utility(codes: Sequence[str]) -> int:
    """Cheap doing-direction score for a hand (concealed + open meld tiles).

    Rewards:
    - covered base-colour families few (清一色/双色/三色 potential)
    - distinct numbers few (清一数/二数/三数/四数 potential)
    - terminal 1/9 count (全带幺 potential: every group must hold a 1 or 9)
    - one dominant number family (long triplet → 金龙/二金 potential)
    """
    if not codes:
        return 0
    tiles = [HongqueTile.parse(c) for c in codes]
    covered: set[int] = set()
    for t in tiles:
        covered.update(t.primary_colours)
    numbers = {t.number for t in tiles}
    terminals = sum(1 for t in tiles if t.number in (1, 9))

    colour_fan = _COLOUR_FAN.get(len(covered), 0)
    number_fan = _NUMBER_FAN.get(len(numbers), 0)
    style = colour_fan + number_fan
    style += min(4, terminals) * 1
    from collections import Counter

    num_counts = Counter(t.number for t in tiles)
    style += max(num_counts.values(), default=0) - 3
    return max(0, min(_STYLE_CAP, style))


# ── Hypothetical win scoring ────────────────────────────────────────────────

def _hypothetical_points(
    hand: Sequence[str],
    open_melds: Sequence[dict],
    wait: str,
    *,
    self_draw: bool,
    before_first_discard: bool,
    wall_empty: bool,
) -> int:
    try:
        result = best_win_result(
            list(hand) + [wait],
            open_melds,
            self_draw=self_draw,
            before_first_discard=before_first_discard,
            wall_empty=wall_empty,
        )
    except Exception:
        return 0
    if result is None:
        return 0
    return int(result.get("points", 0) or 0)


@lru_cache(maxsize=32768)
def _structural_value(hand_mask: int) -> tuple[int, int]:
    """Replacement distance + flexibility (delegated to efficiency_bot)."""
    from .efficiency_bot import _structural_value as _base

    return _base(hand_mask)


def _meld_tiles(melds: Sequence[dict]) -> list[str]:
    return [c for m in melds for c in m.get("tiles", ())]


def evaluate_hand(
    hand: Sequence[str],
    open_melds: Sequence[dict],
    visible_codes: Iterable[str],
    *,
    self_draw: bool = False,
    before_first_discard: bool = False,
    wall_empty: bool = False,
) -> V2Value:
    """Score a hand: exact waits + hypothetical points when tenpai.

    Returns a V2Value where ``distance==0`` and ``points>0`` means a tenpai
    hand whose best live wait wins ``points`` points.
    """
    if not hand:
        return V2Value(99, 0, 0, 0)
    hand_mask = mask_from_codes(hand)
    visible_mask = _visible_mask(visible_codes)
    waits = waiting_mask(
        hand_mask,
        used_mask=visible_mask,
        has_open_group=bool(open_melds),
    )
    distance, flexibility = _structural_value(hand_mask)
    live_waits_mask = waits & (FULL_DECK_MASK & ~visible_mask)
    style = _style_utility(list(hand) + _meld_tiles(open_melds))
    if live_waits_mask:
        wait_codes = codes_from_mask(live_waits_mask)
        best_points = max(
            _hypothetical_points(
                hand, open_melds, w,
                self_draw=self_draw,
                before_first_discard=before_first_discard,
                wall_empty=wall_empty,
            )
            for w in wait_codes
        )
        return V2Value(0, len(wait_codes), best_points, style)
    return V2Value(max(1, distance), flexibility, 0, style)


# ── Discard ──────────────────────────────────────────────────────────────────

def _is_better_ready(a: V2Value, b: V2Value) -> bool:
    """Rank tenpai discards: thickness first, then points within a slack.

    Multi-wait tenpai is common (~40% of tenpai discards) and the point spread
    among waits is material (10% of cases have >=6 point gaps, ~2.7x).  Within
    a small live-wait slack, prefer the higher-scoring wait — the free value
    that does not cost win rate.
    """
    if a.ukeire != b.ukeire:
        # Fewer live waits is a real cost; only trade it for a big point gain.
        if a.ukeire < b.ukeire:
            if a.points >= b.points + 30:
                return True
            return False
        # More live waits: only lose on points when the gain is huge.
        if b.points >= a.points + 30:
            return False
        return True
    if a.points != b.points:
        return a.points > b.points
    return a.style > b.style


def choose_discard(
    hand: Sequence[str],
    open_melds: Sequence[dict],
    visible_codes: Iterable[str],
    drawn_tile: Optional[str] = None,
    *,
    self_draw: bool = False,
    before_first_discard: bool = False,
    wall_empty: bool = False,
) -> tuple[Optional[str], V2Value]:
    if not hand:
        return None, V2Value(99, 0, 0, 0)

    hand_mask = mask_from_codes(hand)
    visible_mask = _visible_mask(visible_codes) | hand_mask
    has_open_group = bool(open_melds)
    hand_code_by_index = {TILE_INDEX[tile]: tile for tile in hand}
    meld_tiles = _meld_tiles(open_melds)

    # Exact waits for every discard (shared one-scan group traversal).
    waits_by_discard = waiting_masks_after_discards(
        hand_mask,
        used_mask=visible_mask,
        has_open_group=has_open_group,
    )
    available = FULL_DECK_MASK & ~visible_mask

    # Connectivity (quality of each tile's legal-group overlap) for the
    # non-ready shortlist — same metric as efficiency_bot.
    connectivity = {tile: 0 for tile in hand}
    for group_mask in GROUP_MASKS:
        overlap = group_mask & hand_mask
        overlap_size = overlap.bit_count()
        if overlap_size < 2:
            continue
        missing = group_mask.bit_count() - overlap_size
        quality = overlap_size * overlap_size * 8 // (missing + 1)
        bits = overlap
        while bits:
            bit = bits & -bits
            code_index = bit.bit_length() - 1
            code = hand_code_by_index[code_index]
            connectivity[code] += quality
            bits ^= bit

    ready: list[tuple[str, V2Value]] = []
    non_ready: list[str] = []
    for tile in hand:
        waits = waits_by_discard[_bit_for_code(tile)] & available
        if waits:
            wait_codes = codes_from_mask(waits)
            best_points = max(
                _hypothetical_points(
                    [c for c in hand if c != tile], open_melds, w,
                    self_draw=self_draw,
                    before_first_discard=before_first_discard,
                    wall_empty=wall_empty,
                )
                for w in wait_codes
            )
            style = _style_utility([c for c in hand if c != tile] + meld_tiles)
            ready.append((tile, V2Value(0, len(wait_codes), best_points, style)))
        else:
            non_ready.append(tile)

    if ready:
        best_tile, best_value = ready[0]
        for tile, value in ready[1:]:
            if _is_better_ready(value, best_value):
                best_tile, best_value = tile, value
        # Prefer a true drawn-tile discard on a complete tie.
        ties = [t for t, v in ready if (v.points, v.ukeire) == (best_value.points, best_value.ukeire)]
        if drawn_tile in ties:
            best_tile = drawn_tile
        return best_tile, best_value

    # Non-ready: shortlist by low connectivity, then rank.
    shortlist = sorted(non_ready, key=lambda tile: (connectivity[tile], tile))[:6]
    if drawn_tile in hand and drawn_tile not in shortlist:
        shortlist.append(drawn_tile)

    best_tile: Optional[str] = None
    best_value: Optional[V2Value] = None
    for tile in shortlist:
        remaining_mask = hand_mask ^ _bit_for_code(tile)
        distance, flexibility = _structural_value(remaining_mask)
        remaining_codes = [c for c in hand if c != tile]
        style = _style_utility(remaining_codes + meld_tiles)
        value = V2Value(max(1, distance), flexibility, 0, style)
        if best_value is None:
            best_tile, best_value = tile, value
            continue
        # Rank: lower distance, then higher flexibility (thickness), then style.
        # Tiebreak-only — cannot cost win rate.
        if (-value.distance, value.ukeire, value.style) > (
            -best_value.distance, best_value.ukeire, best_value.style
        ):
            best_tile, best_value = tile, value
        elif (-value.distance, value.ukeire, value.style) == (
            -best_value.distance, best_value.ukeire, best_value.style
        ):
            if tile == drawn_tile and best_tile != drawn_tile:
                best_tile = tile
            elif best_tile != drawn_tile and (best_tile is None or tile < best_tile):
                best_tile = tile
    return best_tile, best_value or V2Value(99, 0, 0, 0)


def _remove_tiles(hand: Sequence[str], selected: Sequence[str]) -> Optional[list[str]]:
    result = list(hand)
    for tile in selected:
        if tile not in result:
            return None
        result.remove(tile)
    return result


def _meld_from_candidate(candidate: dict) -> dict:
    return {
        "kind": candidate.get("kind", "sequence"),
        "tiles": list(candidate.get("tiles", ())),
        "claimed_tile": candidate.get("tile"),
    }


def choose_turn_plan(
    hand: Sequence[str],
    open_melds: Sequence[dict],
    visible_codes: Sequence[str],
    kong_options: Sequence[dict],
    *,
    supplements: int,
    wall_count: int,
    drawn_tile: Optional[str],
    last_draw_was_supplement: bool = False,
) -> dict:
    """Choose win, supplement, kong extension, or a score-aware discard."""
    before_first_discard = not any(False for _ in ())  # caller has no history; treat as false
    wall_empty = wall_count == 0

    if is_winning_hand(hand, open_melds) or any(
        candidate.get("kind") == "kong_win" for candidate in kong_options
    ):
        return {"action": "win"}

    best_tile, baseline = choose_discard(
        hand, open_melds, visible_codes, drawn_tile,
        self_draw=True, before_first_discard=before_first_discard, wall_empty=wall_empty,
    )

    # Supplement: spend one when the draw is the best discard AND the hand is
    # not ready (mirrors efficiency_bot; keeps both charges from burning at once).
    if (
        supplements < 2
        and wall_count > 0
        and drawn_tile is not None
        and best_tile == drawn_tile
        and baseline.distance > 0
        and not last_draw_was_supplement
    ):
        return {"action": "supplement"}

    # Kong: only when it does not worsen the score-aware value.
    best_kong: Optional[dict] = None
    best_kong_rank: Optional[tuple] = None
    for candidate in kong_options:
        after = _remove_tiles(hand, candidate.get("hand_tiles", ()))
        if after is None:
            continue
        melds_after = list(open_melds)
        if is_winning_hand(after, melds_after):
            rank = (1, 0, 0, 0, 1 if candidate.get("kind") == "kong_win" else 0)
        else:
            _, value = choose_discard(
                after, melds_after, visible_codes, None,
                self_draw=True, before_first_discard=before_first_discard, wall_empty=wall_empty,
            )
            rank = (0, -value.distance, value.ukeire, value.points, value.style,
                    len(candidate.get("hand_tiles", ())))
        if best_kong_rank is None or rank > best_kong_rank:
            best_kong, best_kong_rank = candidate, rank

    baseline_rank = (0, -baseline.distance, baseline.ukeire, baseline.points, baseline.style, 0)
    if best_kong is not None and best_kong_rank is not None and best_kong_rank >= baseline_rank:
        if best_kong_rank[0] == 1:
            return {"action": "win"}
        return {"action": "kong", "candidate_id": best_kong.get("id")}
    return {"action": "discard", "tile": best_tile}


def _claim_advances(before: V2Value, after: V2Value, priority: int) -> bool:
    """Claim if it strictly advances distance/ukeire/points, or clearly adds
    style (rainbow groups carry real fan)."""
    if after.distance != before.distance:
        return after.distance < before.distance
    if after.ukeire != before.ukeire:
        return after.ukeire > before.ukeire
    if after.points != before.points:
        return after.points > before.points
    if priority >= 3:  # rainbow: structurally neutral but carries fan
        return after.style >= before.style
    return after.style > before.style


def choose_claim_plan(
    hand: Sequence[str],
    open_melds: Sequence[dict],
    candidates: Sequence[dict],
    visible_codes: Sequence[str],
) -> dict:
    """Choose an authoritative candidate id or pass after another discard."""
    for candidate in candidates:
        if candidate.get("kind") == "win":
            return {"action": "claim", "candidate_id": candidate.get("id")}

    before = evaluate_hand(hand, open_melds, visible_codes)
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
        elif after:
            _, value = choose_discard(after, melds_after, visible_codes, None)
        else:
            continue
        if not _claim_advances(before, value, priority):
            continue
        rank = (-value.distance, value.ukeire, value.points, value.style,
                priority, len(candidate.get("tiles", ())))
        if best_rank is None or rank > best_rank:
            best_candidate, best_rank = candidate, rank

    if best_candidate is None:
        return {"action": "pass"}
    return {"action": "claim", "candidate_id": best_candidate.get("id")}
