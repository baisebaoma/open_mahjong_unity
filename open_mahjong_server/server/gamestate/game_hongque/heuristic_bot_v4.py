"""Hongque heuristic v4 bot (user_id=3): v3 defense + own-value awareness.

Two changes on top of ``heuristic_bot_v3`` (see HONGQUE_HEURISTIC_RESEARCH.md §6.1,
Xelnaga's feedback):

- **Own-value-aware threat gate**: the threat level is discounted by the hand's
  own win-value potential (``_style_utility``, a cheap doing-direction proxy).
  A high-potential hand (e.g. a colour-concentrated shape) keeps attacking; a
  scattered low-value hand folds harder.  This mirrors "remaining draws ×
  expected points" without a binary all-or-nothing switch.
- **End-of-wall forced genbutsu**: when the wall is nearly empty (<= 2 tiles)
  and defense is engaged, discard the globally safest tile regardless of the
  distance budget.  With a single-copy deck a genbutsu tile is 100% safe.

Tenpai-wait selection, claims, supplements and kong stay identical to v3.
This module is deliberately import/mutation-free like ``efficiency_bot``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from .heuristic_bot import (
    V2Value,
    _bit_for_code,
    _claim_advances,
    _hypothetical_points,
    _is_better_ready,
    _meld_from_candidate,
    _meld_tiles,
    _remove_tiles,
    _structural_value,
    _style_utility,
    _visible_mask,
    evaluate_hand,
    _STYLE_CAP,
)
from .heuristic_bot_v3 import (
    _DEFENSE_THREAT,
    _STRONG_DEFENSE_THREAT,
    _WALL_START,
    OpponentView,
    _threat_level,
    _tile_danger,
)
from .group_index import (
    FULL_DECK_MASK,
    GROUP_MASKS,
    TILE_INDEX,
    codes_from_mask,
    mask_from_codes,
    waiting_masks_after_discards,
)
from .tile import HongqueTile
from .win_check import is_winning_hand

# A high style score halves the threat; a zero-value hand keeps it intact.
_VALUE_WEIGHT = 0.5
# End-of-wall window: force genbutsu within the last two wall draws.
_END_WALL = 2


def choose_discard(
    hand: Sequence[str],
    open_melds: Sequence[dict],
    visible_codes: Iterable[str],
    drawn_tile: Optional[str] = None,
    *,
    self_draw: bool = False,
    before_first_discard: bool = False,
    wall_empty: bool = False,
    opponents: Sequence[OpponentView] = (),
    wall_count: int = _WALL_START,
) -> tuple[Optional[str], V2Value]:
    if not hand:
        return None, V2Value(99, 0, 0, 0)

    hand_mask = mask_from_codes(hand)
    visible_mask = _visible_mask(visible_codes) | hand_mask
    has_open_group = bool(open_melds)
    hand_code_by_index = {TILE_INDEX[tile]: tile for tile in hand}
    meld_tiles = _meld_tiles(open_melds)

    waits_by_discard = waiting_masks_after_discards(
        hand_mask,
        used_mask=visible_mask,
        has_open_group=has_open_group,
    )
    available = FULL_DECK_MASK & ~visible_mask

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
        ties = [t for t, v in ready if (v.points, v.ukeire) == (best_value.points, best_value.ukeire)]
        if drawn_tile in ties:
            best_tile = drawn_tile
        return best_tile, best_value

    threat = _threat_level(opponents, wall_count)
    if threat < _DEFENSE_THREAT:
        # Pure offense: identical ranking to the v2 heuristic.
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

    # Defense engaged: own-value discount then end-of-wall genbutsu.
    style = _style_utility(list(hand) + meld_tiles)
    value_factor = 1.0 - _VALUE_WEIGHT * min(1.0, style / _STYLE_CAP)
    threat *= value_factor

    distances = {
        tile: _structural_value(hand_mask ^ _bit_for_code(tile))[0]
        for tile in hand
    }
    min_distance = min(distances.values())

    if wall_count <= _END_WALL:
        # End of wall: force the globally safest discard (genbutsu first).
        best_tile = min(non_ready, key=lambda t: _tile_danger(t, opponents))
        return best_tile, V2Value(max(1, distances[best_tile]), 0, 0, 0)

    if threat < _DEFENSE_THREAT or min_distance < 2:
        # Close hand (or discount dropped below the gate): safety is a tie-break
        # after distance and flexibility.
        shortlist = sorted(non_ready, key=lambda tile: (connectivity[tile], tile))[:6]
        if drawn_tile in hand and drawn_tile not in shortlist:
            shortlist.append(drawn_tile)
        best_tile: Optional[str] = None
        best_key: Optional[tuple] = None
        for tile in shortlist:
            remaining_mask = hand_mask ^ _bit_for_code(tile)
            distance, flexibility = _structural_value(remaining_mask)
            style = _style_utility([c for c in hand if c != tile] + meld_tiles)
            danger = _tile_danger(tile, opponents)
            key = (-distance, flexibility, -danger, style, tile)
            if best_key is None or key > best_key:
                best_tile, best_key = tile, key
        if best_tile is None:
            return non_ready[0], V2Value(99, 0, 0, 0)
        distance = _structural_value(hand_mask ^ _bit_for_code(best_tile))[0]
        return best_tile, V2Value(max(1, distance), 0, 0, 0)

    # Far hand: fold within a small distance budget toward the safest discard.
    # Every tile is a candidate (a safe tile may be a "good" shape tile).
    slack = 2 if threat >= _STRONG_DEFENSE_THREAT else 1
    best_tile: Optional[str] = None
    best_key: Optional[tuple] = None
    for tile in hand:
        distance = distances[tile]
        if distance > min_distance + slack:
            continue
        remaining_mask = hand_mask ^ _bit_for_code(tile)
        _, flexibility = _structural_value(remaining_mask)
        danger = _tile_danger(tile, opponents)
        style = _style_utility([c for c in hand if c != tile] + meld_tiles)
        key = (distance - min_distance, danger, -flexibility, -style, tile)
        if best_key is None or key < best_key:
            best_tile, best_key = tile, key
    if best_tile is None:
        return non_ready[0], V2Value(99, 0, 0, 0)
    flexibility = -best_key[2]
    style = -best_key[3]
    return best_tile, V2Value(max(1, distances[best_tile]), int(flexibility), 0, style)


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
    opponents: Sequence[OpponentView] = (),
) -> dict:
    """Choose win, supplement, kong extension, or a defensive discard."""
    before_first_discard = not any(False for _ in ())  # caller has no history; treat as false
    wall_empty = wall_count == 0

    if is_winning_hand(hand, open_melds) or any(
        candidate.get("kind") == "kong_win" for candidate in kong_options
    ):
        return {"action": "win"}

    best_tile, baseline = choose_discard(
        hand, open_melds, visible_codes, drawn_tile,
        self_draw=True, before_first_discard=before_first_discard, wall_empty=wall_empty,
        opponents=opponents, wall_count=wall_count,
    )

    if (
        supplements < 2
        and wall_count > 0
        and drawn_tile is not None
        and best_tile == drawn_tile
        and baseline.distance > 0
        and not last_draw_was_supplement
    ):
        return {"action": "supplement"}

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
                opponents=opponents, wall_count=wall_count,
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
            _, value = choose_discard(after, melds_after, visible_codes, None,
                                      opponents=(), wall_count=_WALL_START)
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
